// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! External commit coordination for conditional put operations on S3-compatible stores.

use async_trait::async_trait;
use std::fmt::Debug;
use std::time::Duration;

use crate::path::Path;
use crate::Result;

/// Trait for external coordination of conditional put operations.
///
/// Implementations provide mutual exclusion for put operations on a given path,
/// allowing conditional semantics on stores that do not natively support them
/// (e.g. S3-compatible stores without `If-Match` / `If-None-Match`).
#[async_trait]
pub trait PutCommit: Send + Sync + Debug {
    /// Acquire an exclusive lock for the given path.
    ///
    /// Returns a [`CommitLock`] containing a deadline by which the caller must
    /// complete its operation. The lock is released when the guard is dropped.
    async fn lock(&self, path: &Path) -> Result<CommitLock>;
}

/// An acquired lock with a deadline.
///
/// The caller must complete the guarded operation before [`Self::deadline`].
/// If the operation exceeds the deadline, the lock may have expired and
/// another writer could proceed concurrently — so the caller should abort.
///
/// The lock is released when this value is dropped.
pub struct CommitLock {
    /// The tokio instant by which the operation must complete.
    pub deadline: tokio::time::Instant,
    /// The lock guard; dropping it releases the lock.
    _guard: Box<dyn Send>,
}

impl Debug for CommitLock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommitLock")
            .field("deadline", &self.deadline)
            .finish_non_exhaustive()
    }
}

impl CommitLock {
    /// Create a new `CommitLock` with the given deadline and guard.
    pub fn new(deadline: tokio::time::Instant, guard: Box<dyn Send>) -> Self {
        Self {
            deadline,
            _guard: guard,
        }
    }
}

/// Lua script for atomic check-and-delete: only deletes the key if its value
/// matches the caller's token, preventing one client from releasing another's lock.
const UNLOCK_SCRIPT: &str =
    "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end";

/// Default lock TTL (safety net for crash recovery).
const DEFAULT_LOCK_TTL: Duration = Duration::from_secs(300);

/// Safety margin subtracted from the lock TTL to produce the operation deadline.
/// Accounts for network round-trip, scheduling delays, and clock skew.
const DEADLINE_SAFETY_MARGIN: Duration = Duration::from_secs(10);

/// Default timeout for acquiring a lock before returning an error.
const DEFAULT_ACQUIRE_TIMEOUT: Duration = Duration::from_secs(60);

/// Interval between retry attempts when the lock is held by another client.
const RETRY_INTERVAL: Duration = Duration::from_millis(50);

/// A [`PutCommit`] implementation backed by [Redis](https://redis.io/).
///
/// Uses `SET key token NX PX ttl` for distributed locking. The lock is released
/// via a Lua script that atomically checks the token before deleting, ensuring
/// only the lock holder can release it. A TTL provides automatic expiry as a
/// safety net in case the holder crashes.
///
/// The operation must complete within the lock TTL. If it doesn't, the caller
/// receives a timeout error to prevent a race with another writer.
///
/// # Example
///
/// ```no_run
/// # use object_store::aws::commit::RedisCommit;
/// let commit = RedisCommit::new("redis://localhost:6379").unwrap();
/// ```
#[derive(Debug)]
pub struct RedisCommit {
    client: redis::Client,
    key_prefix: String,
    lock_ttl: Duration,
    acquire_timeout: Duration,
}

impl RedisCommit {
    /// Create a new `RedisCommit` connected to the given Redis URL.
    ///
    /// The URL should be in the format `redis://[user:password@]host:port[/db]`
    /// or `rediss://...` for TLS. No connection is opened until [`lock`](PutCommit::lock)
    /// is called.
    pub fn new(url: impl redis::IntoConnectionInfo) -> Result<Self> {
        let client = redis::Client::open(url).map_err(|e| crate::Error::Generic {
            store: "RedisCommit",
            source: format!("Failed to create Redis client: {e}").into(),
        })?;
        Ok(Self {
            client,
            key_prefix: "s3-lock".into(),
            lock_ttl: DEFAULT_LOCK_TTL,
            acquire_timeout: DEFAULT_ACQUIRE_TIMEOUT,
        })
    }

    /// Set the key prefix used for lock keys in Redis.
    ///
    /// Lock keys are formatted as `{prefix}:{hex_encoded_path}`.
    /// Defaults to `"s3-lock"`.
    #[must_use]
    pub fn with_key_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.key_prefix = prefix.into();
        self
    }

    /// Set the lock TTL (time-to-live).
    ///
    /// This controls both the Redis key expiry and the operation deadline.
    /// If the lock holder crashes, the lock automatically expires after this
    /// duration. Operations that exceed this duration are aborted.
    /// Defaults to 30 seconds.
    #[must_use]
    pub fn with_lock_ttl(mut self, ttl: Duration) -> Self {
        self.lock_ttl = ttl;
        self
    }

    /// Set the maximum time to wait when acquiring a lock.
    ///
    /// If the lock cannot be acquired within this duration, an error is returned.
    /// Defaults to 60 seconds.
    #[must_use]
    pub fn with_acquire_timeout(mut self, timeout: Duration) -> Self {
        self.acquire_timeout = timeout;
        self
    }
}

#[async_trait]
impl PutCommit for RedisCommit {
    async fn lock(&self, path: &Path) -> Result<CommitLock> {
        let key = format!("{}:{}", self.key_prefix, hex_encode(path.as_ref().as_bytes()));
        let token = generate_token();

        let mut conn = self
            .client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| crate::Error::Generic {
                store: "RedisCommit",
                source: format!("Failed to connect to Redis: {e}").into(),
            })?;

        let ttl_ms = self.lock_ttl.as_millis() as u64;
        let acquire_deadline = tokio::time::Instant::now() + self.acquire_timeout;

        loop {
            let result: redis::Value = redis::cmd("SET")
                .arg(&key)
                .arg(&token)
                .arg("NX")
                .arg("PX")
                .arg(ttl_ms)
                .query_async(&mut conn)
                .await
                .map_err(|e| crate::Error::Generic {
                    store: "RedisCommit",
                    source: format!("Redis SET error: {e}").into(),
                })?;

            if matches!(result, redis::Value::Okay) {
                let deadline = tokio::time::Instant::now()
                    + self.lock_ttl.saturating_sub(DEADLINE_SAFETY_MARGIN);
                let guard = Box::new(RedisLockGuard { conn, key, token });
                return Ok(CommitLock::new(deadline, guard));
            }

            if tokio::time::Instant::now() >= acquire_deadline {
                return Err(crate::Error::Generic {
                    store: "RedisCommit",
                    source: format!("Timeout acquiring lock for {path}").into(),
                });
            }

            tokio::time::sleep(RETRY_INTERVAL).await;
        }
    }
}

/// Guard that releases a Redis lock on drop.
///
/// Uses a fire-and-forget `tokio::spawn` for the async unlock. The lock's TTL
/// serves as a safety net if the spawn fails (e.g. runtime shutting down).
struct RedisLockGuard {
    conn: redis::aio::MultiplexedConnection,
    key: String,
    token: String,
}

impl Drop for RedisLockGuard {
    fn drop(&mut self) {
        let mut conn = self.conn.clone();
        let key = std::mem::take(&mut self.key);
        let token = std::mem::take(&mut self.token);
        let _ = tokio::spawn(async move {
            let script = redis::Script::new(UNLOCK_SCRIPT);
            let _: std::result::Result<i32, _> =
                script.key(&key).arg(&token).invoke_async(&mut conn).await;
        });
    }
}

/// Generate a random 32-character hex token for lock ownership.
fn generate_token() -> String {
    use rand::Rng;
    let bytes: [u8; 16] = rand::rng().random();
    hex_encode(&bytes)
}

/// Hex-encode a byte slice (lowercase).
fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(char::from_digit((b >> 4) as u32, 16).unwrap());
        s.push(char::from_digit((b & 0xf) as u32, 16).unwrap());
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_encode() {
        assert_eq!(hex_encode(b"foo/bar"), "666f6f2f626172");
        assert_eq!(hex_encode(b""), "");
    }

    #[test]
    fn test_generate_token() {
        let t1 = generate_token();
        let t2 = generate_token();
        assert_eq!(t1.len(), 32);
        assert_eq!(t2.len(), 32);
        assert_ne!(t1, t2);
    }

    #[test]
    fn test_redis_commit_construction() {
        let commit = RedisCommit::new("redis://localhost:6379").unwrap();
        assert_eq!(commit.key_prefix, "s3-lock");
        assert_eq!(commit.lock_ttl, DEFAULT_LOCK_TTL);
        assert_eq!(commit.acquire_timeout, DEFAULT_ACQUIRE_TIMEOUT);
    }

    #[test]
    fn test_redis_commit_builder_methods() {
        let commit = RedisCommit::new("redis://localhost:6379")
            .unwrap()
            .with_key_prefix("my-prefix")
            .with_lock_ttl(Duration::from_secs(10))
            .with_acquire_timeout(Duration::from_secs(5));
        assert_eq!(commit.key_prefix, "my-prefix");
        assert_eq!(commit.lock_ttl, Duration::from_secs(10));
        assert_eq!(commit.acquire_timeout, Duration::from_secs(5));
    }

    #[test]
    fn test_redis_commit_invalid_url() {
        let result = RedisCommit::new("not-a-valid-url");
        assert!(result.is_err());
    }
}
