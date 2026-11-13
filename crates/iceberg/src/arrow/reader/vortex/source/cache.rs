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

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use moka::future::Cache;
use vortex::file::VortexFile;
use vortex::session::VortexSession;

use super::super::open_vortex_file;
use crate::io::FileIO;
use crate::{Error, Result, Runtime};

const MAX_OPEN_FILES: u64 = 128;

/// Reuses immutable Vortex file handles across Iceberg scans.
///
/// Each cache is bound to one [`FileIO`], [`VortexSession`], and [`Runtime`]. Files are identified by their
/// path and declared size; callers must obey Iceberg's immutable data-file contract and must
/// not reuse a cache after changing the storage binding or access authorization. Cloning a
/// cache shares its handles and counters. The cache uses Moka's bounded policy with a capacity
/// of 128 entries; active scans may keep evicted handles alive. Layout readers are created per
/// scan so their expression and pruning caches do not grow across queries. Files are opened on
/// the supplied runtime's IO executor; the caller must keep that runtime alive while cached files
/// are in use, as required by [`Runtime`].
#[derive(Clone)]
pub struct VortexScanCache {
    file_io: FileIO,
    session: VortexSession,
    runtime: Runtime,
    files: Cache<(String, u64), (Arc<VortexFile>, u64)>,
    bytes_read: Arc<AtomicU64>,
}

impl fmt::Debug for VortexScanCache {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VortexScanCache")
            .field("entry_count", &self.files.entry_count())
            .field("bytes_read", &self.bytes_read())
            .finish_non_exhaustive()
    }
}

impl VortexScanCache {
    /// Creates a cache scoped to this storage binding, Vortex session, and explicit IO runtime.
    ///
    /// The supplied session must dispatch onto the current Tokio runtime (as Vortex's default
    /// session does), or be explicitly configured to use this runtime's IO executor.
    pub fn new(file_io: FileIO, session: VortexSession, runtime: Runtime) -> Self {
        Self {
            file_io,
            session,
            runtime,
            files: Cache::new(MAX_OPEN_FILES),
            bytes_read: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Returns the session used by all cached files.
    pub fn session(&self) -> &VortexSession {
        &self.session
    }

    /// Returns cumulative bytes read through this cache's FileIO, including file metadata.
    ///
    /// This is shared by all scans and cache clones, including reads through handles retained
    /// by active scans after eviction. It is not a per-query IO measurement.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read.load(Ordering::Relaxed)
    }

    pub(super) fn bytes_read_counter(&self) -> &Arc<AtomicU64> {
        &self.bytes_read
    }

    pub(super) async fn open(
        &self,
        path: &str,
        declared_size: u64,
    ) -> Result<(Arc<VortexFile>, u64)> {
        self.files
            .try_get_with((path.to_owned(), declared_size), async {
                let path = path.to_owned();
                let file_io = self.file_io.clone();
                let session = self.session.clone();
                let bytes_read = Arc::clone(&self.bytes_read);
                // Opening a Vortex file spawns a persistent IO driver. Anchor that driver to
                // the explicit cache runtime, independently of the scan's planning runtime.
                let (file, size) = self
                    .runtime
                    .io()
                    .spawn(async move {
                        open_vortex_file(&path, declared_size, None, &file_io, bytes_read, &session)
                            .await
                    })
                    .await??;
                Ok::<_, Error>((Arc::new(file), size))
            })
            .await
            .map_err(|error| {
                Error::new(error.kind(), "Failed to open a cached Vortex file").with_source(error)
            })
    }
}
