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

//! Shared helpers for the vortex file format integration.

use vortex::VortexSessionDefault;
use vortex::error::VortexError;
use vortex::session::VortexSession;

use crate::{Error, ErrorKind};

/// Creates a [`VortexSession`] holding the array, layout and runtime
/// registries.
///
/// The session captures the current tokio runtime handle at construction
/// time, so it must be created from within the runtime that will drive the
/// vortex reads and writes.
pub(crate) fn vortex_session() -> VortexSession {
    <VortexSession as VortexSessionDefault>::default()
}

/// Converts a [`VortexError`] into an iceberg [`Error`].
pub(crate) fn to_iceberg_error(err: VortexError) -> Error {
    Error::new(ErrorKind::Unexpected, "Vortex error").with_source(err)
}
