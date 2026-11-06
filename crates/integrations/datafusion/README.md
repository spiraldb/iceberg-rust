# Iceberg DataFusion provider

`IcebergTableProvider` scans a loaded Iceberg snapshot using Parquet data files.
It implements the standard DataFusion 55 `TableProvider` interface.

```rust,no_run
use std::sync::Arc;
use datafusion::common::Result;
use datafusion::prelude::{SessionConfig, SessionContext};
use iceberg::table::Table;
use iceberg_datafusion::IcebergTableProvider;

fn context(table: Table) -> Result<SessionContext> {
    let context = SessionContext::new_with_config(
        SessionConfig::new().with_target_partitions(8),
    );
    context.register_table("events", Arc::new(IcebergTableProvider::try_new(table)?))?;
    Ok(context)
}
```

The caller keeps the table's Tokio runtime alive. A provider retains the loaded
table's schema and snapshot; construct a new provider to observe catalog updates.

Top-level string and binary columns use Arrow view arrays. Nested columns and
extension storage keep their Iceberg Arrow types.

Scans expose the session's target partition count, with a minimum of two. File
byte ranges are balanced across consumers and own disjoint Parquet row groups.
Each consumer has its own reader stream. Decoding runs on the table's CPU
executor, with bounded buffering per partition and cancellation when a consumer
stops. The table's I/O executor supplies asynchronous I/O and timer drivers.
Small files with one row group can leave some partitions empty.

Projection, schema evolution, and Iceberg positional and equality deletes use
the Iceberg reader. Supported primitive predicates prune manifests and Parquet
row groups; DataFusion retains the complete residual filter. Whole nested
values, initial defaults, and floating comparisons remain residual to preserve
DataFusion semantics. Limits are enforced by DataFusion.

DataFusion's live join and top-k filters can prune rows during Parquet decoding.
Fields are matched by Iceberg ID, with lossless type promotions. Files whose
filter fields cannot be adapted keep the complete DataFusion residual filter.

The default dependency graph excludes Vortex. The provider rejects data-file
formats other than Parquet even if another application enables optional formats
on the core Iceberg crate.

Run the real-file integration and predicate tests with:

```sh
cargo test -p iceberg-datafusion
```
