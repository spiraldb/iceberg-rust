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

mod cache;

use std::any::Any;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::Poll;

use async_trait::async_trait;
pub use cache::VortexScanCache;
use futures::{StreamExt, stream};
use vortex::array::stats::StatsSet;
use vortex::array::stream::{ArrayStreamAdapter, SendableArrayStream};
use vortex::arrow::ArrowSessionExt;
use vortex::dtype::{DType, FieldPath, Nullability};
use vortex::error::{VortexResult, vortex_bail};
use vortex::expr::stats::Precision;
use vortex::expr::transform::replace;
use vortex::expr::{BoundExpression, Expression, and_collect, cast, get_item, lit, pack, root};
use vortex::scalar::Scalar;
use vortex::scan::selection::Selection;
use vortex::scan::{
    DataSource, DataSourceScan, DataSourceScanRef, Partition, PartitionRef, PartitionStream,
    ScanRequest,
};
use vortex::session::VortexSession;

use super::{byte_range_to_rows, convert_predicate_to_vortex};
use crate::arrow::vortex_util::VORTEX_SCHEMA_KEY;
use crate::arrow::{schema_to_arrow_schema, to_iceberg_error};
use crate::io::FileIO;
use crate::metadata_columns::is_metadata_field;
use crate::scan::FileScanTask;
use crate::spec::{DataFileFormat, PrimitiveType, Schema, SchemaRef, Type};
use crate::{Result, Runtime};

/// Scans supported Iceberg Vortex tasks into Vortex arrays.
///
/// Row ranges and selections use row ordinals relative to each source partition.
/// Partition ranges and selections address this source's stable partition indices.
pub struct VortexScanSource {
    dtype: DType,
    tasks: Vec<Arc<PreparedTask>>,
    bytes_read: Arc<AtomicU64>,
}

impl VortexScanSource {
    /// Prepares all tasks, returning `None` if the whole scan needs the Arrow reader.
    ///
    /// Supports top-level primitive fields, renames, safe primitive promotions, and missing
    /// optional fields without defaults. Deletes, partition constants, and metadata columns
    /// need the Arrow reader.
    /// All file schemas are checked before this method returns a usable source.
    /// The caller must keep the current Tokio runtime alive while the source is in use.
    pub async fn try_new(
        file_io: FileIO,
        schema: SchemaRef,
        tasks: Vec<FileScanTask>,
        session: VortexSession,
        target_parallelism: NonZeroUsize,
    ) -> Result<Option<Self>> {
        let cache = VortexScanCache::new(file_io, session, Runtime::try_current()?);
        Self::try_new_with_cache(&cache, schema, tasks, target_parallelism).await
    }

    /// Prepares a scan while reusing immutable file handles from `cache`.
    ///
    /// The same supported-task checks as [`Self::try_new`] apply. Schemas, defaults,
    /// predicates, and ranges are validated for this scan, independently of cached file handles.
    pub async fn try_new_with_cache(
        cache: &VortexScanCache,
        schema: SchemaRef,
        tasks: Vec<FileScanTask>,
        target_parallelism: NonZeroUsize,
    ) -> Result<Option<Self>> {
        let session = cache.session();
        if schema.as_struct().fields().iter().any(|field| {
            !matches!(field.field_type.as_ref(), Type::Primitive(_)) || is_metadata_field(field.id)
        }) || tasks.iter().any(|task| {
            task.data_file_format() != DataFileFormat::Vortex
                || !task.deletes().is_empty()
                || task.key_metadata().is_some()
                || task
                    .partition_spec()
                    .is_some_and(|spec| !spec.fields().is_empty())
                || task.schema() != schema.as_ref()
                || task
                    .project_field_ids()
                    .iter()
                    .any(|id| is_metadata_field(*id))
        }) {
            return Ok(None);
        }
        let Some(dtype) = array_source_dtype(&schema, session) else {
            return Ok(None);
        };
        let bytes_read = Arc::clone(cache.bytes_read_counter());
        let mut prepared = Vec::with_capacity(tasks.len());
        // A partially supported snapshot must never yield rows before the caller falls back.
        for task in tasks {
            let (file, file_size) = cache
                .open(task.data_file_path(), task.file_size_in_bytes())
                .await?;
            let file_schema = match file.metadata_segment(VORTEX_SCHEMA_KEY) {
                Some(bytes) => serde_json::from_slice::<Schema>(bytes.as_slice())?,
                None => {
                    let arrow_schema = Arc::new(
                        session
                            .arrow()
                            .to_arrow_schema(file.dtype())
                            .map_err(to_iceberg_error)?,
                    );
                    let arrow_schema = if let Some(mapping) = task.name_mapping() {
                        super::super::apply_name_mapping_to_arrow_schema(arrow_schema, mapping)?
                    } else {
                        super::super::add_fallback_field_ids_to_arrow_schema(&arrow_schema)
                    };
                    Schema::try_from(arrow_schema.as_ref())?
                }
            };
            let Some(expected_file_dtype) = array_source_dtype(&file_schema, session) else {
                return Ok(None);
            };
            if file.dtype() != &expected_file_dtype {
                return Ok(None);
            }
            let Some(normalize) = normalization(&schema, &file_schema, file.dtype(), &dtype)?
            else {
                return Ok(None);
            };
            let predicate = task
                .predicate()
                .map(|predicate| {
                    // Directly built or deserialized tasks may bypass TableScan's NOT rewrite.
                    // A conservative missing-field match must not be negated into a false match.
                    let predicate = predicate.clone().rewrite_not();
                    convert_predicate_to_vortex(&predicate, &file_schema, file.dtype())
                })
                .transpose()?;
            let rows = if task.start() == 0 && task.length() == 0 {
                0..file.row_count()
            } else {
                byte_range_to_rows(task.start(), task.length(), file_size, file.row_count())
            };
            prepared.push(Arc::new(PreparedTask {
                file,
                normalize,
                predicate,
                rows,
                bytes: if task.length() == 0 {
                    file_size
                } else {
                    task.length()
                },
            }));
        }
        let rows = prepared.iter().fold(0u64, |rows, task| {
            rows.saturating_add(task.rows.end - task.rows.start)
        });
        let target_partitions = target_parallelism.get().saturating_mul(2) as u64;
        let partition_rows = rows.div_ceil(target_partitions).max(1);
        let tasks = prepared
            .into_iter()
            .flat_map(|task| {
                task_ranges(task.rows.clone(), task.bytes, partition_rows)
                    .into_iter()
                    .map(move |(rows, bytes)| {
                        Arc::new(PreparedTask {
                            file: Arc::clone(&task.file),
                            normalize: task.normalize.clone(),
                            predicate: task.predicate.clone(),
                            rows,
                            bytes,
                        })
                    })
            })
            .collect();
        Ok(Some(Self {
            dtype,
            tasks,
            bytes_read,
        }))
    }

    /// Returns cumulative bytes read through this source's cache, including file metadata.
    ///
    /// Sources prepared with the same [`VortexScanCache`] share this counter. It includes reads
    /// from concurrent and previous scans, and is not a per-query IO measurement.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read.load(Ordering::Relaxed)
    }
}

struct PreparedTask {
    file: Arc<vortex::file::VortexFile>,
    normalize: Expression,
    predicate: Option<Expression>,
    rows: Range<u64>,
    bytes: u64,
}

fn array_source_dtype(schema: &Schema, session: &VortexSession) -> Option<DType> {
    let arrow_schema = schema_to_arrow_schema(schema).ok()?;
    session.arrow().from_arrow_schema(&arrow_schema).ok()
}

#[async_trait]
impl DataSource for VortexScanSource {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> Precision<u64> {
        let Some(rows) = self.tasks.iter().try_fold(0u64, |rows, task| {
            rows.checked_add(task.rows.end - task.rows.start)
        }) else {
            return Precision::Absent;
        };
        if self.tasks.iter().any(|task| task.predicate.is_some()) {
            Precision::Inexact(rows)
        } else {
            Precision::Exact(rows)
        }
    }

    async fn field_statistics(&self, _field: &FieldPath) -> VortexResult<StatsSet> {
        Ok(StatsSet::default())
    }

    async fn scan(&self, request: ScanRequest) -> VortexResult<DataSourceScanRef> {
        if request
            .row_range
            .as_ref()
            .is_some_and(|range| range.start > range.end)
            || request
                .partition_range
                .as_ref()
                .is_some_and(|range| range.start > range.end)
        {
            vortex_bail!("Iceberg Vortex scan range starts after its end")
        }
        let dtype = request
            .projection
            .clone()
            .bind(&self.dtype)?
            .dtype()
            .clone();
        if let Some(filter) = &request.filter
            && !matches!(filter.clone().bind(&self.dtype)?.dtype(), DType::Bool(_))
        {
            vortex_bail!("Iceberg Vortex scan filter must return booleans")
        }
        let remaining = request
            .limit
            .filter(|_| !request.ordered)
            .map(|limit| Arc::new(AtomicU64::new(limit)));
        let partitions =
            self.tasks
                .iter()
                .enumerate()
                .filter(|(index, _)| {
                    request
                        .partition_range
                        .as_ref()
                        .is_none_or(|range| range.contains(&(*index as u64)))
                        && !request
                            .partition_selection
                            .row_mask(&(*index as u64..*index as u64 + 1))
                            .mask()
                            .all_false()
                })
                .map(|(index, task)| {
                    let projection =
                        replace(request.projection.clone(), &root(), task.normalize.clone())
                            .bind(task.file.dtype())?
                            .optimize_recursive()?;
                    if projection.dtype() != &dtype {
                        vortex_bail!("Iceberg Vortex file projection has a different dtype")
                    }
                    let filter =
                        and_collect(task.predicate.iter().cloned().chain(
                            request.filter.iter().map(|filter| {
                                replace(filter.clone(), &root(), task.normalize.clone())
                            }),
                        ))
                        .map(|filter| filter.bind(task.file.dtype())?.optimize_recursive())
                        .transpose()?;
                    Ok(Box::new(TaskPartition {
                        index,
                        task: Arc::clone(task),
                        rows: task_rows(&task.rows, request.row_range.as_ref()),
                        selection: task_selection(&request.selection, &task.rows),
                        projection,
                        filter,
                        ordered: request.ordered,
                        limit: request.limit,
                        remaining: remaining.clone(),
                    }) as PartitionRef)
                })
                .collect::<VortexResult<Vec<_>>>()?;
        Ok(Box::new(TaskScan { dtype, partitions }))
    }
}

struct TaskScan {
    dtype: DType,
    partitions: Vec<PartitionRef>,
}

impl DataSourceScan for TaskScan {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn partition_count(&self) -> Precision<usize> {
        Precision::Exact(self.partitions.len())
    }

    fn partitions(self: Box<Self>) -> PartitionStream {
        stream::iter(self.partitions.into_iter().map(Ok)).boxed()
    }
}

struct TaskPartition {
    index: usize,
    task: Arc<PreparedTask>,
    rows: Range<u64>,
    selection: Selection,
    projection: BoundExpression,
    filter: Option<BoundExpression>,
    ordered: bool,
    limit: Option<u64>,
    remaining: Option<Arc<AtomicU64>>,
}

impl Partition for TaskPartition {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn index(&self) -> usize {
        self.index
    }

    fn row_count(&self) -> Precision<u64> {
        let rows = self.rows.end - self.rows.start;
        if self.task.predicate.is_some()
            || self.filter.is_some()
            || !matches!(self.selection, Selection::All)
            || self.limit.is_some()
        {
            Precision::Inexact(rows)
        } else {
            Precision::Exact(rows)
        }
    }

    fn byte_size(&self) -> Precision<u64> {
        Precision::Inexact(self.task.bytes)
    }

    fn execute(self: Box<Self>) -> VortexResult<SendableArrayStream> {
        let dtype = self.projection.dtype().clone();
        let mut scan = self
            .task
            .file
            .scan()?
            .with_projection(self.projection)
            .with_some_filter(self.filter)
            .with_row_range(self.rows)
            .with_selection(self.selection)
            .with_ordered(self.ordered);
        if let Some(limit) = self.limit {
            scan = scan.with_limit(limit);
        }
        let mut arrays = Box::pin(scan.into_array_stream()?);
        let Some(remaining) = self.remaining else {
            return Ok(arrays);
        };
        let limited = stream::poll_fn(move |cx| {
            if remaining.load(Ordering::Acquire) == 0 {
                return Poll::Ready(None);
            }
            match arrays.poll_next_unpin(cx) {
                Poll::Ready(Some(Ok(array))) => {
                    if array.is_empty() {
                        return Poll::Ready(Some(Ok(array)));
                    }
                    let granted = reserve_rows(&remaining, array.len() as u64) as usize;
                    if granted == 0 {
                        Poll::Ready(None)
                    } else if granted < array.len() {
                        Poll::Ready(Some(array.slice(0..granted)))
                    } else {
                        Poll::Ready(Some(Ok(array)))
                    }
                }
                other => other,
            }
        });
        Ok(Box::pin(ArrayStreamAdapter::new(dtype, limited)))
    }
}

fn reserve_rows(remaining: &AtomicU64, requested: u64) -> u64 {
    let mut rows = remaining.load(Ordering::Acquire);
    loop {
        let granted = rows.min(requested);
        match remaining.compare_exchange_weak(
            rows,
            rows - granted,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => return granted,
            Err(current) => rows = current,
        }
    }
}

fn task_rows(rows: &Range<u64>, requested: Option<&Range<u64>>) -> Range<u64> {
    match requested {
        Some(requested) => {
            rows.start.saturating_add(requested.start).min(rows.end)
                ..rows.start.saturating_add(requested.end).min(rows.end)
        }
        None => rows.clone(),
    }
}

fn task_ranges(rows: Range<u64>, bytes: u64, partition_rows: u64) -> Vec<(Range<u64>, u64)> {
    let row_count = rows.end - rows.start;
    if row_count == 0 {
        return Vec::new();
    }
    let byte_boundary = |row: u64| {
        ((u128::from(row - rows.start) * u128::from(bytes)) / u128::from(row_count)) as u64
    };
    let mut pieces = Vec::new();
    let mut start = rows.start;
    while start < rows.end {
        let end = start.saturating_add(partition_rows).min(rows.end);
        pieces.push((start..end, byte_boundary(end) - byte_boundary(start)));
        start = end;
    }
    pieces
}

fn task_selection(selection: &Selection, rows: &Range<u64>) -> Selection {
    let (include, indices): (bool, Box<dyn Iterator<Item = u64> + '_>) = match selection {
        Selection::All => return Selection::All,
        Selection::IncludeByIndex(indices) => (true, Box::new(indices.iter().copied())),
        Selection::ExcludeByIndex(indices) => (false, Box::new(indices.iter().copied())),
        Selection::IncludeRoaring(indices) => (true, Box::new(indices.iter())),
        Selection::ExcludeRoaring(indices) => (false, Box::new(indices.iter())),
    };
    let selected = indices
        .take_while(|index| *index < rows.end - rows.start)
        .map(|index| rows.start + index)
        .collect();
    if include {
        Selection::IncludeRoaring(selected)
    } else {
        Selection::ExcludeRoaring(selected)
    }
}

fn normalization(
    schema: &Schema,
    file_schema: &Schema,
    file_dtype: &DType,
    dtype: &DType,
) -> Result<Option<Expression>> {
    let Some(target_fields) = dtype.as_struct_fields_opt() else {
        return Ok(None);
    };
    let Some(file_fields) = file_dtype.as_struct_fields_opt() else {
        return Ok(None);
    };
    let mut fields = Vec::with_capacity(schema.as_struct().fields().len());
    for field in schema.as_struct().fields() {
        let target = target_fields.field(&field.name).ok_or_else(|| {
            crate::Error::new(
                crate::ErrorKind::Unexpected,
                "Iceberg Vortex source dtype lost a field",
            )
        })?;
        let expression = match file_schema.as_struct().field_by_id(field.id) {
            Some(file_field) => {
                if (!file_field.required && field.required)
                    || !safe_promotion(file_field.field_type.as_ref(), field.field_type.as_ref())
                {
                    return Ok(None);
                }
                let Some(physical) = file_fields.field(&file_field.name) else {
                    return Ok(None);
                };
                let value = get_item(file_field.name.as_str(), root());
                if physical == target {
                    value
                } else {
                    cast(value, target)
                }
            }
            None if !field.required && field.initial_default.is_none() => lit(Scalar::null(target)),
            None => return Ok(None),
        };
        fields.push((field.name.as_str(), expression));
    }
    let normalized = pack(fields, Nullability::NonNullable);
    if normalized
        .clone()
        .bind(file_dtype)
        .map_err(to_iceberg_error)?
        .dtype()
        != dtype
    {
        return Ok(None);
    }
    Ok(Some(normalized))
}

fn safe_promotion(from: &Type, to: &Type) -> bool {
    from == to
        || matches!(
            (from, to),
            (
                Type::Primitive(PrimitiveType::Int),
                Type::Primitive(PrimitiveType::Long)
            ) | (
                Type::Primitive(PrimitiveType::Float),
                Type::Primitive(PrimitiveType::Double)
            )
        )
        || matches!((from, to),
            (Type::Primitive(PrimitiveType::Decimal { precision: from_precision, scale: from_scale }),
             Type::Primitive(PrimitiveType::Decimal { precision: to_precision, scale: to_scale }))
            if from_scale == to_scale && from_precision <= to_precision)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    use arrow_array::cast::AsArray;
    use arrow_array::{Int64Array, RecordBatch};
    use futures::TryStreamExt;
    use tempfile::TempDir;
    use vortex::VortexSessionDefault;
    use vortex::array::VortexSessionExecute;
    use vortex::arrow::{ArrowSessionExt, ExecuteArrowExt};
    use vortex::expr::{root, select};
    use vortex::scan::selection::Selection;
    use vortex::scan::{DataSource, ScanRequest};
    use vortex::session::VortexSession;

    use super::{
        VortexScanCache, VortexScanSource, array_source_dtype, reserve_rows, task_ranges,
        task_rows, task_selection,
    };
    use crate::Runtime;
    use crate::arrow::schema_to_arrow_schema;
    use crate::expr::{Bind, BoundPredicate, Reference};
    use crate::io::FileIO;
    use crate::scan::FileScanTask;
    use crate::spec::{DataFileFormat, Datum, NestedField, PrimitiveType, Schema, SchemaRef, Type};
    use crate::writer::file_writer::{FileWriter, FileWriterBuilder, VortexWriterBuilder};

    fn source_task(
        path: &str,
        size: u64,
        schema: SchemaRef,
        predicate: Option<BoundPredicate>,
    ) -> FileScanTask {
        FileScanTask::builder()
            .with_file_size_in_bytes(size)
            .with_start(0)
            .with_length(0)
            .with_data_file_path(path.to_owned())
            .with_data_file_format(DataFileFormat::Vortex)
            .with_schema(schema)
            .with_case_sensitive(true)
            .with_project_field_ids(vec![1])
            .with_predicate(predicate)
            .build()
            .unwrap()
    }

    async fn source_fixture() -> (TempDir, FileIO, SchemaRef, FileScanTask) {
        let directory = TempDir::new().unwrap();
        let path = directory.path().join("cached.vortex");
        let path = path.to_str().unwrap();
        let file_io = FileIO::new_with_fs();
        let schema = Arc::new(
            Schema::builder()
                .with_fields([Arc::new(NestedField::required(
                    1,
                    "id",
                    Type::Primitive(PrimitiveType::Long),
                ))])
                .build()
                .unwrap(),
        );
        let batch = RecordBatch::try_new(Arc::new(schema_to_arrow_schema(&schema).unwrap()), vec![
            Arc::new(Int64Array::from_iter_values(0..128)),
        ])
        .unwrap();
        let mut writer = VortexWriterBuilder::new(schema.clone(), VortexSession::default())
            .build(file_io.new_output(path).unwrap())
            .await
            .unwrap();
        writer.write(&batch).await.unwrap();
        let file = writer
            .close()
            .await
            .unwrap()
            .pop()
            .unwrap()
            .partition_spec_id(0)
            .build()
            .unwrap();
        let task = source_task(path, file.file_size_in_bytes(), schema.clone(), None);
        (directory, file_io, schema, task)
    }

    async fn source_ids(source: &VortexScanSource, session: &VortexSession) -> Vec<i64> {
        let scan = source
            .scan(ScanRequest {
                projection: select(["id"], root()),
                ..Default::default()
            })
            .await
            .unwrap();
        let field = session.arrow().to_arrow_field("", scan.dtype()).unwrap();
        let partitions = scan.partitions().try_collect::<Vec<_>>().await.unwrap();
        let mut context = session.create_execution_ctx();
        let mut ids = Vec::new();
        for partition in partitions {
            let arrays = partition
                .execute()
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .unwrap();
            for array in arrays {
                let array = array.execute_arrow(Some(&field), &mut context).unwrap();
                let batch = RecordBatch::from(array.as_struct());
                ids.extend(
                    batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap()
                        .values()
                        .iter()
                        .copied(),
                );
            }
        }
        ids.sort_unstable();
        ids
    }

    #[tokio::test]
    async fn cached_preparation_reuses_metadata_and_keeps_predicates_query_local() {
        let (_directory, file_io, schema, task) = source_fixture().await;
        let cache = VortexScanCache::new(file_io, VortexSession::default(), Runtime::current());
        let parallelism = NonZeroUsize::new(4).unwrap();
        let source = VortexScanSource::try_new_with_cache(
            &cache,
            schema.clone(),
            vec![task.clone()],
            parallelism,
        )
        .await
        .unwrap()
        .unwrap();
        let metadata_bytes = cache.bytes_read();
        assert!(metadata_bytes > 0);
        let predicate = Reference::new("id")
            .greater_than(Datum::long(120))
            .bind(schema.clone(), true)
            .unwrap();
        let filtered_task = source_task(
            task.data_file_path(),
            task.file_size_in_bytes(),
            schema.clone(),
            Some(predicate),
        );
        let filtered =
            VortexScanSource::try_new_with_cache(&cache, schema, vec![filtered_task], parallelism)
                .await
                .unwrap()
                .unwrap();
        assert_eq!(cache.bytes_read(), metadata_bytes);
        assert_eq!(
            source_ids(&source, cache.session()).await,
            (0..128).collect::<Vec<_>>()
        );
        assert_eq!(
            source_ids(&filtered, cache.session()).await,
            (121..128).collect::<Vec<_>>()
        );
        assert_eq!(source.bytes_read(), cache.bytes_read());
        assert_eq!(filtered.bytes_read(), cache.bytes_read());
    }

    #[tokio::test]
    async fn concurrent_cached_preparation_coalesces_file_opens() {
        let (_directory, file_io, schema, task) = source_fixture().await;
        let session = VortexSession::default();
        let single_cache =
            VortexScanCache::new(file_io.clone(), session.clone(), Runtime::current());
        let parallelism = NonZeroUsize::new(4).unwrap();
        VortexScanSource::try_new_with_cache(
            &single_cache,
            schema.clone(),
            vec![task.clone()],
            parallelism,
        )
        .await
        .unwrap()
        .unwrap();
        let cache = VortexScanCache::new(file_io, session, Runtime::current());
        let (first, second) = futures::try_join!(
            VortexScanSource::try_new_with_cache(
                &cache,
                schema.clone(),
                vec![task.clone()],
                parallelism
            ),
            VortexScanSource::try_new_with_cache(&cache, schema, vec![task], parallelism),
        )
        .unwrap();
        assert_eq!(cache.bytes_read(), single_cache.bytes_read());
        for source in [first.unwrap(), second.unwrap()] {
            assert_eq!(
                source_ids(&source, cache.session()).await,
                (0..128).collect::<Vec<_>>()
            );
        }
    }

    #[tokio::test]
    async fn directly_built_tasks_rewrite_not_before_matching_missing_fields() {
        let (_directory, file_io, schema, task) = source_fixture().await;
        let evolved = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(schema.as_struct().fields().iter().cloned())
                .with_fields([Arc::new(NestedField::optional(
                    2,
                    "added",
                    Type::Primitive(PrimitiveType::Long),
                ))])
                .build()
                .unwrap(),
        );
        // Bind directly rather than passing through TableScan, leaving an actual NOT node
        // around a field absent from the physical file. Added optional fields read as NULL.
        let predicate = (!Reference::new("added").is_not_null())
            .bind(evolved.clone(), true)
            .unwrap();
        assert!(matches!(&predicate, BoundPredicate::Not(_)));
        let task = source_task(
            task.data_file_path(),
            task.file_size_in_bytes(),
            evolved.clone(),
            Some(predicate),
        );
        let parallelism = NonZeroUsize::new(4).unwrap();
        let session = VortexSession::default();
        let fresh = VortexScanSource::try_new(
            file_io.clone(),
            evolved.clone(),
            vec![task.clone()],
            session.clone(),
            parallelism,
        )
        .await
        .unwrap()
        .unwrap();
        let cache = VortexScanCache::new(file_io, session, Runtime::current());
        let cached = VortexScanSource::try_new_with_cache(&cache, evolved, vec![task], parallelism)
            .await
            .unwrap()
            .unwrap();
        for source in [fresh, cached] {
            assert_eq!(
                source_ids(&source, cache.session()).await,
                (0..128).collect::<Vec<_>>()
            );
        }
    }

    #[tokio::test]
    async fn unsupported_historical_fields_require_the_arrow_reader() {
        let visible = Arc::new(NestedField::optional(
            1,
            "visible",
            Type::Primitive(PrimitiveType::Long),
        ));
        let current = Schema::builder()
            .with_fields([Arc::clone(&visible)])
            .build()
            .unwrap();
        let historical = Schema::builder()
            .with_fields([
                visible,
                Arc::new(NestedField::optional(
                    2,
                    "dropped",
                    Type::Primitive(PrimitiveType::Geometry(Default::default())),
                )),
            ])
            .build()
            .unwrap();
        let session = VortexSession::default();
        assert_eq!(
            [
                array_source_dtype(&current, &session).is_some(),
                array_source_dtype(&historical, &session).is_some(),
            ],
            [true, false]
        );
    }

    #[test]
    fn split_ranges_preserve_rows_and_bytes() {
        assert_eq!(task_ranges(10..21, 17, 4), vec![
            (10..14, 6),
            (14..18, 6),
            (18..21, 5)
        ]);
        assert_eq!(task_ranges(10..10, 17, 4), vec![]);
        assert_eq!(task_ranges(u64::MAX - 4..u64::MAX, u64::MAX, 3), vec![
            (u64::MAX - 4..u64::MAX - 1, 13_835_058_055_282_163_711),
            (u64::MAX - 1..u64::MAX, 4_611_686_018_427_387_904),
        ]);
    }

    #[test]
    fn requested_ranges_are_relative_to_the_source_partition() {
        assert_eq!(task_rows(&(100..120), Some(&(2..8))), 102..108);
        assert_eq!(task_rows(&(100..120), Some(&(18..u64::MAX))), 118..120);
        assert_eq!(task_rows(&(100..120), Some(&(30..u64::MAX))), 120..120);
    }

    #[test]
    fn selections_use_source_partition_positions() {
        let selected = task_selection(
            &Selection::IncludeRoaring([0, 4, 9, 10].into_iter().collect()),
            &(100..110),
        );
        let Selection::IncludeRoaring(selected) = selected else {
            panic!("include selection");
        };
        assert_eq!(selected.iter().collect::<Vec<_>>(), vec![100, 104, 109]);
        let selected = task_selection(
            &Selection::ExcludeRoaring([0, 4, 20].into_iter().collect()),
            &(100..110),
        );
        let Selection::ExcludeRoaring(selected) = selected else {
            panic!("exclude selection");
        };
        assert_eq!(selected.iter().collect::<Vec<_>>(), vec![100, 104]);
    }

    #[test]
    fn concurrent_partitions_share_one_limit() {
        let remaining = Arc::new(AtomicU64::new(101));
        let granted = std::thread::scope(|scope| {
            (0..16)
                .map(|_| {
                    let remaining = Arc::clone(&remaining);
                    scope.spawn(move || reserve_rows(&remaining, 20))
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|thread| thread.join().unwrap())
                .sum::<u64>()
        });
        assert_eq!((granted, remaining.load(Ordering::Relaxed)), (101, 0));
        assert_eq!(reserve_rows(&remaining, 1), 0);
    }
}
