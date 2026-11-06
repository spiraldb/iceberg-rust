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

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::array::{
    ArrayRef, BooleanArray, Float64Array, Int32Array, Int64Array, LargeBinaryArray, StringArray,
    StructArray,
};
use datafusion::arrow::buffer::NullBuffer;
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::catalog::TableProvider;
use datafusion::datasource::MemTable;
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::{
    BinaryExpr, Column, DynamicFilterPhysicalExpr, Literal as PhysicalLiteral,
};
use datafusion::physical_plan::filter_pushdown::{
    ChildFilterPushdownResult, ChildPushdownResult, FilterPushdownPhase, PushedDown,
};
use datafusion::physical_plan::{ExecutionPlanProperties, collect, collect_partitioned};
use datafusion::prelude::{SessionConfig, SessionContext, col, lit};
use datafusion::scalar::ScalarValue;
use futures::{StreamExt, TryStreamExt};
use iceberg::arrow::{arrow_schema_to_schema_auto_assign_ids, schema_to_arrow_schema};
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::metadata_columns::{
    RESERVED_COL_NAME_DELETE_FILE_PATH, RESERVED_COL_NAME_DELETE_FILE_POS,
    RESERVED_FIELD_ID_DELETE_FILE_PATH, RESERVED_FIELD_ID_DELETE_FILE_POS,
};
use iceberg::spec::{
    DataContentType, DataFileBuilder, DataFileFormat, FormatVersion, Literal, ManifestListWriter,
    ManifestWriterBuilder, NestedField, Operation, PrimitiveType, Snapshot, Struct, Summary, Type,
};
use iceberg::table::Table;
use iceberg::transaction::{AddColumn, ApplyTransactionAction, Transaction};
use iceberg::writer::file_writer::{FileWriter, FileWriterBuilder, ParquetWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, MemoryCatalog, NamespaceIdent, Runtime, TableCreation};
use iceberg_datafusion::IcebergTableProvider;
use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use tempfile::TempDir;

const ROWS: i64 = 257;

fn with_dynamic_filter(
    plan: Arc<dyn datafusion::physical_plan::ExecutionPlan>,
    filter: Arc<DynamicFilterPhysicalExpr>,
    context: &SessionContext,
) -> Arc<dyn datafusion::physical_plan::ExecutionPlan> {
    let result = plan
        .handle_child_pushdown_result(
            FilterPushdownPhase::Post,
            ChildPushdownResult {
                parent_filters: vec![ChildFilterPushdownResult {
                    filter,
                    child_results: vec![],
                }],
                self_filters: vec![],
            },
            context.state().config_options(),
        )
        .unwrap();
    assert!(matches!(result.filters.as_slice(), [PushedDown::No]));
    result.updated_node.unwrap()
}

struct Fixture {
    _root: TempDir,
    catalog: MemoryCatalog,
    table: Table,
    batch: RecordBatch,
}

impl Fixture {
    async fn new(write: bool) -> Self {
        Self::new_with_id_type(write, DataType::Int64).await
    }

    async fn new_with_id_type(write: bool, id_type: DataType) -> Self {
        let root = tempfile::tempdir().unwrap();
        let catalog = MemoryCatalogBuilder::default()
            .with_storage_factory(Arc::new(LocalFsStorageFactory))
            .with_runtime(Runtime::current())
            .load(
                "test",
                HashMap::from([(
                    MEMORY_CATALOG_WAREHOUSE.to_owned(),
                    format!("file://{}", root.path().display()),
                )]),
            )
            .await
            .unwrap();
        let namespace = NamespaceIdent::new("test".to_owned());
        catalog
            .create_namespace(&namespace, HashMap::new())
            .await
            .unwrap();
        let schema = arrow_schema_to_schema_auto_assign_ids(&Schema::new(vec![
            Field::new("id", id_type.clone(), id_type == DataType::Int32),
            Field::new("label", DataType::Utf8, true),
            Field::new("flag", DataType::Boolean, true),
            Field::new(
                "payload",
                DataType::Struct(
                    vec![
                        Field::new("code", DataType::Int64, true),
                        Field::new("text", DataType::Utf8, true),
                    ]
                    .into(),
                ),
                true,
            ),
            Field::new("special_float", DataType::Float64, true),
            Field::new("blob", DataType::Binary, true),
        ]))
        .unwrap();
        let mut table = catalog
            .create_table(
                &namespace,
                TableCreation::builder()
                    .name("rows".to_owned())
                    .schema(schema)
                    .build(),
            )
            .await
            .unwrap();
        let arrow_schema =
            Arc::new(schema_to_arrow_schema(table.metadata().current_schema()).unwrap());
        let DataType::Struct(payload_fields) = arrow_schema.field(3).data_type() else {
            unreachable!()
        };
        let ids: ArrayRef = match id_type {
            DataType::Int32 => Arc::new(Int32Array::from_iter(
                (0..ROWS).map(|id| (id % 13 != 0).then_some(id as i32)),
            )),
            DataType::Int64 => Arc::new(Int64Array::from_iter_values(0..ROWS)),
            _ => unreachable!(),
        };
        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
            ids,
            Arc::new(StringArray::from_iter(
                (0..ROWS).map(|id| (id % 11 != 0).then(|| format!("row-{id}"))),
            )),
            Arc::new(BooleanArray::from_iter(
                (0..ROWS).map(|id| (id % 7 != 0).then_some(id % 2 == 0)),
            )),
            Arc::new(StructArray::new(
                payload_fields.clone(),
                vec![
                    Arc::new(Int64Array::from_iter(
                        (0..ROWS).map(|id| (id % 5 != 0).then_some(id * 3)),
                    )),
                    Arc::new(StringArray::from_iter(
                        (0..ROWS).map(|id| (id % 5 != 0).then(|| format!("nested-{id}"))),
                    )),
                ],
                Some(NullBuffer::from_iter((0..ROWS).map(|id| id % 5 != 0))),
            )),
            Arc::new(Float64Array::from(vec![f64::NAN; ROWS as usize])),
            Arc::new(LargeBinaryArray::from_iter(
                (0..ROWS).map(|id| (id % 11 != 0).then(|| id.to_le_bytes().to_vec())),
            )),
        ])
        .unwrap();
        if write {
            let output = table
                .file_io()
                .new_output(format!("{}/rows.parquet", table.metadata().location()))
                .unwrap();
            let mut writer = ParquetWriterBuilder::new(
                WriterProperties::builder()
                    .set_max_row_group_row_count(Some(32))
                    .build(),
                table.metadata().current_schema().clone(),
            )
            .build(output)
            .await
            .unwrap();
            writer.write(&batch).await.unwrap();
            let files: Vec<_> = writer
                .close()
                .await
                .unwrap()
                .into_iter()
                .map(|mut file| {
                    file.content(DataContentType::Data)
                        .partition(Struct::empty())
                        .partition_spec_id(0)
                        .build()
                        .unwrap()
                })
                .collect();
            let tx = Transaction::new(&table);
            table = tx
                .fast_append()
                .add_data_files(files)
                .apply(tx)
                .unwrap()
                .commit(&catalog)
                .await
                .unwrap();
        }
        Self {
            _root: root,
            catalog,
            table,
            batch,
        }
    }

    fn context(&self) -> SessionContext {
        SessionContext::new_with_config(
            SessionConfig::new()
                .with_target_partitions(4)
                .with_batch_size(19),
        )
    }

    fn evolve_field(&mut self, index: usize, update: impl FnOnce(&mut NestedField)) {
        let current = self.table.metadata().current_schema();
        let mut fields = current.as_struct().fields().to_vec();
        let mut field = fields[index].as_ref().clone();
        update(&mut field);
        fields[index] = Arc::new(field);
        let schema_id = current.schema_id() + 1;
        let schema = iceberg::spec::Schema::builder()
            .with_schema_id(schema_id)
            .with_fields(fields)
            .build()
            .unwrap();
        let metadata = self
            .table
            .metadata()
            .clone()
            .into_builder(self.table.metadata_location().map(str::to_owned))
            .add_schema(schema)
            .unwrap()
            .set_current_schema(schema_id)
            .unwrap()
            .build()
            .unwrap()
            .metadata;
        self.table = Table::builder()
            .metadata(metadata)
            .identifier(self.table.identifier().clone())
            .file_io(self.table.file_io().clone())
            .runtime(Runtime::current())
            .build()
            .unwrap();
    }

    async fn add_deletes(&mut self) {
        let table = &self.table;
        let positions = [0, 32, 128, 256];
        let pos_schema =
            Arc::new(Schema::new(vec![
                Field::new(RESERVED_COL_NAME_DELETE_FILE_PATH, DataType::Utf8, false)
                    .with_metadata(HashMap::from([(
                        PARQUET_FIELD_ID_META_KEY.to_owned(),
                        RESERVED_FIELD_ID_DELETE_FILE_PATH.to_string(),
                    )])),
                Field::new(RESERVED_COL_NAME_DELETE_FILE_POS, DataType::Int64, false)
                    .with_metadata(HashMap::from([(
                        PARQUET_FIELD_ID_META_KEY.to_owned(),
                        RESERVED_FIELD_ID_DELETE_FILE_POS.to_string(),
                    )])),
            ]));
        let pos_batch = RecordBatch::try_new(pos_schema, vec![
            Arc::new(StringArray::from_iter_values(std::iter::repeat_n(
                format!("{}/rows.parquet", table.metadata().location()),
                positions.len(),
            ))),
            Arc::new(Int64Array::from_iter_values(positions)),
        ])
        .unwrap();
        let eq_batch =
            RecordBatch::try_new(Arc::new(self.batch.schema().project(&[0]).unwrap()), vec![
                Arc::new(Int64Array::from(vec![31, 97, 255])),
            ])
            .unwrap();
        let parent = table.metadata().current_snapshot().unwrap();
        let sequence = table.metadata().next_sequence_number();
        let snapshot_id = parent.snapshot_id() + 1;
        let mut delete_manifest = ManifestWriterBuilder::new(
            table
                .file_io()
                .new_output(format!(
                    "{}/delete-manifest.avro",
                    table.metadata().location()
                ))
                .unwrap(),
            Some(snapshot_id),
            table.metadata().current_schema().clone(),
            table.metadata().default_partition_spec().as_ref().clone(),
        )
        .build_v2_deletes();
        for (name, batch, content, equality_ids) in [
            (
                "position",
                pos_batch,
                DataContentType::PositionDeletes,
                None,
            ),
            (
                "equality",
                eq_batch,
                DataContentType::EqualityDeletes,
                Some(vec![
                    table
                        .metadata()
                        .current_schema()
                        .field_by_name("id")
                        .unwrap()
                        .id,
                ]),
            ),
        ] {
            let path = self._root.path().join(format!("{name}-delete.parquet"));
            let file = std::fs::File::create(&path).unwrap();
            let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
            let file = DataFileBuilder::default()
                .content(content)
                .file_path(path.to_string_lossy().into_owned())
                .file_format(DataFileFormat::Parquet)
                .file_size_in_bytes(std::fs::metadata(&path).unwrap().len())
                .record_count(batch.num_rows() as u64)
                .partition_spec_id(0)
                .partition(Struct::empty())
                .equality_ids(equality_ids)
                .build()
                .unwrap();
            delete_manifest.add_file(file, sequence).unwrap();
        }
        let mut manifests = table
            .manifest_list_reader(parent)
            .load()
            .await
            .unwrap()
            .entries()
            .to_vec();
        manifests.push(delete_manifest.write_manifest_file().await.unwrap());
        let manifest_list = format!("{}/delete-manifest-list.avro", table.metadata().location());
        let mut writer = ManifestListWriter::v2(
            table
                .file_io()
                .new_output(&manifest_list)
                .unwrap()
                .writer()
                .await
                .unwrap(),
            snapshot_id,
            Some(parent.snapshot_id()),
            sequence,
        );
        writer.add_manifests(manifests.into_iter()).unwrap();
        writer.close().await.unwrap();
        let snapshot = Snapshot::builder()
            .with_snapshot_id(snapshot_id)
            .with_parent_snapshot_id(Some(parent.snapshot_id()))
            .with_sequence_number(sequence)
            .with_timestamp_ms(parent.timestamp_ms() + 1)
            .with_manifest_list(manifest_list)
            .with_summary(Summary {
                operation: Operation::Delete,
                additional_properties: HashMap::new(),
            })
            .with_schema_id(table.metadata().current_schema_id())
            .build();
        let metadata = table
            .metadata()
            .clone()
            .into_builder(table.metadata_location().map(str::to_owned))
            .set_branch_snapshot(snapshot, "main")
            .unwrap()
            .build()
            .unwrap()
            .metadata;
        self.table = Table::builder()
            .metadata(metadata)
            .identifier(table.identifier().clone())
            .file_io(table.file_io().clone())
            .runtime(Runtime::current())
            .build()
            .unwrap();
    }
}

#[test]
fn parallel_single_file_scans_preserve_projection_and_row_counts() {
    let query_runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let cpu_runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .build()
        .unwrap();
    let runtime = Runtime::new_with_split(&query_runtime, &cpu_runtime);
    let (_fixture, plan, context) = query_runtime.block_on(async {
        let fixture = Fixture::new(true).await;
        let provider =
            IcebergTableProvider::try_new(fixture.table.clone().with_runtime(runtime)).unwrap();
        let context = fixture.context();
        let state = context.state();
        assert_eq!(provider.schema().field(1).data_type(), &DataType::Utf8);
        assert_eq!(
            provider.schema().field(5).data_type(),
            &DataType::LargeBinary
        );
        let DataType::Struct(payload) = provider.schema().field(3).data_type().clone() else {
            unreachable!()
        };
        assert_eq!(payload[1].data_type(), &DataType::Utf8);
        let projection = vec![3, 1, 0, 5];
        let plan = provider
            .scan(&state, Some(&projection), &[], None)
            .await
            .unwrap();
        assert_eq!(plan.output_partitioning().partition_count(), 4);
        let partitions = collect_partitioned(plan, context.task_ctx()).await.unwrap();
        assert!(
            partitions
                .iter()
                .filter(|batches| !batches.is_empty())
                .count()
                > 1
        );
        let mut ids = Vec::new();
        for batch in partitions.into_iter().flatten() {
            assert_eq!(batch.schema().field(0).name(), "payload");
            assert_eq!(batch.schema().field(1).name(), "label");
            assert_eq!(batch.column(1).data_type(), &DataType::Utf8);
            assert_eq!(batch.column(3).data_type(), &DataType::LargeBinary);
            let column = batch
                .column(2)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let blobs = batch
                .column(3)
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .unwrap();
            assert_eq!(
                blobs
                    .iter()
                    .map(|value| value.map(<[u8]>::to_vec))
                    .collect::<Vec<_>>(),
                column
                    .values()
                    .iter()
                    .map(|id| (id % 11 != 0).then(|| id.to_le_bytes().to_vec()))
                    .collect::<Vec<_>>()
            );
            ids.extend(column.values().iter().copied());
        }
        ids.sort_unstable();
        assert_eq!(ids, (0..ROWS).collect::<Vec<_>>());

        let empty = vec![];
        let plan = provider
            .scan(&state, Some(&empty), &[], None)
            .await
            .unwrap();
        let batches = collect(plan, context.task_ctx()).await.unwrap();
        assert_eq!(
            (
                batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
                batches.iter().all(|b| b.num_columns() == 0)
            ),
            (ROWS as usize, true)
        );
        let plan = provider.scan(&state, None, &[], None).await.unwrap();
        (fixture, plan, context.task_ctx())
    });
    cpu_runtime.shutdown_background();
    let error = query_runtime.block_on(collect(plan, context)).unwrap_err();
    let mut cause: &(dyn std::error::Error + 'static) = &error;
    loop {
        if let Some(join) = cause.downcast_ref::<tokio::task::JoinError>() {
            assert!(join.is_cancelled());
            break;
        }
        cause = cause
            .source()
            .expect("CPU executor cancellation must reach the consumer");
    }
}

#[tokio::test]
async fn multiple_files_distribute_rows_across_independent_consumers() {
    const ROWS_PER_FILE: i64 = ROWS * 4;
    const ROW_GROUP_ROWS: usize = 128;
    let fixture = Fixture::new(false).await;
    let mut files = Vec::new();
    let padding = "x".repeat(2048);
    let base = datafusion::arrow::compute::concat_batches(
        &fixture.batch.schema(),
        std::iter::repeat_n(&fixture.batch, 4),
    )
    .unwrap();
    for file in 0..8 {
        let first = file * ROWS_PER_FILE;
        let mut columns = base.columns().to_vec();
        columns[0] = Arc::new(Int64Array::from_iter_values(first..first + ROWS_PER_FILE));
        columns[1] = Arc::new(StringArray::from_iter_values(
            (first..first + ROWS_PER_FILE).map(|id| format!("{id:08}{padding}")),
        ));
        let batch = RecordBatch::try_new(fixture.batch.schema(), columns).unwrap();
        let output = fixture
            .table
            .file_io()
            .new_output(format!(
                "{}/rows-{file}.parquet",
                fixture.table.metadata().location()
            ))
            .unwrap();
        let mut writer = ParquetWriterBuilder::new(
            WriterProperties::builder()
                .set_compression(parquet::basic::Compression::UNCOMPRESSED)
                .set_statistics_enabled(EnabledStatistics::None)
                .set_max_row_group_row_count(Some(ROW_GROUP_ROWS))
                .build(),
            fixture.table.metadata().current_schema().clone(),
        )
        .build(output)
        .await
        .unwrap();
        writer.write(&batch).await.unwrap();
        files.extend(writer.close().await.unwrap().into_iter().map(|mut file| {
            file.content(DataContentType::Data)
                .partition(Struct::empty())
                .partition_spec_id(0)
                .build()
                .unwrap()
        }));
    }
    let tx = Transaction::new(&fixture.table);
    let table = tx
        .fast_append()
        .add_data_files(files)
        .apply(tx)
        .unwrap()
        .commit(&fixture.catalog)
        .await
        .unwrap();
    let provider = IcebergTableProvider::try_new(table).unwrap();
    let context = SessionContext::new_with_config(
        SessionConfig::new()
            .with_target_partitions(14)
            .with_batch_size(19),
    );
    let projection = vec![0];
    let plan = provider
        .scan(&context.state(), Some(&projection), &[], None)
        .await
        .unwrap();
    let partitions = collect_partitioned(plan, context.task_ctx()).await.unwrap();
    assert_eq!(partitions.len(), 14);
    let loads = partitions
        .iter()
        .map(|batches| batches.iter().map(RecordBatch::num_rows).sum::<usize>())
        .collect::<Vec<_>>();
    assert!(loads.iter().all(|rows| *rows > 0));
    assert!(
        *loads.iter().max().unwrap()
            <= (ROWS_PER_FILE as usize * 8).div_ceil(loads.len()) + ROW_GROUP_ROWS,
        "consumer rows: {loads:?}"
    );
    let mut ids = partitions
        .into_iter()
        .flatten()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect::<Vec<_>>();
    ids.sort_unstable();
    assert_eq!(ids, (0..ROWS_PER_FILE * 8).collect::<Vec<_>>());
}

#[tokio::test]
async fn pushed_and_residual_filters_match_datafusion() {
    let fixture = Fixture::new(true).await;
    let context = fixture.context();
    context
        .register_table(
            "ice",
            Arc::new(IcebergTableProvider::try_new(fixture.table.clone()).unwrap()),
        )
        .unwrap();
    context
        .register_table(
            "reference",
            Arc::new(
                MemTable::try_new(fixture.batch.schema(), vec![vec![fixture.batch.clone()]])
                    .unwrap(),
            ),
        )
        .unwrap();
    for filter in [
        "id BETWEEN 17 AND 142",
        "id IN (1, 8, 130, NULL)",
        "flag IS NULL OR label LIKE '%1%'",
        "NOT (id > 12 AND label LIKE '%1%')",
        "flag != false",
        "label LIKE 'row-1%' AND id > 115",
        "CAST(id AS VARCHAR) = '128'",
        "payload.code BETWEEN 12 AND 30",
        "payload IS NOT NULL",
        "payload IS NULL OR id > 200",
        "special_float > 1.0",
        "special_float < 1.0",
        "NOT (special_float < 1.0)",
    ] {
        let mut results = Vec::new();
        for name in ["ice", "reference"] {
            let batches = context
                .sql(&format!(
                    "SELECT label, id FROM {name} WHERE {filter} ORDER BY id"
                ))
                .await
                .unwrap()
                .collect()
                .await
                .unwrap();
            results.push(pretty_format_batches(&batches).unwrap().to_string());
        }
        assert_eq!(results[0], results[1], "filter: {filter}");
    }
    let keys = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "join_key",
            DataType::Int64,
            false,
        )])),
        vec![Arc::new(Int64Array::from(vec![5, 19, 205]))],
    )
    .unwrap();
    context
        .register_table(
            "keys",
            Arc::new(MemTable::try_new(keys.schema(), vec![vec![keys]]).unwrap()),
        )
        .unwrap();
    let plan = context
        .sql("SELECT data.id FROM keys JOIN ice data ON keys.join_key = data.id ORDER BY data.id")
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let mut nodes = vec![plan.clone()];
    let mut scan = None;
    while let Some(node) = nodes.pop() {
        if node.name() == "IcebergScan" {
            scan = Some(node.clone());
        }
        nodes.extend(node.children().into_iter().cloned());
    }
    let batches = collect(plan, context.task_ctx()).await.unwrap();
    let ids = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect::<Vec<_>>();
    assert_eq!(ids, vec![5, 19, 205]);
    assert!(
        scan.unwrap()
            .metrics()
            .unwrap()
            .iter()
            .filter(|metric| metric.value().name() == "dynamic_filter_rows_pruned")
            .map(|metric| metric.value().as_usize())
            .sum::<usize>()
            > 0
    );
}

#[tokio::test]
async fn deletes_apply_across_split_ranges_and_unprojected_keys() {
    let mut fixture = Fixture::new(true).await;
    fixture.add_deletes().await;
    let provider = Arc::new(IcebergTableProvider::try_new(fixture.table.clone()).unwrap());
    let context = fixture.context();
    context.register_table("ice", provider.clone()).unwrap();
    let removed = [0, 31, 32, 97, 128, 255, 256];
    for (filter, expected) in [
        (
            "true",
            (0..ROWS)
                .filter(|id| !removed.contains(id))
                .collect::<Vec<_>>(),
        ),
        (
            "id >= 64 AND id < 200",
            (64..200).filter(|id| !removed.contains(id)).collect(),
        ),
    ] {
        let batches = context
            .sql(&format!("SELECT id FROM ice WHERE {filter} ORDER BY id"))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let actual = batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
    let projection = vec![1];
    let plan = provider
        .scan(&context.state(), Some(&projection), &[], None)
        .await
        .unwrap();
    assert_eq!(plan.output_partitioning().partition_count(), 4);
    let batches = collect(plan, context.task_ctx()).await.unwrap();
    let mut labels = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .iter()
                .map(|value| value.map(str::to_owned))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    labels.sort_unstable();
    let mut expected = (0..ROWS)
        .filter(|id| !removed.contains(id))
        .map(|id| (id % 11 != 0).then(|| format!("row-{id}")))
        .collect::<Vec<_>>();
    expected.sort_unstable();
    assert_eq!(labels, expected);
}

#[tokio::test]
async fn added_columns_and_empty_snapshots_keep_their_schema() {
    let mut fixture = Fixture::new(true).await;
    let tx = Transaction::new(&fixture.table);
    let table = tx
        .upgrade_table_version()
        .set_format_version(FormatVersion::V3)
        .apply(tx)
        .unwrap()
        .commit(&fixture.catalog)
        .await
        .unwrap();
    let tx = Transaction::new(&table);
    let table = tx
        .update_schema()
        .add_column(AddColumn::optional(
            "new_column",
            Type::Primitive(PrimitiveType::String),
        ))
        .add_column(AddColumn::required(
            "defaulted",
            Type::Primitive(PrimitiveType::Int),
            Literal::int(7),
        ))
        .add_column(AddColumn::required(
            "default_string",
            PrimitiveType::String.into(),
            Literal::string("fallback"),
        ))
        .apply(tx)
        .unwrap()
        .commit(&fixture.catalog)
        .await
        .unwrap();
    fixture.table = table;
    fixture.evolve_field(1, |field| field.name = "renamed_label".to_owned());
    let provider = Arc::new(IcebergTableProvider::try_new(fixture.table.clone()).unwrap());
    let context = fixture.context();
    let projection = vec![6];
    let plan = provider
        .scan(&context.state(), Some(&projection), &[], None)
        .await
        .unwrap();
    let batches = collect(plan, context.task_ctx()).await.unwrap();
    assert_eq!(
        (
            batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
            batches
                .iter()
                .map(|b| b.column(0).null_count())
                .sum::<usize>()
        ),
        (ROWS as usize, ROWS as usize)
    );
    context.register_table("evolved", provider.clone()).unwrap();
    for filter in [
        "defaulted = 7",
        "defaulted > 6",
        "default_string = 'fallback'",
    ] {
        let batches = context
            .sql(&format!(
                "SELECT id FROM evolved WHERE {filter} ORDER BY id"
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let ids = batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect::<Vec<_>>();
        assert_eq!(ids, (0..ROWS).collect::<Vec<_>>(), "filter: {filter}");
    }
    let batches = context
        .sql("SELECT renamed_label, default_string FROM evolved ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let labels = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .iter()
                .map(|value| value.map(str::to_owned))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let expected = fixture
        .batch
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .map(|value| value.map(str::to_owned))
        .collect::<Vec<_>>();
    assert_eq!(labels, expected);
    for batch in batches {
        assert!(
            batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .iter()
                .all(|value| value == Some("fallback"))
        );
    }
    let projection = vec![8, 0];
    let plan = provider
        .scan(&context.state(), Some(&projection), &[], None)
        .await
        .unwrap();
    let key: Arc<dyn PhysicalExpr> = Arc::new(Column::new("default_string", 0));
    let dynamic = Arc::new(DynamicFilterPhysicalExpr::new(
        vec![key.clone()],
        Arc::new(PhysicalLiteral::new(ScalarValue::Boolean(Some(true)))),
    ));
    dynamic
        .update(Arc::new(BinaryExpr::new(
            key,
            Operator::Eq,
            Arc::new(PhysicalLiteral::new(ScalarValue::Utf8(Some(
                "fallback".to_owned(),
            )))),
        )))
        .unwrap();
    let plan = with_dynamic_filter(plan, dynamic, &context);
    let batches = collect(plan, context.task_ctx()).await.unwrap();
    assert_eq!(
        batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
        ROWS as usize
    );

    let empty = Fixture::new(false).await;
    let provider = IcebergTableProvider::try_new(empty.table).unwrap();
    let plan = provider
        .scan(&context.state(), None, &[], None)
        .await
        .unwrap();
    assert_eq!(plan.schema().as_ref(), empty.batch.schema().as_ref());
    assert_eq!(plan.output_partitioning().partition_count(), 4);
    assert!(collect(plan, context.task_ctx()).await.unwrap().is_empty());
}

#[tokio::test]
async fn promoted_integer_predicates_preserve_out_of_range_literals() {
    let mut fixture = Fixture::new_with_id_type(true, DataType::Int32).await;
    fixture.evolve_field(0, |field| *field.field_type = PrimitiveType::Long.into());
    let provider = Arc::new(IcebergTableProvider::try_new(fixture.table.clone()).unwrap());
    let context = fixture.context();
    context
        .register_table("promoted", provider.clone())
        .unwrap();
    let unfiltered = context
        .sql("SELECT * FROM promoted")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(
        unfiltered.iter().map(RecordBatch::num_rows).sum::<usize>(),
        ROWS as usize
    );
    assert_eq!(
        unfiltered
            .iter()
            .map(|batch| batch.column(0).null_count())
            .sum::<usize>(),
        20
    );
    assert_eq!(provider.schema().field(0).data_type(), &DataType::Int64);
    context
        .register_table(
            "reference",
            Arc::new(MemTable::try_new(provider.schema(), vec![unfiltered]).unwrap()),
        )
        .unwrap();
    for filter in [
        "id < 2147483648",
        "id >= -2147483649",
        "id != 2147483648",
        "id IN (4, 97, 2147483648)",
        "id NOT IN (4, 97, 2147483648)",
        "id IN (-2147483649, 4, 97)",
        "id NOT IN (-2147483649, 4, 97)",
        "id IS NULL OR id < 2147483648",
    ] {
        let mut results = Vec::new();
        for name in ["promoted", "reference"] {
            let batches = context
                .sql(&format!(
                    "SELECT id, label FROM {name} WHERE {filter} ORDER BY id NULLS FIRST, label"
                ))
                .await
                .unwrap()
                .collect()
                .await
                .unwrap();
            results.push(pretty_format_batches(&batches).unwrap().to_string());
        }
        assert_eq!(results[0], results[1], "filter: {filter}");
    }
}

#[tokio::test]
async fn dynamic_filters_remain_live_after_projection_and_field_rename() {
    let mut fixture = Fixture::new(true).await;
    fixture.add_deletes().await;
    fixture.evolve_field(0, |field| field.name = "key".to_owned());
    let provider = IcebergTableProvider::try_new(fixture.table.clone()).unwrap();
    let context = SessionContext::new_with_config(
        SessionConfig::new()
            .with_target_partitions(2)
            .with_batch_size(8),
    );
    let projection = vec![1, 0];
    let plan = provider
        .scan(
            &context.state(),
            Some(&projection),
            &[col("key").gt(lit(50_i64))],
            None,
        )
        .await
        .unwrap();
    let baseline = plan
        .execute(0, context.task_ctx())
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let baseline_rows = baseline.iter().map(RecordBatch::num_rows).sum::<usize>();
    assert!(baseline_rows > 32);
    let key: Arc<dyn PhysicalExpr> = Arc::new(Column::new("key", 1));
    let dynamic = Arc::new(DynamicFilterPhysicalExpr::new(
        vec![key.clone()],
        Arc::new(PhysicalLiteral::new(ScalarValue::Boolean(Some(true)))),
    ));
    let plan = with_dynamic_filter(plan, dynamic.clone(), &context);
    let mut visited = false;
    plan.apply_expressions(&mut |expression| {
        visited |= expression.expression_id() == dynamic.expression_id();
        Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
    })
    .unwrap();
    assert!(visited);
    let mut input = plan.execute(0, context.task_ctx()).unwrap();
    let first = input.next().await.unwrap().unwrap();
    dynamic
        .update(Arc::new(BinaryExpr::new(
            key,
            Operator::GtEq,
            Arc::new(PhysicalLiteral::new(ScalarValue::Int64(Some(200)))),
        )))
        .unwrap();
    let later = input.try_collect::<Vec<_>>().await.unwrap();
    assert!(
        first.num_rows() + later.iter().map(RecordBatch::num_rows).sum::<usize>() < baseline_rows
    );
    dynamic.mark_complete();
    let batches = collect(plan.clone(), context.task_ctx()).await.unwrap();
    let mut keys = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect::<Vec<_>>();
    keys.sort_unstable();
    assert_eq!(keys, (200..255).collect::<Vec<_>>());
    assert!(
        plan.metrics()
            .unwrap()
            .iter()
            .filter(|metric| metric.value().name() == "dynamic_filter_rows_pruned")
            .map(|metric| metric.value().as_usize())
            .sum::<usize>()
            > 0
    );
}
