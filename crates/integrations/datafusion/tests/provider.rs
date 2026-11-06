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
    ArrayRef, BooleanArray, Float64Array, Int32Array, Int64Array, StringArray, StructArray,
};
use datafusion::arrow::buffer::NullBuffer;
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::catalog::TableProvider;
use datafusion::datasource::MemTable;
use datafusion::physical_plan::{ExecutionPlanProperties, collect, collect_partitioned};
use datafusion::prelude::{SessionConfig, SessionContext};
use iceberg::arrow::{arrow_schema_to_schema_auto_assign_ids, schema_to_arrow_schema};
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::metadata_columns::{
    RESERVED_COL_NAME_DELETE_FILE_PATH, RESERVED_COL_NAME_DELETE_FILE_POS,
    RESERVED_FIELD_ID_DELETE_FILE_PATH, RESERVED_FIELD_ID_DELETE_FILE_POS,
};
use iceberg::spec::{
    DataContentType, DataFileBuilder, DataFileFormat, FormatVersion, Literal, ManifestListWriter,
    ManifestWriterBuilder, Operation, PrimitiveType, Snapshot, Struct, Summary, Type,
};
use iceberg::table::Table;
use iceberg::transaction::{AddColumn, ApplyTransactionAction, Transaction};
use iceberg::writer::file_writer::{FileWriter, FileWriterBuilder, ParquetWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, MemoryCatalog, NamespaceIdent, Runtime, TableCreation};
use iceberg_datafusion::IcebergTableProvider;
use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

const ROWS: i64 = 257;

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
                DataType::Struct(vec![Field::new("code", DataType::Int64, true)].into()),
                true,
            ),
            Field::new("special_float", DataType::Float64, true),
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
                vec![Arc::new(Int64Array::from_iter(
                    (0..ROWS).map(|id| (id % 5 != 0).then_some(id * 3)),
                ))],
                Some(NullBuffer::from_iter((0..ROWS).map(|id| id % 5 != 0))),
            )),
            Arc::new(Float64Array::from(vec![f64::NAN; ROWS as usize])),
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

#[tokio::test]
async fn parallel_single_file_scans_preserve_projection_and_row_counts() {
    let fixture = Fixture::new(true).await;
    let provider = IcebergTableProvider::try_new(fixture.table.clone()).unwrap();
    let context = fixture.context();
    let state = context.state();
    let projection = vec![3, 1, 0];
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
        let column = batch
            .column(2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
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
    let fixture = Fixture::new(true).await;
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
        .apply(tx)
        .unwrap()
        .commit(&fixture.catalog)
        .await
        .unwrap();
    let provider = Arc::new(IcebergTableProvider::try_new(table).unwrap());
    let context = fixture.context();
    let projection = vec![5];
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
    context.register_table("evolved", provider).unwrap();
    for filter in ["defaulted = 7", "defaulted > 6"] {
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

    let empty = Fixture::new(false).await;
    let provider = IcebergTableProvider::try_new(empty.table).unwrap();
    let plan = provider
        .scan(&context.state(), None, &[], None)
        .await
        .unwrap();
    assert_eq!(plan.schema(), empty.batch.schema());
    assert_eq!(plan.output_partitioning().partition_count(), 4);
    assert!(collect(plan, context.task_ctx()).await.unwrap().is_empty());
}

#[tokio::test]
async fn promoted_integer_predicates_preserve_out_of_range_literals() {
    let fixture = Fixture::new_with_id_type(true, DataType::Int32).await;
    let current = fixture.table.metadata().current_schema();
    let mut fields = current.as_struct().fields().to_vec();
    let mut id = fields[0].as_ref().clone();
    id.field_type = Box::new(Type::Primitive(PrimitiveType::Long));
    fields[0] = Arc::new(id);
    let schema_id = current.schema_id() + 1;
    let schema = iceberg::spec::Schema::builder()
        .with_schema_id(schema_id)
        .with_fields(fields)
        .build()
        .unwrap();
    let metadata = fixture
        .table
        .metadata()
        .clone()
        .into_builder(fixture.table.metadata_location().map(str::to_owned))
        .add_schema(schema)
        .unwrap()
        .set_current_schema(schema_id)
        .unwrap()
        .build()
        .unwrap()
        .metadata;
    let table = Table::builder()
        .metadata(metadata)
        .identifier(fixture.table.identifier().clone())
        .file_io(fixture.table.file_io().clone())
        .runtime(Runtime::current())
        .build()
        .unwrap();
    let provider = Arc::new(IcebergTableProvider::try_new(table).unwrap());
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
