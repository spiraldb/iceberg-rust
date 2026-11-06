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
    ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray, StructArray,
};
use datafusion::arrow::buffer::NullBuffer;
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::catalog::TableProvider;
use datafusion::physical_plan::{ExecutionPlanProperties, collect, collect_partitioned};
use datafusion::prelude::{SessionConfig, SessionContext};
use iceberg::arrow::{arrow_schema_to_schema_auto_assign_ids, schema_to_arrow_schema};
use iceberg::io::LocalFsStorageFactory;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{DataContentType, Struct};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::file_writer::{FileWriter, FileWriterBuilder, ParquetWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, Runtime, TableCreation};
use iceberg_datafusion::IcebergTableProvider;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

const ROWS: i64 = 257;

struct Fixture {
    _root: TempDir,
    table: Table,
}

impl Fixture {
    async fn new(write: bool) -> Self {
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
            Field::new("id", DataType::Int64, false),
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
        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
            Arc::new(Int64Array::from_iter_values(0..ROWS)) as ArrayRef,
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
        Self { _root: root, table }
    }

    fn context(&self) -> SessionContext {
        SessionContext::new_with_config(
            SessionConfig::new()
                .with_target_partitions(4)
                .with_batch_size(19),
        )
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
