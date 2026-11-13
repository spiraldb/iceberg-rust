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

use std::fs::metadata;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int16Array, Int32Array, RecordBatch, UInt16Array};
use arrow_schema::{Field, Schema as ArrowSchema};
use futures::TryStreamExt;
use futures::stream::iter;
use iceberg::Runtime;
use iceberg::arrow::ArrowReaderBuilder;
use iceberg::expr::{Bind, Predicate, Reference};
use iceberg::io::FileIO;
use iceberg::scan::FileScanTask;
use iceberg::spec::{DataFileFormat, Datum, NestedField, PrimitiveType, Schema};
use tempfile::TempDir;
use tokio::fs::File;
use vortex::VortexSessionDefault;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::arrow::ArrowSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::session::VortexSession;

#[tokio::test]
async fn signed_equality_above_max_matches_nothing() {
    let (_directory, path) = signed_file().await;
    let actual = read_values(
        &path,
        Reference::new("value").equal_to(Datum::int(i32::from(i16::MAX) + 1)),
    )
    .await;
    assert_eq!(actual, Vec::<i32>::new());
}

#[tokio::test]
async fn signed_greater_than_below_min_excludes_null() {
    let (_directory, path) = signed_file().await;
    let actual = read_values(
        &path,
        Reference::new("value").greater_than(Datum::int(i32::from(i16::MIN) - 1)),
    )
    .await;
    assert_eq!(actual, vec![i32::from(i16::MIN), i32::from(i16::MAX)]);
}

#[tokio::test]
async fn signed_less_than_above_max_excludes_null() {
    let (_directory, path) = signed_file().await;
    let actual = read_values(
        &path,
        Reference::new("value").less_than(Datum::int(i32::from(i16::MAX) + 1)),
    )
    .await;
    assert_eq!(actual, vec![i32::from(i16::MIN), i32::from(i16::MAX)]);
}

#[tokio::test]
async fn signed_in_discards_out_of_range_literal() {
    let (_directory, path) = signed_file().await;
    let actual = read_values(
        &path,
        Reference::new("value").is_in([
            Datum::int(i32::from(i16::MIN)),
            Datum::int(i32::from(i16::MAX) + 1),
        ]),
    )
    .await;
    assert_eq!(actual, vec![i32::from(i16::MIN)]);
}

#[tokio::test]
async fn signed_in_out_of_range_literals_matches_nothing() {
    let (_directory, path) = signed_file().await;
    let actual = read_values(
        &path,
        Reference::new("value").is_in([
            Datum::int(i32::from(i16::MAX) + 1),
            Datum::int(i32::from(i16::MAX) + 2),
        ]),
    )
    .await;
    assert_eq!(actual, Vec::<i32>::new());
}

#[tokio::test]
async fn signed_not_in_out_of_range_literals_excludes_null() {
    let (_directory, path) = signed_file().await;
    let actual = read_values(
        &path,
        Reference::new("value").is_not_in([
            Datum::int(i32::from(i16::MAX) + 1),
            Datum::int(i32::from(i16::MAX) + 2),
        ]),
    )
    .await;
    assert_eq!(actual, vec![i32::from(i16::MIN), i32::from(i16::MAX)]);
}

#[tokio::test]
async fn signed_not_in_discards_out_of_range_literal() {
    let (_directory, path) = signed_file().await;
    let actual = read_values(
        &path,
        Reference::new("value").is_not_in([
            Datum::int(i32::from(i16::MIN)),
            Datum::int(i32::from(i16::MAX) + 1),
        ]),
    )
    .await;
    assert_eq!(actual, vec![i32::from(i16::MAX)]);
}

#[tokio::test]
async fn unsigned_equality_above_max_matches_nothing() {
    let (_directory, path) = unsigned_file().await;
    let actual = read_values(
        &path,
        Reference::new("value").equal_to(Datum::int(i32::from(u16::MAX) + 1)),
    )
    .await;
    assert_eq!(actual, Vec::<i32>::new());
}

#[tokio::test]
async fn unsigned_greater_than_below_min_excludes_null() {
    let (_directory, path) = unsigned_file().await;
    let actual = read_values(
        &path,
        Reference::new("value").greater_than(Datum::int(i32::from(u16::MIN) - 1)),
    )
    .await;
    assert_eq!(actual, vec![i32::from(u16::MIN), i32::from(u16::MAX)]);
}

#[tokio::test]
async fn unsigned_less_than_above_max_excludes_null() {
    let (_directory, path) = unsigned_file().await;
    let actual = read_values(
        &path,
        Reference::new("value").less_than(Datum::int(i32::from(u16::MAX) + 1)),
    )
    .await;
    assert_eq!(actual, vec![i32::from(u16::MIN), i32::from(u16::MAX)]);
}

#[tokio::test]
async fn unsigned_in_discards_out_of_range_literal() {
    let (_directory, path) = unsigned_file().await;
    let actual = read_values(
        &path,
        Reference::new("value").is_in([
            Datum::int(i32::from(u16::MIN)),
            Datum::int(i32::from(u16::MAX) + 1),
        ]),
    )
    .await;
    assert_eq!(actual, vec![i32::from(u16::MIN)]);
}

#[tokio::test]
async fn unsigned_in_out_of_range_literals_matches_nothing() {
    let (_directory, path) = unsigned_file().await;
    let actual = read_values(
        &path,
        Reference::new("value").is_in([
            Datum::int(i32::from(u16::MAX) + 1),
            Datum::int(i32::from(u16::MAX) + 2),
        ]),
    )
    .await;
    assert_eq!(actual, Vec::<i32>::new());
}

#[tokio::test]
async fn unsigned_not_in_out_of_range_literals_excludes_null() {
    let (_directory, path) = unsigned_file().await;
    let actual = read_values(
        &path,
        Reference::new("value").is_not_in([
            Datum::int(i32::from(u16::MAX) + 1),
            Datum::int(i32::from(u16::MAX) + 2),
        ]),
    )
    .await;
    assert_eq!(actual, vec![i32::from(u16::MIN), i32::from(u16::MAX)]);
}

#[tokio::test]
async fn unsigned_not_in_discards_out_of_range_literal() {
    let (_directory, path) = unsigned_file().await;
    let actual = read_values(
        &path,
        Reference::new("value").is_not_in([
            Datum::int(i32::from(u16::MIN)),
            Datum::int(i32::from(u16::MAX) + 1),
        ]),
    )
    .await;
    assert_eq!(actual, vec![i32::from(u16::MAX)]);
}

async fn signed_file() -> (TempDir, String) {
    write_vortex_file(Arc::new(Int16Array::from(vec![
        Some(i16::MIN),
        None,
        Some(i16::MAX),
    ])))
    .await
}

async fn unsigned_file() -> (TempDir, String) {
    write_vortex_file(Arc::new(UInt16Array::from(vec![
        Some(u16::MIN),
        None,
        Some(u16::MAX),
    ])))
    .await
}

async fn write_vortex_file(values: ArrayRef) -> (TempDir, String) {
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("data.vortex");
    let batch = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![Field::new(
            "value",
            values.data_type().clone(),
            true,
        )])),
        vec![values],
    )
    .unwrap();
    let session = VortexSession::default();
    let array = session
        .arrow()
        .from_arrow_record_batch(batch.clone(), batch.schema_ref())
        .unwrap();
    let stream = ArrayStreamAdapter::new(array.dtype().clone(), iter([Ok(array)]));
    session
        .write_options()
        .write(File::create(&path).await.unwrap(), stream)
        .await
        .unwrap();
    (directory, path.to_str().unwrap().to_owned())
}

async fn read_values(path: &str, predicate: Predicate) -> Vec<i32> {
    let schema = Arc::new(
        Schema::builder()
            .with_fields(vec![Arc::new(NestedField::optional(
                1,
                "value",
                PrimitiveType::Int.into(),
            ))])
            .build()
            .unwrap(),
    );
    let task = FileScanTask::builder()
        .with_file_size_in_bytes(metadata(path).unwrap().len())
        .with_start(0)
        .with_length(0)
        .with_record_count(Some(3))
        .with_data_file_path(path.to_owned())
        .with_data_file_format(DataFileFormat::Vortex)
        .with_schema(schema.clone())
        .with_project_field_ids(vec![1])
        .with_case_sensitive(true)
        .with_predicate(Some(predicate.bind(schema, true).unwrap()))
        .build()
        .unwrap();
    let batches: Vec<RecordBatch> =
        ArrowReaderBuilder::new(FileIO::new_with_fs(), Runtime::current())
            .build()
            .read(Box::pin(iter([Ok(task)])))
            .unwrap()
            .stream()
            .try_collect()
            .await
            .unwrap();
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values()
                .iter()
                .copied()
        })
        .collect()
}
