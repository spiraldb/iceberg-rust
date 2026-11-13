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
use std::num::NonZeroUsize;
use std::sync::Arc;

use arrow_array::{ArrayRef, Float32Array, Int16Array, Int32Array, RecordBatch, UInt16Array};
use arrow_schema::{Field, Schema as ArrowSchema};
use iceberg::arrow::VortexScanSource;
use iceberg::expr::{Bind, Reference};
use iceberg::io::FileIO;
use iceberg::scan::FileScanTask;
use iceberg::spec::{DataFile, Datum, NestedField, PrimitiveType, Schema};
use iceberg::writer::file_writer::vortex_data_file_builder;
use tempfile::TempDir;
use vortex::VortexSessionDefault;
use vortex::array::stats::StatsSet;
use vortex::arrow::ArrowSessionExt;
use vortex::dtype::FieldPath;
use vortex::expr::stats::{Stat, StatsProvider};
use vortex::file::{OpenOptionsSessionExt, WriteOptionsSessionExt};
use vortex::scan::DataSource;
use vortex::session::VortexSession;

struct File {
    _directory: TempDir,
    data: DataFile,
}

fn schema(name: &str, primitive: PrimitiveType) -> Arc<Schema> {
    Arc::new(
        Schema::builder()
            .with_fields(vec![Arc::new(NestedField::optional(
                17,
                name,
                primitive.into(),
            ))])
            .build()
            .unwrap(),
    )
}

async fn file(values: Vec<Option<i32>>) -> File {
    file_array(Arc::new(Int32Array::from(values))).await
}

async fn file_array(array: ArrayRef) -> File {
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("part.vortex");
    let file_schema = schema("old_key", PrimitiveType::Int);
    let arrow = Arc::new(ArrowSchema::new(vec![Field::new(
        "old_key",
        array.data_type().clone(),
        true,
    )]));
    let batch = RecordBatch::try_new(arrow.clone(), vec![array]).unwrap();
    let session = VortexSession::default();
    let array = session
        .arrow()
        .from_arrow_record_batch(batch, &arrow)
        .unwrap();
    session
        .write_options()
        .with_metadata_segment(
            "iceberg.schema",
            serde_json::to_vec(file_schema.as_ref()).unwrap(),
        )
        .write(
            tokio::fs::File::create(&path).await.unwrap(),
            array.to_array_stream(),
        )
        .await
        .unwrap();
    let size = path.metadata().unwrap().len();
    let opened = session
        .open_options()
        .with_file_size(size)
        .open_path(&path)
        .await
        .unwrap();
    let original_bytes = std::fs::read(&path).unwrap();
    let data = vortex_data_file_builder(
        path.to_str().unwrap().to_owned(),
        size,
        &file_schema,
        opened.footer(),
    )
    .unwrap()
    .partition_spec_id(0)
    .build()
    .unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), original_bytes);
    File {
        _directory: directory,
        data,
    }
}

fn task(file: &File, schema: Arc<Schema>) -> FileScanTask {
    FileScanTask::builder()
        .with_file_size_in_bytes(file.data.file_size_in_bytes())
        .with_start(0)
        .with_length(0)
        .with_record_count(Some(file.data.record_count()))
        .with_data_file_path(file.data.file_path().to_owned())
        .with_data_file_format(file.data.file_format())
        .with_schema(schema)
        .with_project_field_ids(vec![17])
        .with_case_sensitive(true)
        .with_lower_bounds(Arc::new(file.data.lower_bounds().clone()))
        .with_upper_bounds(Arc::new(file.data.upper_bounds().clone()))
        .build()
        .unwrap()
}

async fn statistics(tasks: Vec<FileScanTask>, schema: Arc<Schema>) -> StatsSet {
    let source = VortexScanSource::try_new(
        FileIO::new_with_fs(),
        schema,
        tasks,
        VortexSession::default(),
        NonZeroUsize::new(2).unwrap(),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(
        source
            .field_statistics(&FieldPath::from_name("old_key"))
            .await
            .unwrap()
            .is_empty()
    );
    source
        .field_statistics(&FieldPath::from_name("renamed"))
        .await
        .unwrap()
}

#[tokio::test]
async fn imported_footer_populates_manifest_metrics_without_rewriting_data() {
    let file = file(vec![Some(-9), None, Some(3)]).await;
    assert_eq!(file.data.record_count(), 3);
    assert_eq!(file.data.value_counts(), &HashMap::from([(17, 3)]));
    assert_eq!(file.data.null_value_counts(), &HashMap::from([(17, 1)]));
    assert_eq!(
        file.data.lower_bounds(),
        &HashMap::from([(17, Datum::int(-9))])
    );
    assert_eq!(
        file.data.upper_bounds(),
        &HashMap::from([(17, Datum::int(3))])
    );
    assert!(!file.data.column_sizes().is_empty());
}

#[tokio::test]
async fn source_merges_typed_bounds_through_rename_promotion_filter_and_ranges() {
    let first = file(vec![Some(-9), None, Some(3)]).await;
    let second = file(vec![Some(7), Some(14), Some(27)]).await;
    let current = schema("renamed", PrimitiveType::Long);
    let mut one = task(&first, current.clone());
    let filter = Reference::new("renamed")
        .greater_than(Datum::long(0))
        .bind(current.clone(), true)
        .unwrap();
    let mut encoded = serde_json::to_value(&one).unwrap();
    encoded["predicate"] = serde_json::to_value(filter).unwrap();
    one = serde_json::from_value(encoded).unwrap();
    let two = task(&second, current.clone());
    let two = two
        .clone()
        .with_byte_range(0, two.file_size_in_bytes() / 2)
        .unwrap();
    let stats = statistics(vec![one, two], current).await;
    let dtype = vortex::dtype::DType::Primitive(
        vortex::dtype::PType::I64,
        vortex::dtype::Nullability::Nullable,
    );
    let typed = stats.as_typed_ref(&dtype);
    assert_eq!(
        typed
            .get(Stat::Min)
            .as_inexact()
            .unwrap()
            .as_primitive()
            .typed_value::<i64>(),
        Some(-9)
    );
    assert_eq!(
        typed
            .get(Stat::Max)
            .as_inexact()
            .unwrap()
            .as_primitive()
            .typed_value::<i64>(),
        Some(27)
    );
    assert_eq!(stats.len(), 2); // No fabricated distinct or null counts.
}

#[tokio::test]
async fn missing_reversed_or_incompatible_bounds_stay_unknown() {
    let zero_file = file(vec![]).await;
    let file = file(vec![Some(1), Some(2)]).await;
    let current = schema("renamed", PrimitiveType::Long);
    let good = task(&file, current.clone());
    let empty = task(&zero_file, current.clone());
    assert!(empty.lower_bounds().is_empty());
    assert!(empty.upper_bounds().is_empty());
    for (lower, upper) in [
        (None, Some(Datum::int(2))),
        (Some(Datum::int(3)), Some(Datum::int(2))),
        (Some(Datum::string("1")), Some(Datum::string("2"))),
    ] {
        // Use the task's public builder for arbitrary persisted bounds.
        let bad = FileScanTask::builder()
            .with_file_size_in_bytes(file.data.file_size_in_bytes())
            .with_start(0)
            .with_length(0)
            .with_record_count(Some(2))
            .with_data_file_path(file.data.file_path().to_owned())
            .with_data_file_format(file.data.file_format())
            .with_schema(current.clone())
            .with_project_field_ids(vec![17])
            .with_case_sensitive(true)
            .with_lower_bounds(Arc::new(
                lower.map(|v| HashMap::from([(17, v)])).unwrap_or_default(),
            ))
            .with_upper_bounds(Arc::new(
                upper.map(|v| HashMap::from([(17, v)])).unwrap_or_default(),
            ))
            .build()
            .unwrap();
        assert!(
            statistics(vec![good.clone(), bad], current.clone())
                .await
                .is_empty()
        );
        assert!(
            !statistics(vec![good.clone(), empty.clone()], current.clone())
                .await
                .is_empty()
        );
    }
}

#[tokio::test]
async fn imported_narrow_integer_bounds_widen_without_panics_or_wrapping() {
    for (array, minimum, maximum) in [
        (
            Arc::new(Int16Array::from(vec![Some(i16::MIN), None, Some(i16::MAX)])) as ArrayRef,
            i32::from(i16::MIN),
            i32::from(i16::MAX),
        ),
        (
            Arc::new(UInt16Array::from(vec![Some(0), None, Some(u16::MAX)])) as ArrayRef,
            0,
            i32::from(u16::MAX),
        ),
    ] {
        let file = file_array(array).await;
        assert_eq!(
            file.data.lower_bounds(),
            &HashMap::from([(17, Datum::int(minimum))])
        );
        assert_eq!(
            file.data.upper_bounds(),
            &HashMap::from([(17, Datum::int(maximum))])
        );
    }
    let incompatible = file_array(Arc::new(Float32Array::from(vec![1.5, 2.5]))).await;
    assert!(incompatible.data.lower_bounds().is_empty());
    assert!(incompatible.data.upper_bounds().is_empty());
}
