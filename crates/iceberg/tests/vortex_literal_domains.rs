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

use std::sync::Arc;

use arrow_array::{ArrayRef, Int16Array, Int32Array, Int64Array, RecordBatch, UInt16Array};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::TryStreamExt;
use iceberg::Runtime;
use iceberg::arrow::ArrowReaderBuilder;
use iceberg::expr::{Bind, BoundPredicate, Predicate, PredicateOperator, Reference, SetExpression};
use iceberg::io::FileIO;
use iceberg::scan::FileScanTask;
use iceberg::spec::{DataFileFormat, Datum, NestedField, PrimitiveType, Schema};
use tempfile::TempDir;
use vortex::VortexSessionDefault;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::arrow::ArrowSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::session::VortexSession;

struct Fixture {
    _directory: TempDir,
    task: FileScanTask,
    values: Vec<Option<i64>>,
    logical_type: PrimitiveType,
}

impl Fixture {
    async fn new(array: ArrayRef, values: Vec<Option<i64>>, logical_type: PrimitiveType) -> Self {
        let directory = TempDir::new().unwrap();
        let path = directory.path().join("narrow.vortex");
        let batch = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![Field::new(
                "value",
                array.data_type().clone(),
                true,
            )])),
            vec![array],
        )
        .unwrap();
        let session = VortexSession::default();
        let array = session
            .arrow()
            .from_arrow_record_batch(batch.clone(), batch.schema_ref())
            .unwrap();
        let stream =
            ArrayStreamAdapter::new(array.dtype().clone(), futures::stream::iter([Ok(array)]));
        session
            .write_options()
            .write(tokio::fs::File::create(&path).await.unwrap(), stream)
            .await
            .unwrap();
        let schema = Arc::new(
            Schema::builder()
                .with_fields(vec![Arc::new(NestedField::optional(
                    1,
                    "value",
                    logical_type.clone().into(),
                ))])
                .build()
                .unwrap(),
        );
        let task = FileScanTask::builder()
            .with_file_size_in_bytes(std::fs::metadata(&path).unwrap().len())
            .with_start(0)
            .with_length(0)
            .with_record_count(Some(values.len() as u64))
            .with_data_file_path(path.to_str().unwrap().to_owned())
            .with_data_file_format(DataFileFormat::Vortex)
            .with_schema(schema)
            .with_project_field_ids(vec![1])
            .with_case_sensitive(true)
            .build()
            .unwrap();
        Self {
            _directory: directory,
            task,
            values,
            logical_type,
        }
    }

    fn datum(&self, value: i64) -> Datum {
        match self.logical_type {
            PrimitiveType::Int => Datum::int(i32::try_from(value).unwrap()),
            PrimitiveType::Long => Datum::long(value),
            _ => unreachable!(),
        }
    }

    async fn scan(&self, predicate: Predicate) -> Vec<Option<i64>> {
        self.scan_bound(
            predicate
                .bind(Arc::new(self.task.schema().clone()), true)
                .unwrap(),
        )
        .await
    }

    async fn scan_bound(&self, predicate: BoundPredicate) -> Vec<Option<i64>> {
        let mut task = serde_json::to_value(&self.task).unwrap();
        task["predicate"] = serde_json::to_value(predicate).unwrap();
        let task: FileScanTask = serde_json::from_value(task).unwrap();
        let reader = ArrowReaderBuilder::new(FileIO::new_with_fs(), Runtime::current()).build();
        let batches = reader
            .read(Box::pin(futures::stream::iter([Ok(task)])))
            .unwrap()
            .stream()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let mut values = Vec::new();
        for batch in batches {
            let array = batch.column(0);
            match array.data_type() {
                DataType::Int32 => values.extend(
                    array
                        .as_any()
                        .downcast_ref::<Int32Array>()
                        .unwrap()
                        .iter()
                        .map(|v| v.map(i64::from)),
                ),
                DataType::Int64 => {
                    values.extend(array.as_any().downcast_ref::<Int64Array>().unwrap().iter())
                }
                other => panic!("unexpected output type: {other}"),
            }
        }
        values.sort();
        values
    }

    async fn comparisons(&self, literals: &[i64]) {
        for &literal in literals {
            for op in [
                PredicateOperator::Eq,
                PredicateOperator::NotEq,
                PredicateOperator::LessThan,
                PredicateOperator::LessThanOrEq,
                PredicateOperator::GreaterThan,
                PredicateOperator::GreaterThanOrEq,
            ] {
                let column = Reference::new("value");
                let datum = self.datum(literal);
                let predicate = match op {
                    PredicateOperator::Eq => column.equal_to(datum),
                    PredicateOperator::NotEq => column.not_equal_to(datum),
                    PredicateOperator::LessThan => column.less_than(datum),
                    PredicateOperator::LessThanOrEq => column.less_than_or_equal_to(datum),
                    PredicateOperator::GreaterThan => column.greater_than(datum),
                    PredicateOperator::GreaterThanOrEq => column.greater_than_or_equal_to(datum),
                    _ => unreachable!(),
                };
                let mut expected = self
                    .values
                    .iter()
                    .copied()
                    .filter(|value| {
                        value.is_some_and(|value| match op {
                            PredicateOperator::Eq => value == literal,
                            PredicateOperator::NotEq => value != literal,
                            PredicateOperator::LessThan => value < literal,
                            PredicateOperator::LessThanOrEq => value <= literal,
                            PredicateOperator::GreaterThan => value > literal,
                            PredicateOperator::GreaterThanOrEq => value >= literal,
                            _ => unreachable!(),
                        })
                    })
                    .collect::<Vec<_>>();
                expected.sort();
                assert_eq!(self.scan(predicate).await, expected, "{op} {literal}");
            }
        }
    }
}

#[tokio::test]
async fn signed_narrow_filters_cover_endpoints_and_outside_domain() {
    let values = vec![
        Some(i16::MIN),
        Some(-1),
        Some(0),
        Some(1),
        Some(i16::MAX),
        None,
    ];
    let fixture = Fixture::new(
        Arc::new(Int16Array::from(values.clone())),
        values.into_iter().map(|v| v.map(i64::from)).collect(),
        PrimitiveType::Int,
    )
    .await;
    fixture
        .comparisons(&[
            i64::from(i32::MIN),
            -32769,
            -32768,
            -1,
            0,
            32767,
            32768,
            i64::from(i32::MAX),
        ])
        .await;
}

#[tokio::test]
async fn unsigned_narrow_filters_cover_negative_and_unsigned_endpoints() {
    let values = vec![
        Some(0_u16),
        Some(1),
        Some(32767),
        Some(32768),
        Some(u16::MAX),
        None,
    ];
    let fixture = Fixture::new(
        Arc::new(UInt16Array::from(values.clone())),
        values.into_iter().map(|v| v.map(i64::from)).collect(),
        PrimitiveType::Int,
    )
    .await;
    fixture
        .comparisons(&[-1, 0, 1, 32767, 32768, 65535, 65536, i64::from(i32::MAX)])
        .await;
}

#[tokio::test]
async fn evolved_long_filters_keep_physical_int32_comparisons() {
    let values = vec![Some(i32::MIN), Some(-1), Some(0), Some(i32::MAX), None];
    let fixture = Fixture::new(
        Arc::new(Int32Array::from(values.clone())),
        values.into_iter().map(|v| v.map(i64::from)).collect(),
        PrimitiveType::Long,
    )
    .await;
    fixture
        .comparisons(&[
            i64::MIN,
            i64::from(i32::MIN) - 1,
            i64::from(i32::MIN),
            0,
            i64::from(i32::MAX),
            i64::from(i32::MAX) + 1,
            i64::MAX,
        ])
        .await;
}

#[tokio::test]
async fn narrowed_sets_and_negation_preserve_null_filtering() {
    let values = vec![Some(-1_i16), Some(0), Some(1), None];
    let fixture = Fixture::new(
        Arc::new(Int16Array::from(values.clone())),
        values.into_iter().map(|v| v.map(i64::from)).collect(),
        PrimitiveType::Int,
    )
    .await;
    let outside = || [Datum::int(-40000), Datum::int(40000)];
    assert_eq!(
        fixture.scan(Reference::new("value").is_in(outside())).await,
        vec![]
    );
    assert_eq!(
        fixture
            .scan(Reference::new("value").is_not_in(outside()))
            .await,
        vec![Some(-1), Some(0), Some(1)]
    );
    assert_eq!(
        fixture
            .scan(Reference::new("value").is_in([
                Datum::int(-40000),
                Datum::int(0),
                Datum::int(40000)
            ]))
            .await,
        vec![Some(0)]
    );
    assert_eq!(
        fixture
            .scan(Reference::new("value").is_not_in([
                Datum::int(-40000),
                Datum::int(0),
                Datum::int(40000)
            ]))
            .await,
        vec![Some(-1), Some(1)]
    );
    assert_eq!(
        fixture
            .scan(!Reference::new("value").equal_to(Datum::int(40000)))
            .await,
        vec![Some(-1), Some(0), Some(1)]
    );
    assert_eq!(
        fixture
            .scan(
                !(Reference::new("value")
                    .equal_to(Datum::int(40000))
                    .or(Reference::new("value").equal_to(Datum::int(0))))
            )
            .await,
        vec![Some(-1), Some(1)]
    );
    let empty = BoundPredicate::Set(SetExpression::new(
        PredicateOperator::NotIn,
        Reference::new("value")
            .bind(Arc::new(fixture.task.schema().clone()), true)
            .unwrap(),
        std::iter::empty().collect(),
    ));
    assert_eq!(fixture.scan_bound(empty).await, vec![
        None,
        Some(-1),
        Some(0),
        Some(1)
    ]);
}
