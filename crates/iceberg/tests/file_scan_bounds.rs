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

use iceberg::scan::FileScanTask;
use iceberg::spec::{DataFileFormat, Datum, NestedField, PrimitiveType, Schema, Type};

#[test]
fn file_bounds_survive_serialization_and_byte_ranges() {
    let schema = Arc::new(
        Schema::builder()
            .with_fields(vec![
                Arc::new(NestedField::required(
                    1,
                    "key",
                    Type::Primitive(PrimitiveType::Long),
                )),
                Arc::new(NestedField::optional(
                    2,
                    "float",
                    Type::Primitive(PrimitiveType::Float),
                )),
                Arc::new(NestedField::optional(
                    3,
                    "double",
                    Type::Primitive(PrimitiveType::Double),
                )),
            ])
            .build()
            .unwrap(),
    );
    let task = FileScanTask::builder()
        .with_file_size_in_bytes(100)
        .with_start(0)
        .with_length(100)
        .with_record_count(Some(10))
        .with_data_file_path("part.parquet".to_owned())
        .with_data_file_format(DataFileFormat::Parquet)
        .with_schema(schema)
        .with_project_field_ids(vec![1, 2, 3])
        .with_case_sensitive(true)
        .with_lower_bounds(Arc::new(HashMap::from([
            (1, Datum::long(i64::MIN)),
            (2, Datum::float(f32::NEG_INFINITY)),
            (3, Datum::double(f64::NEG_INFINITY)),
        ])))
        .with_upper_bounds(Arc::new(HashMap::from([
            (1, Datum::long(i64::MAX)),
            (2, Datum::float(f32::INFINITY)),
            (3, Datum::double(f64::INFINITY)),
        ])))
        .build()
        .unwrap();
    let sliced = task.clone().with_byte_range(25, 50).unwrap();
    assert!(std::ptr::eq(task.lower_bounds(), sliced.lower_bounds()));
    assert!(std::ptr::eq(task.upper_bounds(), sliced.upper_bounds()));
    let encoded = serde_json::to_value(&sliced).unwrap();
    let decoded: FileScanTask = serde_json::from_value(encoded.clone()).unwrap();
    assert_eq!(decoded, sliced);
    let mut legacy = encoded;
    legacy.as_object_mut().unwrap().remove("lower_bounds");
    legacy.as_object_mut().unwrap().remove("upper_bounds");
    let decoded: FileScanTask = serde_json::from_value(legacy.clone()).unwrap();
    assert!(decoded.lower_bounds().is_empty());
    assert!(decoded.upper_bounds().is_empty());
    assert_eq!(serde_json::to_value(decoded).unwrap(), legacy);
}
