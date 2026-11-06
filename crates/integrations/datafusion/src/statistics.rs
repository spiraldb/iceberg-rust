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

use datafusion::arrow::datatypes::{DataType, Schema};
use datafusion::common::Statistics;
use datafusion::common::stats::Precision;

pub(crate) fn output_bytes(statistics: &mut Statistics, schema: &Schema) {
    let rows = statistics.num_rows.get_value().copied();
    let mut total = rows.map(|_| 0usize);
    for (column, field) in statistics.column_statistics.iter_mut().zip(schema.fields()) {
        let bytes = rows.and_then(|rows| {
            if rows == 0 {
                return Some(0);
            }
            let values = match field.data_type() {
                DataType::Boolean => rows.div_ceil(8),
                DataType::FixedSizeBinary(width) => {
                    rows.checked_mul(usize::try_from(*width).ok()?)?
                }
                data_type => rows.checked_mul(data_type.primitive_width()?)?,
            };
            values.checked_add(if field.is_nullable() {
                rows.div_ceil(8)
            } else {
                0
            })
        });
        column.byte_size = bytes.map_or(Precision::Absent, Precision::Inexact);
        total = total
            .zip(bytes)
            .and_then(|(total, bytes)| total.checked_add(bytes));
    }
    statistics.total_byte_size = total.map_or(Precision::Absent, Precision::Inexact);
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::datatypes::Field;
    use datafusion::common::stats::Precision::{Absent, Exact, Inexact};

    fn byte_sizes(statistics: &Statistics) -> (Precision<usize>, Vec<Precision<usize>>) {
        (
            statistics.total_byte_size,
            statistics
                .column_statistics
                .iter()
                .map(|column| column.byte_size)
                .collect(),
        )
    }

    #[test]
    fn projections_estimate_values_and_nullable_bitmaps() {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Int32, true),
            Field::new("flag", DataType::Boolean, false),
            Field::new("optional_flag", DataType::Boolean, true),
            Field::new("uuid", DataType::FixedSizeBinary(16), false),
        ]);
        let mut statistics = Statistics::new_unknown(&schema).with_num_rows(Exact(9));
        output_bytes(&mut statistics, &schema);
        assert_eq!(
            byte_sizes(&statistics),
            (
                Inexact(260),
                vec![Inexact(72), Inexact(38), Inexact(2), Inexact(4), Inexact(144)]
            )
        );
        let projection = schema.project(&[3, 1]).unwrap();
        let mut projected = Statistics::new_unknown(&projection).with_num_rows(Exact(9));
        output_bytes(&mut projected, &projection);
        assert_eq!(
            (projected.num_rows, byte_sizes(&projected)),
            (Exact(9), (Inexact(42), vec![Inexact(4), Inexact(38)]))
        );
    }

    #[test]
    fn unknown_columns_and_row_counts_keep_byte_estimates_unknown() {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("label", DataType::Utf8, true),
            Field::new("nested", DataType::Struct(Default::default()), true),
        ]);
        let mut statistics = Statistics::new_unknown(&schema)
            .with_num_rows(Inexact(4))
            .with_total_byte_size(Exact(12345));
        output_bytes(&mut statistics, &schema);
        assert_eq!(
            (statistics.num_rows, byte_sizes(&statistics)),
            (Inexact(4), (Absent, vec![Inexact(32), Absent, Absent]))
        );
        statistics.num_rows = Absent;
        output_bytes(&mut statistics, &schema);
        assert_eq!(byte_sizes(&statistics), (Absent, vec![Absent; 3]));
        statistics.num_rows = Exact(0);
        output_bytes(&mut statistics, &schema);
        assert_eq!(byte_sizes(&statistics), (Inexact(0), vec![Inexact(0); 3]));
    }

    #[test]
    fn overflowing_columns_and_totals_stay_unknown() {
        for (fields, expected) in [
            (
                vec![Field::new("wide", DataType::Int64, false)],
                vec![Absent],
            ),
            (
                vec![
                    Field::new("left", DataType::Int8, false),
                    Field::new("right", DataType::Int8, false),
                ],
                vec![Inexact(usize::MAX), Inexact(usize::MAX)],
            ),
            (
                vec![Field::new("nullable", DataType::Int8, true)],
                vec![Absent],
            ),
        ] {
            let schema = Schema::new(fields);
            let mut statistics =
                Statistics::new_unknown(&schema).with_num_rows(Exact(usize::MAX));
            output_bytes(&mut statistics, &schema);
            assert_eq!(byte_sizes(&statistics), (Absent, expected));
        }
    }
}
