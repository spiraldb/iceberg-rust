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

use datafusion::arrow::datatypes::{DataType, FieldRef, SchemaRef};
use datafusion::arrow::error::ArrowError;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::Result;
use datafusion::common::cast::as_boolean_array;
use datafusion::common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion::logical_expr::{ColumnarValue, Operator};
use datafusion::physical_expr::expressions::{
    BinaryExpr, CastExpr, Column, InListExpr, IsNotNullExpr, IsNullExpr, Literal, NotExpr,
};
use datafusion::physical_expr::utils::collect_columns;
use datafusion::physical_expr::{DynamicFilterTracking, PhysicalExpr};
use datafusion::physical_expr_common::physical_expr::snapshot_physical_expr;
use datafusion::physical_plan::metrics::Count;
use datafusion::scalar::ScalarValue;
use iceberg::scan::FileScanTask;
use parquet::arrow::arrow_reader::{ArrowPredicate, ArrowPredicateFn, RowFilter};
use parquet::arrow::{PARQUET_FIELD_ID_META_KEY, ProjectionMask};
use parquet::file::metadata::ParquetMetaData;

const MAX_DYNAMIC_LITERAL_VALUES: usize = 150;

struct AdaptedFilter {
    predicate: Arc<dyn PhysicalExpr>,
    leaves: Vec<usize>,
}

struct LiveCheapFilter {
    live: Arc<dyn PhysicalExpr>,
    tracking: DynamicFilterTracking,
    cached: Option<Arc<dyn PhysicalExpr>>,
}

impl LiveCheapFilter {
    fn new(live: Arc<dyn PhysicalExpr>) -> Result<Option<Self>> {
        let tracking = DynamicFilterTracking::classify(&live);
        let cached = cheap_predicate(&snapshot_physical_expr(live.clone())?, true);
        if !matches!(tracking, DynamicFilterTracking::Watching(_)) && cached.is_none() {
            return Ok(None);
        }
        Ok(Some(Self {
            live,
            tracking,
            cached,
        }))
    }

    fn evaluate(&mut self, batch: &RecordBatch) -> Result<ColumnarValue> {
        if self
            .tracking
            .watcher()
            .is_some_and(|watcher| watcher.changed())
        {
            self.cached = cheap_predicate(&snapshot_physical_expr(self.live.clone())?, true);
        }
        match &self.cached {
            Some(predicate) => predicate.evaluate(batch),
            None => Ok(ColumnarValue::Scalar(ScalarValue::Boolean(Some(true)))),
        }
    }
}

fn cheap_predicate(
    expr: &Arc<dyn PhysicalExpr>,
    partial_conjunction: bool,
) -> Option<Arc<dyn PhysicalExpr>> {
    if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
        return match binary.op() {
            Operator::And => {
                let left = cheap_predicate(binary.left(), partial_conjunction);
                let right = cheap_predicate(binary.right(), partial_conjunction);
                match (left, right) {
                    (Some(left), Some(right)) => {
                        Some(Arc::new(BinaryExpr::new(left, Operator::And, right)))
                    }
                    (left, right) if partial_conjunction => left.or(right),
                    _ => None,
                }
            }
            Operator::Or => {
                let left = cheap_predicate(binary.left(), false)?;
                let right = cheap_predicate(binary.right(), false)?;
                Some(Arc::new(BinaryExpr::new(left, Operator::Or, right)))
            }
            Operator::Eq
            | Operator::NotEq
            | Operator::Lt
            | Operator::LtEq
            | Operator::Gt
            | Operator::GtEq
                if cheap_scalar(binary.left()) && cheap_scalar(binary.right()) =>
            {
                Some(expr.clone())
            }
            _ => None,
        };
    }
    if let Some(literal) = expr.downcast_ref::<Literal>() {
        return match literal.value() {
            ScalarValue::Boolean(Some(false) | None) => Some(expr.clone()),
            _ => None,
        };
    }
    if let Some(list) = expr.downcast_ref::<InListExpr>() {
        return (list.len() <= MAX_DYNAMIC_LITERAL_VALUES
            && cheap_scalar(list.expr())
            && list.list().iter().all(|value| value.is::<Literal>()))
        .then(|| expr.clone());
    }
    if let Some(is_null) = expr.downcast_ref::<IsNullExpr>() {
        return cheap_scalar(is_null.arg()).then(|| expr.clone());
    }
    if let Some(is_not_null) = expr.downcast_ref::<IsNotNullExpr>() {
        return cheap_scalar(is_not_null.arg()).then(|| expr.clone());
    }
    if let Some(not) = expr.downcast_ref::<NotExpr>() {
        let inner = cheap_predicate(not.arg(), false)?;
        return Some(Arc::new(NotExpr::new(inner)));
    }
    None
}

fn cheap_scalar(expr: &Arc<dyn PhysicalExpr>) -> bool {
    expr.is::<Column>() || expr.is::<Literal>()
}

pub(super) fn row_filter(
    filters: &[Arc<dyn PhysicalExpr>],
    logical_schema: &SchemaRef,
    task: &FileScanTask,
    physical_schema: &SchemaRef,
    metadata: &Arc<ParquetMetaData>,
    pruned: &Count,
) -> Result<Option<RowFilter>> {
    let mut predicates: Vec<Box<dyn ArrowPredicate>> = Vec::new();
    for filter in filters {
        let Some(AdaptedFilter {
            predicate: filter,
            leaves,
        }) = adapt(filter, logical_schema, task, physical_schema, metadata)?
        else {
            continue;
        };
        let Some(mut filter) = LiveCheapFilter::new(filter)? else {
            continue;
        };
        let projection = ProjectionMask::leaves(metadata.file_metadata().schema_descr(), leaves);
        let pruned = pruned.clone();
        predicates.push(Box::new(ArrowPredicateFn::new(projection, move |batch| {
            filter
                .evaluate(&batch)
                .and_then(|value| value.into_array(batch.num_rows()))
                .and_then(|array| {
                    let matches = as_boolean_array(&array)?.clone();
                    pruned.add(matches.len() - matches.true_count());
                    Ok(matches)
                })
                .map_err(|error| ArrowError::ExternalError(Box::new(error)))
        })));
    }
    Ok((!predicates.is_empty()).then(|| RowFilter::new(predicates)))
}

fn adapt(
    filter: &Arc<dyn PhysicalExpr>,
    logical_schema: &SchemaRef,
    task: &FileScanTask,
    physical_schema: &SchemaRef,
    metadata: &ParquetMetaData,
) -> Result<Option<AdaptedFilter>> {
    let parquet_schema = metadata.file_metadata().schema_descr();
    let mut fields: HashMap<Column, (usize, FieldRef)> = HashMap::new();
    let mut roots = Vec::new();
    let mut leaves = Vec::new();
    // Dynamic nodes declare their columns even while their current expression is TRUE.
    for column in collect_columns(filter) {
        let Some(logical) = logical_schema.fields().get(column.index()) else {
            return Ok(None);
        };
        if logical.name() != column.name() {
            return Ok(None);
        }
        let Some(id) = field_id(logical) else {
            return Ok(None);
        };
        if !task
            .schema()
            .as_struct()
            .fields()
            .iter()
            .any(|field| field.id == id && field.field_type.is_primitive())
        {
            return Ok(None);
        }
        let mut matching = physical_schema
            .fields()
            .iter()
            .enumerate()
            .filter(|(_, field)| field_id(field) == Some(id));
        let Some((root, physical)) = matching.next() else {
            return Ok(None);
        };
        if matching.next().is_some() || !lossless_type(physical.data_type(), logical.data_type()) {
            return Ok(None);
        }
        let mut matching_leaves = (0..parquet_schema.num_columns())
            .filter(|&leaf| parquet_schema.get_column_root_idx(leaf) == root);
        let Some(leaf) = matching_leaves.next() else {
            return Ok(None);
        };
        if matching_leaves.next().is_some() || parquet_schema.get_column_root(leaf).is_group() {
            return Ok(None);
        }
        roots.push(root);
        leaves.push(leaf);
        fields.insert(column, (root, physical.clone()));
    }
    roots.sort_unstable();
    roots.dedup();
    leaves.sort_unstable();
    leaves.dedup();
    let filter = filter
        .clone()
        .transform_up(|expr| {
            let Some(column) = expr.downcast_ref::<Column>() else {
                return Ok(Transformed::no(expr));
            };
            let (root, physical) = &fields[column];
            let index = roots
                .binary_search(root)
                .expect("filter roots were collected from its columns");
            let mut replacement: Arc<dyn PhysicalExpr> =
                Arc::new(Column::new(physical.name(), index));
            let logical = &logical_schema.fields()[column.index()];
            if physical.data_type() != logical.data_type() {
                replacement = Arc::new(CastExpr::new_with_target_field(
                    replacement,
                    logical.clone(),
                    None,
                ));
            }
            Ok(Transformed::yes(replacement))
        })
        .data()?;
    Ok(Some(AdaptedFilter {
        predicate: filter,
        leaves,
    }))
}

fn field_id(field: &datafusion::arrow::datatypes::Field) -> Option<i32> {
    field
        .metadata()
        .get(PARQUET_FIELD_ID_META_KEY)?
        .parse()
        .ok()
}

fn lossless_type(physical: &DataType, logical: &DataType) -> bool {
    physical == logical
        || match (physical, logical) {
            (DataType::Int32, DataType::Int64) | (DataType::Float32, DataType::Float64) => true,
            (DataType::Decimal128(from, from_scale), DataType::Decimal128(to, to_scale)) => {
                from <= to && from_scale == to_scale
            }
            (
                DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View,
                DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View,
            ) => true,
            (
                DataType::Binary | DataType::LargeBinary | DataType::BinaryView,
                DataType::Binary | DataType::LargeBinary | DataType::BinaryView,
            ) => true,
            _ => false,
        }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs::File;
    use std::sync::Arc;

    use datafusion::arrow::array::Int64Array;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::arrow::record_batch::RecordBatch;
    use datafusion::common::cast::as_boolean_array;
    use datafusion::logical_expr::Operator;
    use datafusion::physical_expr::PhysicalExpr;
    use datafusion::physical_expr::expressions::{
        BinaryExpr, CaseExpr, CastExpr, Column, DynamicFilterPhysicalExpr, InListExpr, NotExpr, lit,
    };
    use datafusion::physical_plan::metrics::Count;
    use iceberg::scan::FileScanTask;
    use iceberg::spec::{DataFileFormat, NestedField, PrimitiveType};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};

    use super::{LiveCheapFilter, row_filter};

    fn batch() -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("key", DataType::Int64, true).with_metadata(HashMap::from([(
                    PARQUET_FIELD_ID_META_KEY.to_owned(),
                    "1".to_owned(),
                )])),
            ])),
            vec![Arc::new(Int64Array::from(vec![
                Some(0),
                Some(1),
                None,
                Some(2),
                Some(3),
            ]))],
        )
        .unwrap()
    }

    fn key() -> Arc<dyn PhysicalExpr> {
        Arc::new(Column::new("key", 0))
    }

    fn comparison(op: Operator, value: i64) -> Arc<dyn PhysicalExpr> {
        Arc::new(BinaryExpr::new(key(), op, lit(value)))
    }

    fn unsupported_case(value: bool) -> Arc<dyn PhysicalExpr> {
        Arc::new(CaseExpr::try_new(None, vec![(lit(true), lit(value))], Some(lit(value))).unwrap())
    }

    fn selected(state: &mut LiveCheapFilter, batch: &RecordBatch) -> Vec<Option<i64>> {
        let mask = state
            .evaluate(batch)
            .unwrap()
            .into_array(batch.num_rows())
            .unwrap();
        let mask = as_boolean_array(mask.as_ref()).unwrap();
        batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .iter()
            .zip(mask.iter())
            .filter_map(|(value, keep)| (keep == Some(true)).then_some(value))
            .collect()
    }

    #[test]
    fn composite_filter_cache_tracks_all_live_generations() {
        let batch = batch();
        let first = Arc::new(DynamicFilterPhysicalExpr::new(vec![key()], lit(true)));
        let second = Arc::new(DynamicFilterPhysicalExpr::new(vec![key()], lit(true)));
        let root: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            first.clone(),
            Operator::And,
            second.clone(),
        ));
        let mut state = LiveCheapFilter::new(root).unwrap().unwrap();
        assert_eq!(selected(&mut state, &batch), vec![
            Some(0),
            Some(1),
            None,
            Some(2),
            Some(3)
        ]);
        first.update(comparison(Operator::GtEq, 1)).unwrap();
        assert_eq!(selected(&mut state, &batch), vec![
            Some(1),
            Some(2),
            Some(3)
        ]);
        second.update(comparison(Operator::LtEq, 2)).unwrap();
        assert_eq!(selected(&mut state, &batch), vec![Some(1), Some(2)]);
        let cached = state.cached.clone().unwrap();
        assert_eq!(selected(&mut state, &batch), vec![Some(1), Some(2)]);
        assert!(Arc::ptr_eq(&cached, state.cached.as_ref().unwrap()));
    }

    #[test]
    fn cheap_subsets_preserve_boolean_polarity_and_bound_list_cost() {
        let batch = batch();
        let lower = comparison(Operator::GtEq, 2);
        let conjunction: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            lower.clone(),
            Operator::And,
            unsupported_case(false),
        ));
        let mut state = LiveCheapFilter::new(conjunction.clone()).unwrap().unwrap();
        assert_eq!(selected(&mut state, &batch), vec![Some(2), Some(3)]);
        let disjunction: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(lower, Operator::Or, unsupported_case(true)));
        assert!(LiveCheapFilter::new(disjunction).unwrap().is_none());
        assert!(
            LiveCheapFilter::new(Arc::new(NotExpr::new(conjunction)))
                .unwrap()
                .is_none()
        );
        let short: Arc<dyn PhysicalExpr> = Arc::new(
            InListExpr::try_new_from_array(
                key(),
                Arc::new(Int64Array::from(vec![1, 3])),
                false,
                batch.schema().as_ref(),
            )
            .unwrap(),
        );
        let mut state = LiveCheapFilter::new(short).unwrap().unwrap();
        assert_eq!(selected(&mut state, &batch), vec![Some(1), Some(3)]);
        let long: Arc<dyn PhysicalExpr> = Arc::new(
            InListExpr::try_new_from_array(
                key(),
                Arc::new(Int64Array::from_iter_values(0..151)),
                false,
                batch.schema().as_ref(),
            )
            .unwrap(),
        );
        assert!(LiveCheapFilter::new(long).unwrap().is_none());
        let failing_cast: Arc<dyn PhysicalExpr> =
            Arc::new(CastExpr::new(lit("not_an_integer"), DataType::Int64, None));
        let comparison: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(failing_cast, Operator::Gt, lit(1_i64)));
        assert!(LiveCheapFilter::new(comparison).unwrap().is_none());
    }

    #[test]
    fn completed_expensive_filter_does_not_install_a_decoder_predicate() {
        let batch = batch();
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("keys.parquet");
        let mut writer =
            ArrowWriter::try_new(File::create(&path).unwrap(), batch.schema(), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        let metadata = ParquetRecordBatchReaderBuilder::try_new(File::open(&path).unwrap())
            .unwrap()
            .metadata()
            .clone();
        let schema = iceberg::spec::Schema::builder()
            .with_fields(vec![
                NestedField::optional(1, "key", PrimitiveType::Long.into()).into(),
            ])
            .build()
            .unwrap();
        let task = FileScanTask::builder()
            .with_file_size_in_bytes(std::fs::metadata(&path).unwrap().len())
            .with_start(0)
            .with_length(0)
            .with_data_file_path(path.to_string_lossy().into_owned())
            .with_data_file_format(DataFileFormat::Parquet)
            .with_schema(Arc::new(schema))
            .with_project_field_ids(vec![1])
            .with_case_sensitive(true)
            .build()
            .unwrap();
        let dynamic = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![key()],
            unsupported_case(false),
        ));
        dynamic.mark_complete();
        let filters: Vec<Arc<dyn PhysicalExpr>> = vec![dynamic];
        assert!(
            row_filter(
                &filters,
                &batch.schema(),
                &task,
                &batch.schema(),
                &metadata,
                &Count::new()
            )
            .unwrap()
            .is_none()
        );
    }
}
