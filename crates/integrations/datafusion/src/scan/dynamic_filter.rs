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
use datafusion::common::Result;
use datafusion::common::cast::as_boolean_array;
use datafusion::common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::{CastExpr, Column};
use datafusion::physical_expr::utils::collect_columns;
use datafusion::physical_plan::metrics::Count;
use iceberg::scan::FileScanTask;
use parquet::arrow::arrow_reader::{ArrowPredicate, ArrowPredicateFn, RowFilter};
use parquet::arrow::{PARQUET_FIELD_ID_META_KEY, ProjectionMask};
use parquet::file::metadata::ParquetMetaData;

struct AdaptedFilter {
    predicate: Arc<dyn PhysicalExpr>,
    leaves: Vec<usize>,
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
