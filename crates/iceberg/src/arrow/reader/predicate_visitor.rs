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

//! Visitors that translate Iceberg bound predicates into the pieces needed for
//! Arrow-level evaluation: collecting referenced field IDs and producing
//! per-record-batch predicate closures.

use std::collections::{HashMap, HashSet};
use std::result::Result as StdResult;
use std::sync::Arc;

use arrow_arith::boolean::{and, and_kleene, is_not_null, is_null, not, or, or_kleene};
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type};
use arrow_array::{Array, ArrayRef, BooleanArray, Datum as ArrowDatum, RecordBatch, Scalar};
use arrow_buffer::BooleanBuffer;
use arrow_cast::cast::cast;
use arrow_ord::cmp::{eq, gt, gt_eq, lt, lt_eq, neq};
use arrow_schema::{ArrowError, DataType};
use arrow_string::like::starts_with;
use fnv::FnvHashSet;
use parquet::schema::types::SchemaDescriptor;

use crate::arrow::get_arrow_datum;
use crate::error::{Result, invalid_data};
use crate::expr::visitors::bound_predicate_visitor::BoundPredicateVisitor;
use crate::expr::{BoundPredicate, BoundReference};
use crate::spec::Datum;

/// A visitor to collect field ids from bound predicates.
pub(super) struct CollectFieldIdVisitor {
    pub(super) field_ids: HashSet<i32>,
}

impl CollectFieldIdVisitor {
    pub(super) fn field_ids(self) -> HashSet<i32> {
        self.field_ids
    }
}

impl BoundPredicateVisitor for CollectFieldIdVisitor {
    type T = ();

    fn always_true(&mut self) -> Result<()> {
        Ok(())
    }

    fn always_false(&mut self) -> Result<()> {
        Ok(())
    }

    fn and(&mut self, _lhs: (), _rhs: ()) -> Result<()> {
        Ok(())
    }

    fn or(&mut self, _lhs: (), _rhs: ()) -> Result<()> {
        Ok(())
    }

    fn not(&mut self, _inner: ()) -> Result<()> {
        Ok(())
    }

    fn is_null(&mut self, reference: &BoundReference, _predicate: &BoundPredicate) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn not_null(&mut self, reference: &BoundReference, _predicate: &BoundPredicate) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn is_nan(&mut self, reference: &BoundReference, _predicate: &BoundPredicate) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn not_nan(&mut self, reference: &BoundReference, _predicate: &BoundPredicate) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn less_than(
        &mut self,
        reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn less_than_or_eq(
        &mut self,
        reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn greater_than(
        &mut self,
        reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn greater_than_or_eq(
        &mut self,
        reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn eq(
        &mut self,
        reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn not_eq(
        &mut self,
        reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn starts_with(
        &mut self,
        reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn not_starts_with(
        &mut self,
        reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn r#in(
        &mut self,
        reference: &BoundReference,
        _literals: &FnvHashSet<Datum>,
        _predicate: &BoundPredicate,
    ) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn not_in(
        &mut self,
        reference: &BoundReference,
        _literals: &FnvHashSet<Datum>,
        _predicate: &BoundPredicate,
    ) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }
}

/// A visitor to convert Iceberg bound predicates to Arrow predicates.
pub(super) struct PredicateConverter<'a> {
    /// The Parquet schema descriptor.
    pub(super) parquet_schema: &'a SchemaDescriptor,
    /// The map between field id and leaf column index in Parquet schema.
    pub(super) column_map: &'a HashMap<i32, usize>,
    /// The required column indices in Parquet schema for the predicates.
    pub(super) column_indices: &'a Vec<usize>,
}

impl PredicateConverter<'_> {
    /// When visiting a bound reference, we return index of the leaf column in the
    /// required column indices which is used to project the column in the record batch.
    /// Return None if the field id is not found in the column map, which is possible
    /// due to schema evolution.
    fn bound_reference(&mut self, reference: &BoundReference) -> Result<Option<usize>> {
        // The leaf column's index in Parquet schema.
        if let Some(column_idx) = self.column_map.get(&reference.field().id) {
            if self.parquet_schema.get_column_root(*column_idx).is_group() {
                return Err(invalid_data!(
                    "Leaf column `{}` in predicates isn't a root column in Parquet schema.",
                    reference.field().name
                ));
            }

            // The leaf column's index in the required column indices.
            let index = self
                .column_indices
                .iter()
                .position(|&idx| idx == *column_idx)
                .ok_or(invalid_data!(
                "Leaf column `{}` in predicates cannot be found in the required column indices.",
                reference.field().name
            ))?;

            Ok(Some(index))
        } else {
            Ok(None)
        }
    }

    /// Build an Arrow predicate that always returns true.
    fn build_always_true(&self) -> Result<Box<PredicateResult>> {
        Ok(Box::new(|batch| {
            Ok(constant_bool_array(true, batch.num_rows()))
        }))
    }

    /// Build an Arrow predicate that always returns false.
    fn build_always_false(&self) -> Result<Box<PredicateResult>> {
        Ok(Box::new(|batch| {
            Ok(constant_bool_array(false, batch.num_rows()))
        }))
    }
}

/// Builds a non-null `BooleanArray` of `len` elements all set to `value`.
fn constant_bool_array(value: bool, len: usize) -> BooleanArray {
    let buffer = if value {
        BooleanBuffer::new_set(len)
    } else {
        BooleanBuffer::new_unset(len)
    };

    BooleanArray::new(buffer, None)
}

/// Gets the leaf column from the record batch for the required column index. Only
/// supports top-level columns for now.
fn project_column(batch: &RecordBatch, column_idx: usize) -> StdResult<ArrayRef, ArrowError> {
    let column = batch.column(column_idx);

    match column.data_type() {
        DataType::Struct(_) => Err(ArrowError::SchemaError(
            "Does not support struct column yet.".to_string(),
        )),
        _ => Ok(column.clone()),
    }
}

fn compute_is_nan(array: &ArrayRef) -> StdResult<BooleanArray, ArrowError> {
    // Compute NaN over the contiguous values slice, then fold the null bitmap
    // in with a single bitwise AND so that null slots become false.
    let (is_nan, nulls) = match array.data_type() {
        DataType::Float32 => {
            let arr = array.as_primitive::<Float32Type>();
            (
                BooleanBuffer::from_iter(arr.values().iter().map(|v| v.is_nan())),
                arr.nulls(),
            )
        }
        DataType::Float64 => {
            let arr = array.as_primitive::<Float64Type>();
            (
                BooleanBuffer::from_iter(arr.values().iter().map(|v| v.is_nan())),
                arr.nulls(),
            )
        }
        _ => unreachable!("is_nan is only valid for float types"),
    };

    let values = match nulls {
        Some(nulls) => &is_nan & nulls.inner(),
        None => is_nan,
    };

    Ok(BooleanArray::new(values, None))
}

pub(super) type PredicateResult =
    dyn FnMut(RecordBatch) -> StdResult<BooleanArray, ArrowError> + Send + 'static;

impl BoundPredicateVisitor for PredicateConverter<'_> {
    type T = Box<PredicateResult>;

    fn always_true(&mut self) -> Result<Box<PredicateResult>> {
        self.build_always_true()
    }

    fn always_false(&mut self) -> Result<Box<PredicateResult>> {
        self.build_always_false()
    }

    fn and(
        &mut self,
        mut lhs: Box<PredicateResult>,
        mut rhs: Box<PredicateResult>,
    ) -> Result<Box<PredicateResult>> {
        Ok(Box::new(move |batch| {
            let left = lhs(batch.clone())?;
            let right = rhs(batch)?;
            and_kleene(&left, &right)
        }))
    }

    fn or(
        &mut self,
        mut lhs: Box<PredicateResult>,
        mut rhs: Box<PredicateResult>,
    ) -> Result<Box<PredicateResult>> {
        Ok(Box::new(move |batch| {
            let left = lhs(batch.clone())?;
            let right = rhs(batch)?;
            or_kleene(&left, &right)
        }))
    }

    fn not(&mut self, mut inner: Box<PredicateResult>) -> Result<Box<PredicateResult>> {
        Ok(Box::new(move |batch| {
            let pred_ret = inner(batch)?;
            not(&pred_ret)
        }))
    }

    fn is_null(
        &mut self,
        reference: &BoundReference,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            Ok(Box::new(move |batch| {
                let column = project_column(&batch, idx)?;
                is_null(&column)
            }))
        } else {
            // A missing column, treating it as null.
            self.build_always_true()
        }
    }

    fn not_null(
        &mut self,
        reference: &BoundReference,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            Ok(Box::new(move |batch| {
                let column = project_column(&batch, idx)?;
                is_not_null(&column)
            }))
        } else {
            // A missing column, treating it as null.
            self.build_always_false()
        }
    }

    fn is_nan(
        &mut self,
        reference: &BoundReference,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            Ok(Box::new(move |batch| {
                let column = project_column(&batch, idx)?;
                compute_is_nan(&column)
            }))
        } else {
            // A missing column, treating it as null.
            self.build_always_false()
        }
    }

    fn not_nan(
        &mut self,
        reference: &BoundReference,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            Ok(Box::new(move |batch| {
                let column = project_column(&batch, idx)?;
                let is_nan = compute_is_nan(&column)?;
                not(&is_nan)
            }))
        } else {
            // A missing column, treating it as null.
            self.build_always_true()
        }
    }

    fn less_than(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            let literal = get_arrow_datum(literal)?;

            Ok(Box::new(move |batch| {
                let (left, literal) = comparison_operands(project_column(&batch, idx)?, &literal)?;
                lt(&left, literal.as_ref())
            }))
        } else {
            // A missing column, treating it as null.
            self.build_always_true()
        }
    }

    fn less_than_or_eq(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            let literal = get_arrow_datum(literal)?;

            Ok(Box::new(move |batch| {
                let (left, literal) = comparison_operands(project_column(&batch, idx)?, &literal)?;
                lt_eq(&left, literal.as_ref())
            }))
        } else {
            // A missing column, treating it as null.
            self.build_always_true()
        }
    }

    fn greater_than(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            let literal = get_arrow_datum(literal)?;

            Ok(Box::new(move |batch| {
                let (left, literal) = comparison_operands(project_column(&batch, idx)?, &literal)?;
                gt(&left, literal.as_ref())
            }))
        } else {
            // A missing column, treating it as null.
            self.build_always_false()
        }
    }

    fn greater_than_or_eq(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            let literal = get_arrow_datum(literal)?;

            Ok(Box::new(move |batch| {
                let (left, literal) = comparison_operands(project_column(&batch, idx)?, &literal)?;
                gt_eq(&left, literal.as_ref())
            }))
        } else {
            // A missing column, treating it as null.
            self.build_always_false()
        }
    }

    fn eq(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            let literal = get_arrow_datum(literal)?;

            Ok(Box::new(move |batch| {
                let (left, literal) = comparison_operands(project_column(&batch, idx)?, &literal)?;
                eq(&left, literal.as_ref())
            }))
        } else {
            // A missing column, treating it as null.
            self.build_always_false()
        }
    }

    fn not_eq(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            let literal = get_arrow_datum(literal)?;

            Ok(Box::new(move |batch| {
                let (left, literal) = comparison_operands(project_column(&batch, idx)?, &literal)?;
                neq(&left, literal.as_ref())
            }))
        } else {
            // A missing column, treating it as null.
            self.build_always_false()
        }
    }

    fn starts_with(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            let literal = get_arrow_datum(literal)?;

            Ok(Box::new(move |batch| {
                let (left, literal) = comparison_operands(project_column(&batch, idx)?, &literal)?;
                starts_with(&left, literal.as_ref())
            }))
        } else {
            // A missing column, treating it as null.
            self.build_always_false()
        }
    }

    fn not_starts_with(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            let literal = get_arrow_datum(literal)?;

            Ok(Box::new(move |batch| {
                let (left, literal) = comparison_operands(project_column(&batch, idx)?, &literal)?;
                // update here if arrow ever adds a native not_starts_with
                not(&starts_with(&left, literal.as_ref())?)
            }))
        } else {
            // A missing column, treating it as null.
            self.build_always_true()
        }
    }

    fn r#in(
        &mut self,
        reference: &BoundReference,
        literals: &FnvHashSet<Datum>,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if literals.is_empty() {
            return self.build_always_false();
        }
        if let Some(idx) = self.bound_reference(reference)? {
            let literals = literals
                .iter()
                .map(get_arrow_datum)
                .collect::<Result<Vec<_>>>()?;
            let literal_type = common_literal_type(&literals)?.clone();

            Ok(Box::new(move |batch| {
                // update this if arrow ever adds a native is_in kernel
                let left = promote_comparison_column(project_column(&batch, idx)?, &literal_type)?;

                let mut acc = constant_bool_array(false, batch.num_rows());
                for literal in &literals {
                    let (left, literal) = comparison_operands(Arc::clone(&left), literal)?;
                    acc = or(&acc, &eq(&left, literal.as_ref())?)?
                }

                Ok(acc)
            }))
        } else {
            // A missing column, treating it as null.
            self.build_always_false()
        }
    }

    fn not_in(
        &mut self,
        reference: &BoundReference,
        literals: &FnvHashSet<Datum>,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if literals.is_empty() {
            return self.build_always_true();
        }
        if let Some(idx) = self.bound_reference(reference)? {
            let literals = literals
                .iter()
                .map(get_arrow_datum)
                .collect::<Result<Vec<_>>>()?;
            let literal_type = common_literal_type(&literals)?.clone();

            Ok(Box::new(move |batch| {
                // update this if arrow ever adds a native not_in kernel
                let left = promote_comparison_column(project_column(&batch, idx)?, &literal_type)?;
                let mut acc = constant_bool_array(true, batch.num_rows());
                for literal in &literals {
                    let (left, literal) = comparison_operands(Arc::clone(&left), literal)?;
                    acc = and(&acc, &neq(&left, literal.as_ref())?)?
                }

                Ok(acc)
            }))
        } else {
            // A missing column, treating it as null.
            self.build_always_true()
        }
    }
}

/// Returns the shared Arrow type of a nonempty literal list, rejecting mixed types.
fn common_literal_type(literals: &[Arc<dyn ArrowDatum + Send + Sync>]) -> Result<&DataType> {
    let (first, rest) = literals
        .split_first()
        .ok_or_else(|| invalid_data!("Cannot determine the type of an empty literal list"))?;
    let literal_array = first.get().0;
    let literal_type = literal_array.data_type();
    for literal in rest {
        let literal_array = literal.get().0;
        let data_type = literal_array.data_type();
        if data_type != literal_type {
            return Err(invalid_data!(
                "Predicate literals must have the same type: expected {literal_type}, found {data_type}"
            ));
        }
    }
    Ok(literal_type)
}

/// Aligns a column and literal to the same Arrow type for comparison.
fn comparison_operands(
    column: ArrayRef,
    literal: &Arc<dyn ArrowDatum + Send + Sync>,
) -> StdResult<(ArrayRef, Arc<dyn ArrowDatum + Send + Sync>), ArrowError> {
    let literal_array = literal.get().0;
    let literal_type = literal_array.data_type();
    let column = promote_comparison_column(column, literal_type)?;
    if column.data_type() == literal_type {
        return Ok((column, Arc::clone(literal)));
    }
    let literal = cast(literal_array, column.data_type())?;
    Ok((column, Arc::new(Scalar::new(literal))))
}

/// Promotes a column to a wider integer, float, or same-scale decimal literal type.
fn promote_comparison_column(
    column: ArrayRef,
    literal_type: &DataType,
) -> StdResult<ArrayRef, ArrowError> {
    let promoted = match (column.data_type(), literal_type) {
        (DataType::Int32, DataType::Int64) | (DataType::Float32, DataType::Float64) => true,
        (
            DataType::Decimal128(source_precision, source_scale),
            DataType::Decimal128(target_precision, target_scale),
        ) => source_precision < target_precision && source_scale == target_scale,
        _ => false,
    };
    if promoted {
        cast(column.as_ref(), literal_type)
    } else {
        Ok(column)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;

    use arrow_array::{Array, BooleanArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use parquet::schema::parser::parse_message_type;
    use parquet::schema::types::SchemaDescriptor;

    use super::{
        CollectFieldIdVisitor, PredicateConverter, common_literal_type, constant_bool_array,
    };
    use crate::ErrorKind;
    use crate::arrow::get_arrow_datum;
    use crate::expr::visitors::bound_predicate_visitor::visit;
    use crate::expr::{Bind, Predicate, Reference};
    use crate::spec::{Datum, NestedField, PrimitiveType, Schema, SchemaRef, Type};

    #[test]
    fn test_common_literal_type_matches() {
        let literals = [
            get_arrow_datum(&Datum::long(1)).unwrap(),
            get_arrow_datum(&Datum::long(2)).unwrap(),
        ];
        assert_eq!(common_literal_type(&literals).unwrap(), &DataType::Int64);
    }

    #[test]
    fn test_common_literal_type_rejects_mixed_types() {
        let literals = [
            get_arrow_datum(&Datum::long(1)).unwrap(),
            get_arrow_datum(&Datum::double(2.0)).unwrap(),
        ];
        let error = common_literal_type(&literals).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
    }

    #[test]
    fn test_common_literal_type_rejects_empty_list() {
        let error = common_literal_type(&[]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
    }

    fn table_schema_simple() -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_identifier_field_ids(vec![2])
                .with_fields(vec![
                    NestedField::optional(1, "foo", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::required(2, "bar", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(3, "baz", Type::Primitive(PrimitiveType::Boolean)).into(),
                    NestedField::optional(4, "qux", Type::Primitive(PrimitiveType::Float)).into(),
                ])
                .build()
                .unwrap(),
        )
    }

    #[test]
    fn test_collect_field_id() {
        let schema = table_schema_simple();
        let expr = Reference::new("qux").is_null();
        let bound_expr = expr.bind(schema, true).unwrap();

        let mut visitor = CollectFieldIdVisitor {
            field_ids: HashSet::default(),
        };
        visit(&mut visitor, &bound_expr).unwrap();

        let mut expected = HashSet::default();
        expected.insert(4_i32);

        assert_eq!(visitor.field_ids, expected);
    }

    #[test]
    fn test_collect_field_id_with_and() {
        let schema = table_schema_simple();
        let expr = Reference::new("qux")
            .is_null()
            .and(Reference::new("baz").is_null());
        let bound_expr = expr.bind(schema, true).unwrap();

        let mut visitor = CollectFieldIdVisitor {
            field_ids: HashSet::default(),
        };
        visit(&mut visitor, &bound_expr).unwrap();

        let mut expected = HashSet::default();
        expected.insert(4_i32);
        expected.insert(3);

        assert_eq!(visitor.field_ids, expected);
    }

    #[test]
    fn test_collect_field_id_with_or() {
        let schema = table_schema_simple();
        let expr = Reference::new("qux")
            .is_null()
            .or(Reference::new("baz").is_null());
        let bound_expr = expr.bind(schema, true).unwrap();

        let mut visitor = CollectFieldIdVisitor {
            field_ids: HashSet::default(),
        };
        visit(&mut visitor, &bound_expr).unwrap();

        let mut expected = HashSet::default();
        expected.insert(4_i32);
        expected.insert(3);

        assert_eq!(visitor.field_ids, expected);
    }

    #[test]
    fn test_constant_bool_array() {
        for len in [0, 8192] {
            let all_true = constant_bool_array(true, len);
            assert_eq!(all_true.len(), len);
            assert_eq!(all_true.null_count(), 0);
            assert!(all_true.iter().all(|v| v == Some(true)));

            let all_false = constant_bool_array(false, len);
            assert_eq!(all_false.len(), len);
            assert_eq!(all_false.null_count(), 0);
            assert!(all_false.iter().all(|v| v == Some(false)));
        }
    }

    fn apply_predicate_to_batch(
        predicate: Predicate,
        schema: SchemaRef,
        batch: RecordBatch,
    ) -> BooleanArray {
        let bound = predicate.bind(schema, true).unwrap();

        // Build a trivial Parquet schema with one float column at field id 4
        let message_type = "
            message schema {
              optional float qux = 4;
            }
        ";
        let parquet_type = parse_message_type(message_type).expect("parse schema");
        let parquet_schema = SchemaDescriptor::new(Arc::new(parquet_type));

        let column_map = HashMap::from([(4i32, 0usize)]);
        let column_indices = vec![0usize];

        let mut converter = PredicateConverter {
            parquet_schema: &parquet_schema,
            column_map: &column_map,
            column_indices: &column_indices,
        };

        let mut predicate_fn = visit(&mut converter, &bound).unwrap();
        predicate_fn(batch).unwrap()
    }

    #[test]
    fn test_predicate_converter_nan() {
        use arrow_array::Float32Array;

        let schema = table_schema_simple();
        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "qux",
            DataType::Float32,
            true,
        )]));
        let values = vec![Some(1.0f32), Some(f32::NAN), None, Some(0.0f32)];

        // is_nan: non-null-propagating per Java's implementation - NULL → false
        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(Float32Array::from(
            values.clone(),
        ))])
        .unwrap();
        let result =
            apply_predicate_to_batch(Reference::new("qux").is_nan(), schema.clone(), batch);
        assert_eq!(
            [
                result.value(0),
                result.value(1),
                result.value(2),
                result.value(3)
            ],
            [false, true, false, false]
        );
        assert!(!result.is_null(2));

        // not_nan: non-null-propagating per Java's implementation - NULL → true
        let batch =
            RecordBatch::try_new(arrow_schema, vec![Arc::new(Float32Array::from(values))]).unwrap();
        let result = apply_predicate_to_batch(Reference::new("qux").is_not_nan(), schema, batch);
        assert_eq!(
            [
                result.value(0),
                result.value(1),
                result.value(2),
                result.value(3)
            ],
            [true, false, true, true]
        );
        assert!(!result.is_null(2));
    }
}
