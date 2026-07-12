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

//! The module contains the file writer for the vortex file format.

use std::collections::HashMap;

use arrow_arith::aggregate::{
    max, max_binary, max_boolean, max_string, min, min_binary, min_boolean, min_string,
};
use arrow_array::{
    Array, BinaryArray, BooleanArray, Date32Array, Decimal128Array, Float32Array, Float64Array,
    Int32Array, Int64Array, RecordBatch, StringArray, Time64MicrosecondArray,
    TimestampMicrosecondArray, TimestampNanosecondArray,
};
use bytes::Bytes;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrow::FromArrowArray;
use vortex::array::{ArrayRef, IntoArray};
use vortex::dtype::DType;
use vortex::dtype::arrow::FromArrowType;
use vortex::file::WriteOptionsSessionExt;

use super::{FileWriter, FileWriterBuilder};
use crate::Result;
use crate::arrow::{FieldMatchMode, NanValueCountVisitor, to_iceberg_error, vortex_session};
use crate::io::OutputFile;
use crate::spec::{
    DataContentType, DataFileBuilder, DataFileFormat, Datum, PrimitiveLiteral, PrimitiveType,
    SchemaRef, Struct, Type,
};

/// VortexWriterBuilder is used to build a [`VortexWriter`].
#[derive(Clone, Debug)]
pub struct VortexWriterBuilder {
    schema: SchemaRef,
    match_mode: FieldMatchMode,
}

impl VortexWriterBuilder {
    /// Create a new `VortexWriterBuilder`.
    pub fn new(schema: SchemaRef) -> Self {
        Self {
            schema,
            match_mode: FieldMatchMode::Id,
        }
    }

    /// Set the field match mode used to map Arrow fields to Iceberg fields.
    ///
    /// Defaults to [`FieldMatchMode::Id`]. Use [`FieldMatchMode::Name`] when the
    /// incoming Arrow schema does not carry Iceberg field-id metadata.
    pub fn with_match_mode(mut self, match_mode: FieldMatchMode) -> Self {
        self.match_mode = match_mode;
        self
    }
}

impl FileWriterBuilder for VortexWriterBuilder {
    type R = VortexWriter;

    async fn build(&self, output_file: OutputFile) -> Result<Self::R> {
        Ok(VortexWriter {
            schema: self.schema.clone(),
            output_file,
            batches: Vec::new(),
            current_row_num: 0,
            buffered_size: 0,
            nan_value_count_visitor: NanValueCountVisitor::new_with_match_mode(self.match_mode),
        })
    }
}

/// `VortexWriter` writes arrow data into vortex files on storage.
///
/// Vortex files are written in a single pass at close time, so incoming record
/// batches are buffered in memory until [`FileWriter::close`] is called.
pub struct VortexWriter {
    schema: SchemaRef,
    output_file: OutputFile,
    batches: Vec<RecordBatch>,
    current_row_num: usize,
    buffered_size: usize,
    nan_value_count_visitor: NanValueCountVisitor,
}

impl VortexWriter {
    fn to_data_file_builder(&self, written_size: usize) -> Result<DataFileBuilder> {
        // Compute value/null counts and min/max bounds for top-level primitive
        // fields directly from the buffered arrow batches. Nested fields are
        // left unset; metrics evaluators treat missing entries as
        // "rows might match".
        let mut value_counts: HashMap<i32, u64> = HashMap::new();
        let mut null_value_counts: HashMap<i32, u64> = HashMap::new();
        let mut lower_bounds: HashMap<i32, Datum> = HashMap::new();
        let mut upper_bounds: HashMap<i32, Datum> = HashMap::new();
        for batch in &self.batches {
            for (field, column) in batch.schema().fields().iter().zip(batch.columns()) {
                let Some(iceberg_field) = self.schema.field_by_name(field.name()) else {
                    continue;
                };
                let Type::Primitive(primitive_type) = iceberg_field.field_type.as_ref() else {
                    continue;
                };
                *value_counts.entry(iceberg_field.id).or_insert(0) += column.len() as u64;
                *null_value_counts.entry(iceberg_field.id).or_insert(0) +=
                    column.null_count() as u64;
                if let Some((min_datum, max_datum)) =
                    column_min_max(column.as_ref(), primitive_type)
                {
                    update_bound_min(&mut lower_bounds, iceberg_field.id, min_datum);
                    update_bound_max(&mut upper_bounds, iceberg_field.id, max_datum);
                }
            }
        }

        let mut builder = DataFileBuilder::default();
        builder
            .content(DataContentType::Data)
            .file_path(self.output_file.location().to_string())
            .file_format(DataFileFormat::Vortex)
            .partition(Struct::empty())
            .record_count(self.current_row_num as u64)
            .file_size_in_bytes(written_size as u64)
            .value_counts(value_counts)
            .null_value_counts(null_value_counts)
            .nan_value_counts(self.nan_value_count_visitor.nan_value_counts.clone())
            .lower_bounds(lower_bounds)
            .upper_bounds(upper_bounds);
        // Vortex files can be split at arbitrary row offsets, so no physical
        // split offsets are recorded.

        Ok(builder)
    }
}

fn update_bound_min(bounds: &mut HashMap<i32, Datum>, field_id: i32, datum: Datum) {
    bounds
        .entry(field_id)
        .and_modify(|entry| {
            if *entry > datum {
                *entry = datum.clone();
            }
        })
        .or_insert(datum);
}

fn update_bound_max(bounds: &mut HashMap<i32, Datum>, field_id: i32, datum: Datum) {
    bounds
        .entry(field_id)
        .and_modify(|entry| {
            if *entry < datum {
                *entry = datum.clone();
            }
        })
        .or_insert(datum);
}

/// Computes the min and max values of an arrow column as iceberg [`Datum`]s.
///
/// Returns `None` when the column is empty or all-null, or when its arrow type
/// does not match the canonical arrow representation of the iceberg type, in
/// which case no bounds are recorded for the field.
fn column_min_max(column: &dyn Array, primitive_type: &PrimitiveType) -> Option<(Datum, Datum)> {
    match primitive_type {
        PrimitiveType::Boolean => {
            let array = column.as_any().downcast_ref::<BooleanArray>()?;
            Some((
                Datum::bool(min_boolean(array)?),
                Datum::bool(max_boolean(array)?),
            ))
        }
        PrimitiveType::Int => {
            let array = column.as_any().downcast_ref::<Int32Array>()?;
            Some((Datum::int(min(array)?), Datum::int(max(array)?)))
        }
        PrimitiveType::Long => {
            let array = column.as_any().downcast_ref::<Int64Array>()?;
            Some((Datum::long(min(array)?), Datum::long(max(array)?)))
        }
        PrimitiveType::Float => {
            let array = column.as_any().downcast_ref::<Float32Array>()?;
            let (min_value, max_value) = float_min_max(array.iter(), |value| value.is_nan())?;
            Some((Datum::float(min_value), Datum::float(max_value)))
        }
        PrimitiveType::Double => {
            let array = column.as_any().downcast_ref::<Float64Array>()?;
            let (min_value, max_value) = float_min_max(array.iter(), |value| value.is_nan())?;
            Some((Datum::double(min_value), Datum::double(max_value)))
        }
        PrimitiveType::Date => {
            let array = column.as_any().downcast_ref::<Date32Array>()?;
            Some((Datum::date(min(array)?), Datum::date(max(array)?)))
        }
        PrimitiveType::Time => {
            let array = column.as_any().downcast_ref::<Time64MicrosecondArray>()?;
            Some((
                Datum::time_micros(min(array)?).ok()?,
                Datum::time_micros(max(array)?).ok()?,
            ))
        }
        PrimitiveType::Timestamp => {
            let array = column
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()?;
            Some((
                Datum::timestamp_micros(min(array)?),
                Datum::timestamp_micros(max(array)?),
            ))
        }
        PrimitiveType::Timestamptz => {
            let array = column
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()?;
            Some((
                Datum::timestamptz_micros(min(array)?),
                Datum::timestamptz_micros(max(array)?),
            ))
        }
        PrimitiveType::TimestampNs => {
            let array = column.as_any().downcast_ref::<TimestampNanosecondArray>()?;
            Some((
                Datum::timestamp_nanos(min(array)?),
                Datum::timestamp_nanos(max(array)?),
            ))
        }
        PrimitiveType::TimestamptzNs => {
            let array = column.as_any().downcast_ref::<TimestampNanosecondArray>()?;
            Some((
                Datum::timestamptz_nanos(min(array)?),
                Datum::timestamptz_nanos(max(array)?),
            ))
        }
        PrimitiveType::String => {
            let array = column.as_any().downcast_ref::<StringArray>()?;
            Some((
                Datum::string(min_string(array)?),
                Datum::string(max_string(array)?),
            ))
        }
        PrimitiveType::Binary => {
            let array = column.as_any().downcast_ref::<BinaryArray>()?;
            Some((
                Datum::binary(min_binary(array)?.iter().copied()),
                Datum::binary(max_binary(array)?.iter().copied()),
            ))
        }
        PrimitiveType::Decimal { .. } => {
            let array = column.as_any().downcast_ref::<Decimal128Array>()?;
            Some((
                Datum::new(
                    primitive_type.clone(),
                    PrimitiveLiteral::Int128(min(array)?),
                ),
                Datum::new(
                    primitive_type.clone(),
                    PrimitiveLiteral::Int128(max(array)?),
                ),
            ))
        }
        // Uuid and Fixed bounds are not computed.
        _ => None,
    }
}

/// Computes the min and max of a float column, skipping nulls and NaN values
/// as required by the iceberg spec (NaN counts are tracked separately).
fn float_min_max<T: PartialOrd + Copy>(
    values: impl Iterator<Item = Option<T>>,
    is_nan: impl Fn(T) -> bool,
) -> Option<(T, T)> {
    let mut bounds: Option<(T, T)> = None;
    for value in values.flatten() {
        if is_nan(value) {
            continue;
        }
        bounds = Some(match bounds {
            None => (value, value),
            Some((min_value, max_value)) => (
                if value < min_value { value } else { min_value },
                if value > max_value { value } else { max_value },
            ),
        });
    }
    bounds
}

impl FileWriter for VortexWriter {
    async fn write(&mut self, batch: &RecordBatch) -> Result<()> {
        // Skip empty batch
        if batch.num_rows() == 0 {
            return Ok(());
        }

        self.current_row_num += batch.num_rows();
        self.buffered_size += batch.get_array_memory_size();
        self.nan_value_count_visitor
            .compute(self.schema.clone(), batch.clone())?;
        self.batches.push(batch.clone());

        Ok(())
    }

    async fn close(self) -> Result<Vec<DataFileBuilder>> {
        if self.batches.is_empty() {
            return Ok(vec![]);
        }

        let dtype = DType::from_arrow(self.batches[0].schema());
        let chunks = self
            .batches
            .iter()
            .map(|batch| ArrayRef::from_arrow(batch.clone(), false))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(to_iceberg_error)?;
        let array = ChunkedArray::try_new(chunks, dtype)
            .map_err(to_iceberg_error)?
            .into_array();

        let session = vortex_session();
        let mut buffer: Vec<u8> = Vec::new();
        session
            .write_options()
            .write(&mut buffer, array.to_array_stream())
            .await
            .map_err(to_iceberg_error)?;

        let written_size = buffer.len();
        self.output_file.write(Bytes::from(buffer)).await?;

        Ok(vec![self.to_data_file_builder(written_size)?])
    }
}

impl super::super::CurrentFileStatus for VortexWriter {
    fn current_file_path(&self) -> String {
        self.output_file.location().to_string()
    }

    fn current_row_num(&self) -> usize {
        self.current_row_num
    }

    fn current_written_size(&self) -> usize {
        // The vortex file is written in one shot at close time, so report the
        // in-memory size of the buffered batches. This overestimates the final
        // (compressed) file size, which makes size-based rolling conservative.
        self.buffered_size
    }
}
