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

//! Parallel Parquet scans of Iceberg snapshots in DataFusion.

mod predicate;
mod scan;

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::stats::Precision;
use datafusion::common::{DataFusionError, Result, Statistics};
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown, TableType};
use datafusion::physical_plan::ExecutionPlan;
use iceberg::arrow::schema_to_arrow_schema;
use iceberg::table::Table;

/// Exposes an immutable Iceberg table snapshot as parallel Parquet scans.
///
/// Scans expose at least two partitions, using the session's target when larger.
#[derive(Debug)]
pub struct IcebergTableProvider {
    table: Table,
    schema: SchemaRef,
}

impl IcebergTableProvider {
    /// Creates a provider for the current schema and snapshot of `table`.
    pub fn try_new(table: Table) -> Result<Self> {
        let schema =
            Arc::new(schema_to_arrow_schema(table.metadata().current_schema()).map_err(external)?);
        Ok(Self { table, schema })
    }
}

fn external(error: iceberg::Error) -> DataFusionError {
    DataFusionError::External(Box::new(error))
}

#[async_trait]
impl TableProvider for IcebergTableProvider {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let schema = match projection {
            Some(indices) => Arc::new(self.schema.project(indices)?),
            None => self.schema.clone(),
        };
        let mut builder = self
            .table
            .scan()
            .select(schema.fields().iter().map(|f| f.name()));
        if let Some(predicate) = filters
            .iter()
            .filter_map(|expr| predicate::predicate(expr, self.table.metadata().current_schema()))
            .reduce(|a, b| a.and(b))
        {
            builder = builder.with_filter(predicate);
        }
        let scan = builder.build().map_err(external)?;
        let tasks = futures::TryStreamExt::try_collect::<Vec<_>>(
            scan.plan_files().await.map_err(external)?,
        )
        .await
        .map_err(external)?;
        Ok(Arc::new(scan::IcebergScan::try_new(
            self.table.clone(),
            schema,
            tasks,
            state.config().target_partitions(),
            state.config().batch_size(),
            !filters.is_empty(),
        )?))
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        Ok(filters
            .iter()
            .map(|expr| {
                if predicate::predicate(expr, self.table.metadata().current_schema()).is_some() {
                    TableProviderFilterPushDown::Inexact
                } else {
                    TableProviderFilterPushDown::Unsupported
                }
            })
            .collect())
    }

    fn statistics(&self) -> Option<Statistics> {
        let snapshot = self.table.metadata().current_snapshot()?;
        let summary = &snapshot.summary().additional_properties;
        let mut statistics = Statistics::new_unknown(&self.schema);
        statistics.num_rows = summary
            .get("total-records")
            .and_then(|value| value.parse().ok())
            .map_or(Precision::Absent, Precision::Inexact);
        statistics.total_byte_size = summary
            .get("total-files-size")
            .and_then(|value| value.parse().ok())
            .map_or(Precision::Absent, Precision::Inexact);
        Some(statistics)
    }
}
