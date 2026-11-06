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

use std::fmt;
use std::sync::Arc;

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::common::stats::Precision;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{DataFusionError, Result, Statistics};
use datafusion::config::ConfigOptions;
use datafusion::execution::TaskContext;
use datafusion::physical_expr::{DynamicFilterTracking, EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::filter_pushdown::{
    ChildPushdownResult, FilterPushdownPhase, FilterPushdownPropagation, PushedDown,
};
use datafusion::physical_plan::metrics::{
    BaselineMetrics, ExecutionPlanMetricsSet, MetricBuilder, MetricsSet,
};
use datafusion::physical_plan::statistics::StatisticsArgs;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream,
};
use futures::{StreamExt, TryStreamExt};
use iceberg::scan::FileScanTask;
use iceberg::table::Table;

use crate::external;

mod dynamic_filter;

#[derive(Debug, Clone)]
pub(crate) struct IcebergScan {
    table: Table,
    partitions: Vec<Vec<FileScanTask>>,
    properties: Arc<PlanProperties>,
    batch_size: usize,
    metrics: ExecutionPlanMetricsSet,
    statistics: Arc<Statistics>,
    dynamic_filters: Vec<Arc<dyn PhysicalExpr>>,
}

impl IcebergScan {
    pub(crate) fn try_new(
        table: Table,
        schema: SchemaRef,
        tasks: Vec<FileScanTask>,
        partitions: usize,
        batch_size: usize,
        filtered: bool,
    ) -> Result<Self> {
        let partitions = partitions.max(2);
        let total_bytes = tasks
            .iter()
            .fold(0u64, |sum, task| sum.saturating_add(task.length()));
        let mut statistics = Statistics::new_unknown(&schema);
        statistics.num_rows = tasks.iter().fold(Precision::Exact(0usize), |sum, task| {
            let rows = task
                .record_count()
                .and_then(|rows| usize::try_from(rows).ok());
            sum.add(&rows.map_or(Precision::Absent, Precision::Exact))
        });
        if filtered
            || tasks
                .iter()
                .any(|task| task.predicate().is_some() || !task.deletes().is_empty())
        {
            statistics.num_rows = statistics.num_rows.to_inexact();
        }
        statistics.total_byte_size =
            usize::try_from(total_bytes).map_or(Precision::Absent, Precision::Inexact);
        let target_bytes = total_bytes.div_ceil(partitions as u64).max(1);
        let mut groups = vec![Vec::new(); partitions];
        let mut partition = 0;
        let mut partition_bytes = 0;
        for task in tasks {
            if task.data_file_format() != iceberg::spec::DataFileFormat::Parquet {
                return Err(DataFusionError::NotImplemented(
                    "Iceberg DataFusion supports only Parquet data files".into(),
                ));
            }
            let end = task.start().checked_add(task.length()).ok_or_else(|| {
                DataFusionError::Plan("Iceberg task byte range overflows u64".into())
            })?;
            let mut start = task.start();
            while start < end {
                if partition_bytes == target_bytes {
                    partition += 1;
                    partition_bytes = 0;
                }
                let length = (target_bytes - partition_bytes).min(end - start);
                groups
                    .get_mut(partition)
                    .ok_or_else(|| {
                        DataFusionError::Plan("Iceberg scan partition byte budget overflow".into())
                    })?
                    .push(
                        task.clone()
                            .with_byte_range(start, length)
                            .map_err(external)?,
                    );
                start += length;
                partition_bytes += length;
            }
        }
        Ok(Self {
            table,
            partitions: groups,
            properties: Arc::new(PlanProperties::new(
                EquivalenceProperties::new(schema),
                Partitioning::UnknownPartitioning(partitions),
                EmissionType::Incremental,
                Boundedness::Bounded,
            )),
            batch_size,
            metrics: ExecutionPlanMetricsSet::new(),
            statistics: Arc::new(statistics),
            dynamic_filters: Vec::new(),
        })
    }
}

impl DisplayAs for IcebergScan {
    fn fmt_as(&self, _: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "IcebergScan: partitions={}, tasks={}, dynamic_filters={}",
            self.partitions.len(),
            self.partitions.iter().map(Vec::len).sum::<usize>(),
            self.dynamic_filters.len(),
        )
    }
}

impl ExecutionPlan for IcebergScan {
    fn name(&self) -> &str {
        "IcebergScan"
    }
    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }
    fn apply_expressions(
        &self,
        visitor: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        datafusion::physical_plan::apply_expression_roots(self.dynamic_filters.iter(), visitor)
    }
    fn handle_child_pushdown_result(
        &self,
        phase: FilterPushdownPhase,
        child_pushdown_result: ChildPushdownResult,
        _: &ConfigOptions,
    ) -> Result<FilterPushdownPropagation<Arc<dyn ExecutionPlan>>> {
        let filters = child_pushdown_result.parent_filters;
        // Dynamic filters can be skipped for files whose fields cannot be adapted; retain every residual.
        let propagation = FilterPushdownPropagation::with_parent_pushdown_result(vec![
                PushedDown::No;
                filters.len()
            ]);
        if phase != FilterPushdownPhase::Post {
            return Ok(propagation);
        }
        let mut node = self.clone();
        for filter in filters {
            if DynamicFilterTracking::classify(&filter.filter).contains_dynamic_filter()
                && !node
                    .dynamic_filters
                    .iter()
                    .any(|existing| existing.eq(&filter.filter))
            {
                node.dynamic_filters.push(filter.filter);
            }
        }
        if node.dynamic_filters.len() == self.dynamic_filters.len() {
            return Ok(propagation);
        }
        let mut statistics = self.statistics.as_ref().clone();
        statistics.num_rows = statistics.num_rows.to_inexact();
        node.statistics = Arc::new(statistics);
        Ok(propagation.with_updated_node(Arc::new(node)))
    }
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(DataFusionError::Plan("IcebergScan is a leaf plan".into()));
        }
        Ok(self)
    }
    fn execute(&self, partition: usize, _: Arc<TaskContext>) -> Result<SendableRecordBatchStream> {
        let tasks = self
            .partitions
            .get(partition)
            .ok_or_else(|| {
                DataFusionError::Execution(format!("Invalid Iceberg partition {partition}"))
            })?
            .clone();
        let baseline = BaselineMetrics::new(&self.metrics, partition);
        let compute = baseline.elapsed_compute().clone();
        let mut reader = self
            .table
            .reader_builder()
            .with_batch_size(self.batch_size)
            .with_data_file_concurrency_limit(1)
            .with_background_reads(true)
            .with_view_types(true)
            .with_compute_time_observer(move |elapsed| compute.add_duration(elapsed));
        if !self.dynamic_filters.is_empty() {
            let filters = self.dynamic_filters.clone();
            let schema = self.schema();
            let pruned =
                MetricBuilder::new(&self.metrics).counter("dynamic_filter_rows_pruned", partition);
            reader = reader.with_row_filter_factory(move |task, physical_schema, metadata| {
                dynamic_filter::row_filter(
                    &filters,
                    &schema,
                    task,
                    physical_schema,
                    metadata,
                    &pruned,
                )
                .map_err(|error| {
                    iceberg::Error::new(
                        iceberg::ErrorKind::Unexpected,
                        "Cannot prepare dynamic Parquet filter",
                    )
                    .with_source(error)
                })
            });
        }
        let native = reader
            .build()
            .read(futures::stream::iter(tasks.into_iter().map(Ok)).boxed())
            .map_err(external)?;
        let io_metrics = native.metrics().clone();
        let bytes = MetricBuilder::new(&self.metrics).counter("bytes_scanned", partition);
        let mut bytes_recorded = 0;
        let mut input = native.stream().map_err(external);
        let stream = futures::stream::poll_fn(move |cx| {
            let timer = baseline.elapsed_compute().timer();
            let poll = input.poll_next_unpin(cx);
            drop(timer);
            let bytes_read = io_metrics.bytes_read();
            bytes.add((bytes_read - bytes_recorded) as usize);
            bytes_recorded = bytes_read;
            baseline.record_poll(poll)
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream,
        )))
    }
    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
    fn statistics_from_inputs(
        &self,
        _: &[Arc<Statistics>],
        args: &StatisticsArgs,
    ) -> Result<Arc<Statistics>> {
        match args.partition() {
            None => Ok(self.statistics.clone()),
            Some(partition) => {
                let tasks = self.partitions.get(partition).ok_or_else(|| {
                    DataFusionError::Plan(format!("Invalid Iceberg partition {partition}"))
                })?;
                let mut statistics = Statistics::new_unknown(&self.schema());
                let total_bytes = tasks.iter().try_fold(0usize, |sum, task| {
                    sum.checked_add(usize::try_from(task.length()).ok()?)
                });
                statistics.total_byte_size =
                    total_bytes.map_or(Precision::Absent, Precision::Inexact);
                if tasks.is_empty() {
                    statistics.num_rows = Precision::Exact(0);
                }
                Ok(Arc::new(statistics))
            }
        }
    }
}
