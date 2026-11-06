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
use datafusion::execution::TaskContext;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
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

#[derive(Debug)]
pub(crate) struct IcebergScan {
    table: Table,
    partitions: Vec<Vec<FileScanTask>>,
    properties: Arc<PlanProperties>,
    batch_size: usize,
    metrics: ExecutionPlanMetricsSet,
    statistics: Arc<Statistics>,
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
        let mut work = Vec::new();
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
                let length = target_bytes.min(end - start);
                work.push(
                    task.clone()
                        .with_byte_range(start, length)
                        .map_err(external)?,
                );
                start += length;
            }
        }
        work.sort_unstable_by_key(|task| std::cmp::Reverse(task.length()));
        let mut groups = vec![Vec::new(); partitions];
        let mut bytes = vec![0u64; partitions];
        for task in work {
            let partition = bytes
                .iter()
                .enumerate()
                .min_by_key(|(_, size)| *size)
                .unwrap()
                .0;
            bytes[partition] += task.length();
            groups[partition].push(task);
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
        })
    }
}

impl DisplayAs for IcebergScan {
    fn fmt_as(&self, _: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "IcebergScan: partitions={}, tasks={}",
            self.partitions.len(),
            self.partitions.iter().map(Vec::len).sum::<usize>()
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
        _: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
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
        let native = self
            .table
            .reader_builder()
            .with_batch_size(self.batch_size)
            .with_data_file_concurrency_limit(1)
            .with_background_reads(true)
            .with_compute_time_observer(move |elapsed| compute.add_duration(elapsed))
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
