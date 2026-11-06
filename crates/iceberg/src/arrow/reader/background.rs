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

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;

use arrow_array::RecordBatch;
use futures::future::{AbortHandle, Abortable, Aborted, poll_fn};
use futures::{Stream, StreamExt};
use tokio::sync::mpsc;

use crate::arrow::ScanMetrics;
use crate::runtime::{JoinHandle, Runtime};
use crate::scan::ArrowRecordBatchStream;
use crate::{Error, ErrorKind, Result};

pub(super) fn read_in_background(
    mut input: ArrowRecordBatchStream,
    runtime: &Runtime,
    metrics: ScanMetrics,
) -> ArrowRecordBatchStream {
    let (sender, receiver) = mpsc::channel(2);
    let (abort, registration) = AbortHandle::new_pair();
    let io_runtime = runtime.io().clone();
    let producer = async move {
        loop {
            let batch = poll_fn(|cx| {
                let start = Instant::now();
                let poll = {
                    // Bind I/O resources to their reactor while decoding on CPU workers.
                    let _entered = io_runtime.enter();
                    input.poll_next_unpin(cx)
                };
                metrics.record_elapsed_compute(start.elapsed());
                poll
            })
            .await;
            let Some(batch) = batch else { break };
            let failed = batch.is_err();
            if sender.send(batch).await.is_err() || failed {
                break;
            }
        }
    };
    let worker = runtime.cpu().spawn(Abortable::new(producer, registration));
    Box::pin(BackgroundRead {
        receiver,
        worker,
        abort,
        finished: false,
    })
}

struct BackgroundRead {
    receiver: mpsc::Receiver<Result<RecordBatch>>,
    worker: JoinHandle<std::result::Result<(), Aborted>>,
    abort: AbortHandle,
    finished: bool,
}

impl Stream for BackgroundRead {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.finished {
            return Poll::Ready(None);
        }
        match this.receiver.poll_recv(cx) {
            Poll::Ready(None) => match Pin::new(&mut this.worker).poll(cx) {
                Poll::Ready(result) => {
                    this.finished = true;
                    Poll::Ready(match result {
                        Ok(Ok(())) => None,
                        Ok(Err(_)) => Some(Err(Error::new(
                            ErrorKind::Unexpected,
                            "Background read task was aborted",
                        ))),
                        Err(error) => Some(Err(error)),
                    })
                }
                Poll::Pending => Poll::Pending,
            },
            poll => poll,
        }
    }
}

impl Drop for BackgroundRead {
    fn drop(&mut self) {
        self.abort.abort();
    }
}

#[cfg(test)]
mod tests {
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::Poll;
    use std::time::Duration;

    use arrow_array::RecordBatch;
    use arrow_schema::Schema;
    use futures::{StreamExt, TryStreamExt, stream};
    use tokio::runtime::Builder;
    use tokio::sync::oneshot;

    use super::read_in_background;
    use crate::Runtime;
    use crate::arrow::ScanMetrics;
    use crate::scan::ArrowRecordBatchStream;

    struct DropSignal(Option<oneshot::Sender<()>>);

    impl Drop for DropSignal {
        fn drop(&mut self) {
            if let Some(sender) = self.0.take() {
                let _ = sender.send(());
            }
        }
    }

    fn batch() -> RecordBatch {
        RecordBatch::new_empty(Arc::new(Schema::empty()))
    }

    #[tokio::test]
    async fn background_reads_bound_prefetch_and_abort_on_drop() {
        let produced = Arc::new(AtomicUsize::new(0));
        let count = produced.clone();
        let (prefetched, prefetch_done) = oneshot::channel();
        let mut prefetched = Some(prefetched);
        let (dropped, drop_done) = oneshot::channel();
        let drop_signal = DropSignal(Some(dropped));
        let input = stream::poll_fn(move |_| {
            let _keep_alive = &drop_signal;
            if count.fetch_add(1, Ordering::Relaxed) == 2 {
                let _ = prefetched.take().unwrap().send(());
            }
            Poll::Ready(Some(Ok(batch())))
        });
        let metrics = ScanMetrics::new();
        let output = read_in_background(Box::pin(input), &Runtime::current(), metrics.clone());
        prefetch_done.await.unwrap();
        assert_eq!(produced.load(Ordering::Relaxed), 3);
        assert!(metrics.elapsed_compute_nanos() > 0);
        drop(output);
        tokio::time::timeout(Duration::from_secs(1), drop_done)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn background_reads_surface_worker_panics() {
        let input: ArrowRecordBatchStream = Box::pin(stream::poll_fn(|_| {
            panic!("background read panic");
        }));
        let mut output = read_in_background(input, &Runtime::current(), ScanMetrics::new());
        let error = output.next().await.unwrap().unwrap_err();
        assert!(error.to_string().contains("background read panic"));
        assert!(output.next().await.is_none());
    }

    #[test]
    fn background_reads_use_io_driver_from_compute_workers() {
        let io = Builder::new_current_thread().enable_all().build().unwrap();
        let cpu = Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("background-compute")
            .build()
            .unwrap();
        let input = stream::once(async {
            assert_eq!(std::thread::current().name(), Some("background-compute"));
            tokio::time::sleep(Duration::from_millis(1)).await;
            Ok(batch())
        });
        let metrics = ScanMetrics::new();
        let output = read_in_background(
            Box::pin(input),
            &Runtime::new_with_split(&io, &cpu),
            metrics.clone(),
        );
        let batches: Vec<_> = io.block_on(output.try_collect()).unwrap();
        assert_eq!(batches, vec![batch()]);
        assert!(metrics.elapsed_compute_nanos() > 0);

        let input = Box::pin(stream::pending());
        let mut output = read_in_background(
            input,
            &Runtime::new_with_split(&io, &cpu),
            ScanMetrics::new(),
        );
        drop(cpu);
        let error = io
            .block_on(Pin::new(&mut output).next())
            .unwrap()
            .unwrap_err();
        assert!(error.to_string().contains("cancelled"));
    }
}
