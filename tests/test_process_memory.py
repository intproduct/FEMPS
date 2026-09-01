import torch

from femps.benchmarks import ProcessRSSMonitor, current_process_rss_bytes


def test_process_rss_monitor_records_native_tensor_allocation() -> None:
    baseline = current_process_rss_bytes()
    with ProcessRSSMonitor(sampling_interval_seconds=0.001) as monitor:
        tensor = torch.ones(1_000_000, dtype=torch.float64)
        assert float(tensor.sum()) == 1_000_000.0
    record = monitor.record()
    assert record.samples >= 2
    assert record.baseline_rss_bytes > 0
    assert record.peak_rss_bytes >= record.baseline_rss_bytes
    assert record.peak_rss_bytes >= record.final_rss_bytes
    assert baseline > 0
