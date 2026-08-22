"""Cloud Auto-Scaling Coordinator for IndicLLM-Bharat.

Monitors queue length, GPU utilization, memory saturation, and concurrent requests.
Dynamically scales compute tier across Local -> GPU Server -> AWS Bedrock Cluster.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ComputeScaleMode(str, Enum):
    LOCAL_ONLY = "LOCAL_ONLY"
    LOCAL_GPU_HYBRID = "LOCAL_GPU_HYBRID"
    AWS_CLOUD_CLUSTER = "AWS_CLOUD_CLUSTER"


@dataclass
class SystemMetricsSnapshot:
    queue_length: int
    active_requests: int
    local_gpu_utilization_pct: float
    local_ram_utilization_pct: float
    avg_latency_ms: float
    recommended_mode: ComputeScaleMode


class CloudAutoScaler:
    """Evaluates telemetry and dynamically reallocates compute resources."""

    def __init__(
        self,
        gpu_scale_up_threshold: float = 85.0,
        latency_scale_up_threshold_ms: float = 80.0,
    ) -> None:
        self.gpu_scale_up_threshold = gpu_scale_up_threshold
        self.latency_scale_up_threshold_ms = latency_scale_up_threshold_ms
        self.current_mode = ComputeScaleMode.LOCAL_ONLY

    def evaluate_scale(
        self,
        queue_length: int,
        active_requests: int,
        gpu_util_pct: float,
        ram_util_pct: float,
        avg_latency_ms: float,
    ) -> SystemMetricsSnapshot:
        """Decide if cloud scale-out or local scale-in is warranted."""
        if (
            gpu_util_pct > self.gpu_scale_up_threshold
            or avg_latency_ms > self.latency_scale_up_threshold_ms
            or queue_length > 10
        ):
            target_mode = ComputeScaleMode.AWS_CLOUD_CLUSTER
        elif active_requests > 3 or gpu_util_pct > 60.0:
            target_mode = ComputeScaleMode.LOCAL_GPU_HYBRID
        else:
            target_mode = ComputeScaleMode.LOCAL_ONLY

        self.current_mode = target_mode

        return SystemMetricsSnapshot(
            queue_length=queue_length,
            active_requests=active_requests,
            local_gpu_utilization_pct=gpu_util_pct,
            local_ram_utilization_pct=ram_util_pct,
            avg_latency_ms=avg_latency_ms,
            recommended_mode=target_mode,
        )
