"""HealthMonitor — background health prober for Qdrant and GPU.

Caches status and refreshes on a configurable interval.
QueryEngine checks get_status() before making calls.

Note: No Ollama probing — models are loaded at server init time,
so model_loaded=True means the server started successfully.
"""
import asyncio
import subprocess

import structlog
from dataclasses import dataclass
from qdrant_client import QdrantClient

from knowledge_hub.config import Settings

logger = structlog.get_logger()


@dataclass
class HealthStatus:
    """Health status of system components."""

    model_loaded: bool = False  # FlagEmbedding models loaded (server started)
    qdrant: bool = False  # Qdrant vector store reachable
    gpu_available: bool = False  # NVIDIA GPU available
    gpu_memory_free_mb: int = 0  # Free GPU memory in MB


class HealthMonitor:
    """Background health prober for Qdrant and GPU.

    No Ollama probing — FlagEmbedding models are loaded at server init time.
    If the server started without crashing, models are loaded.
    """

    def __init__(self, settings: Settings, qdrant_client: QdrantClient):
        """Initialize the health monitor.

        Args:
            settings: Application settings.
            qdrant_client: Qdrant client instance to probe.
        """
        self.settings = settings
        self._qdrant = qdrant_client
        self._cached_status: HealthStatus | None = None
        self._task: asyncio.Task | None = None

    async def start(self, interval_seconds: int = 30):
        """Start the background probe loop.

        Args:
            interval_seconds: Interval between probes (default 30s).
        """
        self._cached_status = await self._probe_all()
        self._task = asyncio.create_task(self._probe_loop(interval_seconds))

    async def get_status(self) -> HealthStatus:
        """Get the current cached health status.

        If not yet cached, probes once and caches the result.

        Returns:
            Current HealthStatus.
        """
        if self._cached_status is None:
            self._cached_status = await self._probe_all()
        return self._cached_status

    async def _probe_loop(self, interval: int):
        """Background loop that periodically probes all components.

        Args:
            interval: Seconds between probes.
        """
        while True:
            try:
                self._cached_status = await self._probe_all()
            except Exception as e:
                logger.error("health_probe_failed", error=str(e))
            await asyncio.sleep(interval)

    async def _probe_all(self) -> HealthStatus:
        """Probe all components and return status.

        Returns:
            HealthStatus with all component states.
        """
        status = HealthStatus()
        # model_loaded = True (server started → models loaded)
        status.model_loaded = True
        status.qdrant = await self._probe_qdrant()
        status.gpu_available, status.gpu_memory_free_mb = await self._probe_gpu()
        return status

    async def _probe_qdrant(self) -> bool:
        """Probe Qdrant to check if it's reachable.

        Returns:
            True if Qdrant responds, False otherwise.
        """
        try:
            collections = self._qdrant.get_collections()
            return collections is not None
        except Exception:
            return False

    async def _probe_gpu(self) -> tuple[bool, int]:
        """Probe GPU availability and free memory via nvidia-smi.

        Returns:
            Tuple of (gpu_available, gpu_memory_free_mb).
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                mem_free = int(result.stdout.strip().split("\n")[0])
                return True, mem_free
        except Exception:
            pass
        return False, 0
