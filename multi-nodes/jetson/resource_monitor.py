"""Edge resource monitoring for Jetson devices."""

from __future__ import annotations

import re
import subprocess
import time
from typing import Dict, Optional

import psutil


class ResourceMonitor:
    """Collect CPU, GPU, RAM, power, and temperature from a Jetson node."""

    GPU_RE = re.compile(r"(?:GR3D_FREQ|GPU)\s+(\d+(?:\.\d+)?)%")
    RAM_RE = re.compile(r"RAM\s+(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)MB")
    POWER_RE = re.compile(r"(?:VDD_IN|POM_5V_IN)\s+(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)")
    TEMP_RE = re.compile(r"(?:GPU|CPU|SOC|Tboard|thermal)[^@\s]*@([\d.]+)C", re.IGNORECASE)

    def __init__(self) -> None:
        self.use_tegrastats = self._check_tegrastats()
        self.last_measurement: Optional[Dict[str, float]] = None

    @staticmethod
    def _check_tegrastats() -> bool:
        try:
            result = subprocess.run(["which", "tegrastats"], capture_output=True, timeout=1)
            return result.returncode == 0
        except Exception:
            return False

    def _read_tegrastats_once(self) -> str:
        if not self.use_tegrastats:
            return ""
        try:
            result = subprocess.run(
                ["tegrastats", "--interval", "100"],
                capture_output=True,
                timeout=1.0,
                text=True,
                check=False,
            )
            lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            return lines[-1] if lines else ""
        except Exception:
            return ""

    @staticmethod
    def _nvidia_gpu_percent() -> Optional[float]:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                timeout=2,
                text=True,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip().splitlines()[0])
        except Exception:
            pass
        return None

    def measure_all(self) -> Dict[str, float]:
        line = self._read_tegrastats_once()

        cpu_percent = float(psutil.cpu_percent(interval=0.1))
        memory = psutil.virtual_memory()
        ram_mb = float(memory.used / (1024 * 1024))

        gpu_percent = 0.0
        power_w = 0.0
        temperature_c = 0.0

        gpu_match = self.GPU_RE.search(line)
        if gpu_match:
            gpu_percent = float(gpu_match.group(1))
        else:
            nvidia_gpu = self._nvidia_gpu_percent()
            if nvidia_gpu is not None:
                gpu_percent = nvidia_gpu

        ram_match = self.RAM_RE.search(line)
        if ram_match:
            ram_mb = float(ram_match.group(1))

        power_match = self.POWER_RE.search(line)
        if power_match:
            # Jetson tegrastats reports the first value in mW.
            power_w = float(power_match.group(1)) / 1000.0

        temperatures = [float(value) for value in self.TEMP_RE.findall(line)]
        if temperatures:
            temperature_c = max(temperatures)
        else:
            try:
                sensor_values = psutil.sensors_temperatures()
                readings = [entry.current for entries in sensor_values.values() for entry in entries if entry.current is not None]
                if readings:
                    temperature_c = float(max(readings))
            except Exception:
                pass

        measurement = {
            "gpu_percent": gpu_percent,
            "ram_mb": ram_mb,
            "power_w": power_w,
            "cpu_percent": cpu_percent,
            "temperature_c": temperature_c,
            "timestamp": time.time(),
        }
        self.last_measurement = measurement
        return measurement
