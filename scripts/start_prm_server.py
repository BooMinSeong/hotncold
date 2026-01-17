#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Standalone script to start vLLM PRM server.
Can be called directly or via Slurm.

Usage:
    python scripts/start_prm_server.py --port 8001
    python scripts/start_prm_server.py --model Qwen/Qwen2.5-Math-PRM-7B --port 8001
"""

import argparse
import atexit
import json
import os
import signal
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_free_port() -> int:
    """Find an available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def write_service_file(
    host: str, port: int, model: str, service_dir: Path
) -> Path:
    """Write service discovery file."""
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    service_file = service_dir / f"{job_id}_prm_service.json"

    service_info = {
        "host": host,
        "port": port,
        "model": model,
        "started_at": datetime.now().isoformat(),
        "job_id": job_id,
        "gpu_count": 1,
    }

    with open(service_file, "w") as f:
        json.dump(service_info, f, indent=2)

    print(f"Service file written: {service_file}")
    return service_file


def cleanup_service_file(service_file: Path):
    """Remove service file on exit."""
    if service_file.exists():
        service_file.unlink()
        print(f"Cleaned up service file: {service_file}")


def main():
    parser = argparse.ArgumentParser(description="Start vLLM PRM server")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Math-PRM-7B",
        help="PRM model to serve",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to serve on (default: auto-select)",
    )
    parser.add_argument(
        "--service-dir",
        default="./services",
        help="Directory for service discovery files",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for model",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model weights",
    )
    args = parser.parse_args()

    port = args.port or get_free_port()
    host = socket.getfqdn()
    service_dir = Path(args.service_dir)
    service_dir.mkdir(exist_ok=True, parents=True)

    service_file = write_service_file(host, port, args.model, service_dir)

    # Register cleanup function
    atexit.register(cleanup_service_file, service_file)

    # Handle signals for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        cleanup_service_file(service_file)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Build vLLM serve command
    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--port",
        str(port),
        "--trust-remote-code",
        "--dtype",
        args.dtype,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
    ]

    print(f"Starting vLLM PRM server on {host}:{port}")
    print(f"Model: {args.model}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"vLLM server exited with error: {e}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cleanup_service_file(service_file)


if __name__ == "__main__":
    main()
