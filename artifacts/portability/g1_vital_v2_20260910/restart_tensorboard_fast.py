#!/usr/bin/env python3
from __future__ import annotations

import json
import shlex
import time
from pathlib import Path

import paramiko

HOST = "100.100.188.39"
OUT = Path(__file__).resolve().parent / "tensorboard_fast.json"
JOBS = [
    {
        "name": "z2",
        "tmux": "z2-reset-tb-20260909",
        "port": 8050,
        "cwd": "/home/nubot/phn_ws/t4_train/TienKung-Lab-z2-reset-20260909",
        "logdir": "/home/nubot/phn_ws/t4_train/TienKung-Lab-z2-reset-20260909/logs/z2_loco_teacher_sparse/2026-09-09_10-33-53_z2_reset_aligned_v1",
        "logfile": "/home/nubot/phn_ws/t4_train/TienKung-Lab-z2-reset-20260909/artifacts/z2_migration/formal_v1/tensorboard_fast.log",
    },
    {
        "name": "g1_v2",
        "tmux": "g1-vital-v2-tb",
        "port": 8045,
        "cwd": "/home/nubot/phn_ws/t4_train/TienKung-Lab-g1-vital-v2-20260910",
        "logdir": "/home/nubot/phn_ws/t4_train/TienKung-Lab-g1-vital-v2-20260910/logs/g1_vital_motion",
        "logfile": "/home/nubot/phn_ws/t4_train/TienKung-Lab-g1-vital-v2-20260910/artifacts/portability/g1_vital_v2_20260910/tensorboard.log",
    },
]


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 30) -> tuple[int, str]:
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = (stdout.read() + stderr.read()).decode("utf-8", "replace")
    return stdout.channel.recv_exit_status(), out


def main() -> None:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print("connect", flush=True)
    client.connect(HOST, username="nubot", password=" ", timeout=20)
    print("connected", flush=True)

    for job in JOBS:
        print("stop", job["name"], flush=True)
        run(client, f"tmux kill-session -t {job['tmux']} || true")
        run(
            client,
            f"pkill -f 'tensorboard.main .*--port {job['port']}' || true",
        )
    time.sleep(2)
    _, leftover = run(client, "ss -tlnp | grep -E ':(8045|8050)\\s' || echo none")
    print("listen after kill\n", leftover, flush=True)

    for job in JOBS:
        run(client, f"mkdir -p $(dirname {job['logfile']})")
        inner = (
            f"cd {job['cwd']} && /usr/bin/python3 -m tensorboard.main "
            f"--logdir {job['logdir']} --host 0.0.0.0 --port {job['port']} "
            f"--load_fast=true --reload_interval 120 "
            f"--samples_per_plugin scalars=1000,images=0,histograms=0 "
            f"> {job['logfile']} 2>&1"
        )
        code, out = run(client, f"tmux new-session -d -s {job['tmux']} {shlex.quote(inner)}")
        print("started", job["name"], code, out, flush=True)
        if code != 0:
            raise RuntimeError(out)

    payload = []
    for job in JOBS:
        listening = False
        last = ""
        for i in range(20):
            time.sleep(1)
            _, last = run(client, f"ss -tlnp | grep ':{job['port']} ' || true")
            print(job["name"], "try", i, repr(last[:120]), flush=True)
            if f":{job['port']}" in last:
                listening = True
                break
        _, logtail = run(client, f"tail -n 15 {job['logfile']} || true")
        payload.append(
            {
                "name": job["name"],
                "url": f"http://{HOST}:{job['port']}/#timeseries",
                "listening": listening,
                "listen": last.strip()[:300],
                "log_tail": logtail[-1200:],
            }
        )

    _, gpu = run(
        client,
        "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader",
    )
    OUT.write_text(json.dumps({"jobs": payload, "gpu": gpu}, indent=2) + "\n")
    print(json.dumps({"jobs": payload, "gpu": gpu}, indent=2), flush=True)
    client.close()


if __name__ == "__main__":
    main()
