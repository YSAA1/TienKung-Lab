#!/usr/bin/env python3
"""Probe G1 vital_v2 and Z2 reset training health; do not kill anything."""
from __future__ import annotations

import json
from pathlib import Path

import paramiko

HOST = "100.100.188.39"
OUT = Path(__file__).resolve().parent / "status_probe.json"

G1_RUN = (
    "/home/nubot/phn_ws/t4_train/TienKung-Lab-g1-vital-v2-20260910/"
    "logs/g1_vital_motion/2026-09-10_01-05-50_vital_motion_v2"
)
Z2_ROOT = "/home/nubot/phn_ws/t4_train/TienKung-Lab-z2-reset-20260909"


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 60) -> tuple[int, str]:
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = (stdout.read() + stderr.read()).decode("utf-8", "replace")
    return stdout.channel.recv_exit_status(), out


def main() -> None:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username="nubot", password=" ", timeout=25)

    cmds = {
        "tmux": "tmux ls",
        "gpu": "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv",
        "g1_ckpt": (
            f"ls -lt {G1_RUN}/model_*.pt | head -n 8; echo ---; "
            f"stat -c '%y %s %n' {G1_RUN}/events.out.tfevents* 2>/dev/null | head; "
            f"ls {G1_RUN}/model_*.pt | sed -n 's/.*model_\\([0-9]*\\)\\.pt/\\1/p' | sort -n | tail -n 3"
        ),
        "g1_tmux": "tmux capture-pane -pt g1-vital-v2 -S -40 || true",
        "z2_logs": f"ls -d {Z2_ROOT}/logs/*/* 2>/dev/null | tail -n 20; echo ---; find {Z2_ROOT}/logs -maxdepth 3 -type d | head -n 40",
        "ports": "ss -tlnp | grep -E ':(8045|8050|8043)\\s' || true",
        "tb8050": "curl -fsS http://127.0.0.1:8050/data/runs || echo FAIL8050",
        "tb8045": "curl -fsS http://127.0.0.1:8045/data/runs || echo FAIL8045",
    }
    results = {}
    for name, cmd in cmds.items():
        code, out = run(client, cmd, timeout=90)
        results[name] = {"code": code, "out": out}
        print(f"===== {name} exit {code} =====")
        print(out[:4000])

    code, z2_latest = run(
        client,
        f"find {Z2_ROOT}/logs -name 'model_*.pt' -printf '%T+ %p\\n' 2>/dev/null | sort | tail -n 15",
        timeout=90,
    )
    results["z2_ckpt"] = {"code": code, "out": z2_latest}
    print("===== z2_ckpt =====")
    print(z2_latest)

    code, z2_events = run(
        client,
        f"find {Z2_ROOT}/logs -name 'events.out.tfevents*' -printf '%T+ %s %p\\n' 2>/dev/null | sort | tail -n 10",
        timeout=90,
    )
    results["z2_events"] = {"code": code, "out": z2_events}
    print("===== z2_events =====")
    print(z2_events)

    code, z2_tmux = run(client, "tmux capture-pane -pt z2-reset-formal-20260909 -S -40 || true")
    results["z2_tmux"] = {"code": code, "out": z2_tmux}
    print("===== z2_tmux =====")
    print(z2_tmux[-3000:])

    code, procs = run(
        client,
        "ps -eo pid,lstart,etime,cmd | grep -E 'train.py|train_g1_vital|z2|g1_vital' | grep -v grep | head -n 40",
    )
    results["procs"] = {"code": code, "out": procs}
    print("===== procs =====")
    print(procs)

    OUT.write_text(json.dumps({k: v["out"][-8000:] for k, v in results.items()}, indent=2))
    client.close()


if __name__ == "__main__":
    main()
