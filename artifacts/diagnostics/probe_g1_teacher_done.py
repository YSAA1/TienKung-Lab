#!/usr/bin/env python3
"""Snapshot G1 teacher runs on nubot."""
from __future__ import annotations

import paramiko

HOST, USER, PASSWORD = "100.100.188.39", "nubot", " "

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASSWORD, timeout=25)


def run(cmd: str, timeout: int = 90) -> str:
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    return (stdout.read() + stderr.read()).decode("utf-8", "replace")


cmds = [
    ("TMUX", "tmux ls 2>/dev/null || true"),
    (
        "GPU",
        "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv",
    ),
    (
        "APPS",
        "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv",
    ),
    (
        "VITAL_MODELS",
        "ls -lh /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-vital-motion-20260908/logs/g1_vital_motion/2026-09-08_16-25-01_vital_motion_v1/model_*.pt 2>/dev/null | tail -25",
    ),
    (
        "VITAL_DIRS",
        "ls -td /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-vital-motion-20260908/logs/g1_vital_motion/*/ 2>/dev/null | head -8",
    ),
    (
        "VITAL_TAIL",
        "tail -n 40 /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-vital-motion-20260908/artifacts/portability/g1_vital_motion_20260908/train.log 2>/dev/null || tail -n 40 /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-vital-motion-20260908/logs/g1_vital_motion/2026-09-08_16-25-01_vital_motion_v1/*.txt 2>/dev/null | tail -40",
    ),
    (
        "CORE_DIRS",
        "ls -td /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-core-fixes-20260908/logs/*/*/ 2>/dev/null | head -8",
    ),
    (
        "CORE_MODELS",
        "ls -lh /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-core-fixes-20260908/logs/g1_core_fixes_B30000/*/model_*.pt 2>/dev/null | tail -15",
    ),
    (
        "PROCS",
        "ps -eo pid,etime,cmd | awk '/g1_loco_teacher|vital_motion|g1_core_fixes|z2_loco|train.py/ && !/awk/'",
    ),
]

for title, cmd in cmds:
    print(f"=== {title} ===")
    print(run(cmd))
    print()
