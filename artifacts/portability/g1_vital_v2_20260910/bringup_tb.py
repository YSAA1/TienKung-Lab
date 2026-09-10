#!/usr/bin/env python3
from __future__ import annotations

import shlex
import time

import paramiko

HOST = "100.100.188.39"


def run(client, cmd, timeout=25):
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = (stdout.read() + stderr.read()).decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    print(cmd[:100], "->", code, flush=True)
    print(out[:1500], flush=True)
    return code, out


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username="nubot", password=" ", timeout=20)
    print("connected", flush=True)
    run(
        client,
        "ps -eo pid,etime,cmd | grep 'g1_motion_experiment vital_v2' | grep -v grep | head",
    )

    jobs = [
        {
            "tmux": "z2-reset-tb-20260909",
            "port": 8050,
            "inner": (
                "cd /home/nubot/phn_ws/t4_train/TienKung-Lab-z2-reset-20260909 && "
                "mkdir -p artifacts/z2_migration/formal_v1 && "
                "bash scripts/nubot_run.sh -m tensorboard.main "
                "--logdir logs/z2_loco_teacher_sparse/2026-09-09_10-33-53_z2_reset_aligned_v1 "
                "--host 0.0.0.0 --port 8050 --samples_per_plugin scalars=1000,images=0,histograms=0 "
                "> artifacts/z2_migration/formal_v1/tensorboard.log 2>&1"
            ),
        },
        {
            "tmux": "g1-vital-v2-tb",
            "port": 8045,
            "inner": (
                "cd /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-vital-v2-20260910 && "
                "mkdir -p artifacts/portability/g1_vital_v2_20260910 && "
                "bash scripts/nubot_run.sh -m tensorboard.main "
                "--logdir logs/g1_vital_motion/2026-09-10_01-05-50_vital_motion_v2 "
                "--host 0.0.0.0 --port 8045 --samples_per_plugin scalars=1000,images=0,histograms=0 "
                "> artifacts/portability/g1_vital_v2_20260910/tensorboard.log 2>&1"
            ),
        },
    ]
    for job in jobs:
        run(client, f"tmux kill-session -t {job['tmux']} || true")
        run(client, f"tmux new-session -d -s {job['tmux']} {shlex.quote(job['inner'])}")

    time.sleep(8)
    run(client, "ss -tlnp | grep -E ':(8045|8050)\\s' || echo still-starting")
    run(client, "tmux ls | grep -E 'g1-vital-v2|z2-reset' || true")
    run(
        client,
        "tail -n 8 /home/nubot/phn_ws/t4_train/TienKung-Lab-z2-reset-20260909/artifacts/z2_migration/formal_v1/tensorboard.log; "
        "echo ====; "
        "tail -n 8 /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-vital-v2-20260910/artifacts/portability/g1_vital_v2_20260910/tensorboard.log",
    )
    client.close()


if __name__ == "__main__":
    main()
