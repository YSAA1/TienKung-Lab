#!/usr/bin/env python3
"""Copy the VITAL frozen tree, overlay vital_v2 files, start 30k on GPU1/3."""
from __future__ import annotations

from pathlib import Path

import paramiko

HOST = "100.100.188.39"
USER = "nubot"
PASSWORD = " "
SRC = "/home/nubot/phn_ws/t4_train/TienKung-Lab-g1-vital-motion-20260908"
DST = "/home/nubot/phn_ws/t4_train/TienKung-Lab-g1-vital-v2-20260910"
LOCAL = Path(r"d:\TienKung-Lab")
OVERLAY = [
    "legged_lab/envs/g1/motion_experiment.py",
    "legged_lab/scripts/train.py",
    "scripts/train_g1_vital_v2.sh",
    "legged_lab/envs/g1/datasets/motion_amp_expert_unitree_v5/_manifest.json",
]


def run(client, cmd: str, timeout: int = 120) -> str:
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = (stdout.read() + stderr.read()).decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    if code != 0:
        raise RuntimeError(f"exit {code}: {cmd}\n{out}")
    return out


def main() -> None:
    print("connecting", flush=True)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=25)
    print(run(client, "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv"), flush=True)
    print(run(client, "tmux ls || true"), flush=True)
    gpu = run(client, "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits")
    print("gpu mem\n", gpu, flush=True)
    for line in gpu.strip().splitlines():
        idx, mem = [p.strip() for p in line.split(",")]
        if idx in {"1", "3"} and int(mem) > 500:
            raise RuntimeError(f"GPU{idx} busy ({mem} MiB); refusing to launch")
        if idx in {"0", "2"} and int(mem) < 100:
            print(f"note GPU{idx} idle, still not using it", flush=True)

    print("sync tree", flush=True)
    run(
        client,
        "test -d "
        + SRC
        + " && mkdir -p "
        + DST
        + " && rsync -a --delete --exclude logs --exclude artifacts "
        + SRC.rstrip("/")
        + "/ "
        + DST.rstrip("/")
        + "/",
        timeout=600,
    )
    amp = run(
        client,
        "ls "
        + DST
        + "/legged_lab/envs/g1/datasets/motion_amp_expert_unitree_v5/walk1_subject1.txt "
        + DST
        + "/legged_lab/envs/g1/datasets/motion_amp_expert_unitree_v5/_manifest.json",
    )
    print(amp, flush=True)

    sftp = client.open_sftp()
    for rel in OVERLAY:
        local = LOCAL / rel
        remote = f"{DST}/{rel}"
        print("put", rel, flush=True)
        sftp.put(str(local), remote)
    sftp.close()
    run(client, "chmod +x " + DST + "/scripts/train_g1_vital_v2.sh")
    run(client, "tmux has-session -t g1-vital-v2 && tmux kill-session -t g1-vital-v2 || true")
    run(
        client,
        "tmux new-session -d -s g1-vital-v2 "
        + f"'cd {DST} && bash scripts/train_g1_vital_v2.sh'",
    )
    print(run(client, "sleep 8; tmux has-session -t g1-vital-v2 && echo TMUX_OK"), flush=True)
    print(run(client, "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv"), flush=True)
    log = DST + "/artifacts/portability/g1_vital_v2_20260910/train.log"
    print(run(client, f"mkdir -p {DST}/artifacts/portability/g1_vital_v2_20260910; tail -n 40 {log} || true"), flush=True)
    client.close()
    print("launch requested", flush=True)


if __name__ == "__main__":
    main()
