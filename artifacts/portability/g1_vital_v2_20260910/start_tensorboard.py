#!/usr/bin/env python3
"""Start TensorBoard for the G1 vital_v2 run on nubot."""
from __future__ import annotations

import json
import time
from pathlib import Path

import paramiko

HOST = "100.100.188.39"
USER = "nubot"
PASSWORD = " "
ROOT = "/home/nubot/phn_ws/t4_train/TienKung-Lab-g1-vital-v2-20260910"
LOGDIR = ROOT + "/logs/g1_vital_motion"
RUN = LOGDIR + "/2026-09-10_01-05-50_vital_motion_v2"
PORT = 8045
TMUX = "g1-vital-v2-tb"
OUT = Path(__file__).resolve().parent


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 60) -> tuple[int, str]:
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = (stdout.read() + stderr.read()).decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    return code, out


def main() -> None:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=25)

    code, out = run(client, "tmux ls || true")
    print("tmux ls\n", out)

    code, out = run(client, f"test -d {RUN} && ls {RUN} | head")
    if code != 0:
        raise RuntimeError(f"run dir missing:\n{out}")
    print("run dir\n", out)

    code, ports = run(
        client,
        "ss -tlnp 2>/dev/null | grep LISTEN || netstat -tln 2>/dev/null | grep LISTEN || true",
    )
    print("listen\n", ports)
    occupied = {str(PORT), "8043", "8044", "8041", "8040", "8038"}
    used = []
    for line in ports.splitlines():
        for p in occupied:
            if f":{p} " in line or f":{p}\n" in line or line.endswith(f":{p}"):
                used.append((p, line.strip()))
    print("watched ports", used)

    already = any(f":{PORT} " in line for line in ports.splitlines())
    if already:
        print(f"port {PORT} already listening; not restarting")
    else:
        run(client, f"tmux has-session -t {TMUX} && tmux kill-session -t {TMUX} || true")
        log = f"{ROOT}/artifacts/portability/g1_vital_v2_20260910/tensorboard.log"
        run(client, f"mkdir -p {ROOT}/artifacts/portability/g1_vital_v2_20260910")
        cmd = (
            f"tmux new-session -d -s {TMUX} "
            f"\"/usr/bin/python3 -m tensorboard.main --logdir {LOGDIR} "
            f"--host 0.0.0.0 --port {PORT} --load_fast=false "
            f"> {log} 2>&1\""
        )
        code, out = run(client, cmd)
        if code != 0:
            raise RuntimeError(f"tmux start failed: {out}")
        print("started", TMUX)

    url = f"http://{HOST}:{PORT}/#scalars"
    ok = False
    last = ""
    for _ in range(20):
        time.sleep(1)
        code, last = run(
            client,
            f"curl -fsS -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{PORT}/ || true",
        )
        print("http", last.strip())
        if last.strip() in {"200", "302", "307"}:
            ok = True
            break

    code, events = run(
        client,
        f"find {RUN} -name 'events.out.tfevents*' -printf '%p %s\\n' | head",
    )
    print("events\n", events)

    payload = {
        "url": url,
        "tmux": TMUX,
        "port": PORT,
        "logdir": LOGDIR,
        "run": RUN,
        "http_ok": ok,
        "http_last": last.strip(),
    }
    (OUT / "tensorboard.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    client.close()
    if not ok:
        raise SystemExit("TensorBoard did not answer HTTP")


if __name__ == "__main__":
    main()
