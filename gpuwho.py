#!/usr/bin/env python3
"""gpuwho: see what is on your NVIDIA GPU, with real names instead of a wall of "python".

A process is named by its program; an interpreter (python, node, java) is named by the
systemd service it runs under, else by its -m module or script. Read-only: it only runs
nvidia-smi and reads /proc. Single file, standard library only.
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

GENERIC = {"pt_main_thread", "node", "java"}  # plus anything starting with "python"
GPU_FIELDS = (
    "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,"
    "power.draw,power.limit,fan.speed,clocks.sm"
)
RESET, DIM, BOLD, CYAN = "\033[0m", "\033[2m", "\033[1m", "\033[36m"


def smi(*args):
    return subprocess.run(
        ["nvidia-smi", *args], capture_output=True, text=True, timeout=10, check=False
    ).stdout


def parse_gpus(text):
    keys = GPU_FIELDS.split(",")
    return [
        dict(zip(keys, (f.strip() for f in line.split(","))))
        for line in text.splitlines()
        if line.strip()
    ]


def parse_pmon(text):
    """`nvidia-smi pmon` output -> one dict per process, keyed by pmon's own header names."""
    rows, names = [], None
    for line in text.splitlines():
        if line.startswith("# gpu"):
            names = line[1:].split()
        elif names and line.strip() and not line.startswith("#"):
            rows.append(dict(zip(names, line.strip().split(maxsplit=len(names) - 1))))
    return rows


def name_from(comm, argv, unit):
    """The program's name, or for an interpreter: its systemd service, -m module or script."""
    if comm not in GENERIC and not comm.startswith("python"):
        return comm
    if unit.endswith(".service"):
        return unit.removesuffix(".service")
    args = argv[1:]
    for i, a in enumerate(args):
        if a == "-m" and i + 1 < len(args):
            return args[i + 1]
        if a.endswith(
            ".py"
        ):  # keep the parent dir: a bare "service.py" or "main.py" says little
            return str(Path(*Path(a).parts[-2:]))
    return comm


def label(pid, fallback="?"):
    proc = Path("/proc", str(pid))
    try:
        comm = (proc / "comm").read_text().strip()
        argv = (proc / "cmdline").read_text().split("\0")
        unit = (proc / "cgroup").read_text().strip().rsplit("/", 1)[-1]
    except OSError:
        return fallback  # exited, or not visible from here (container, other pid namespace)
    return name_from(comm, argv, unit)


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def fmt(s, unit):
    """'61' -> '61°C'; '[N/A]' -> '–'."""
    try:
        return f"{float(s):.0f}{unit}"
    except (TypeError, ValueError):
        return "–"


def bar(frac, width=30):
    n = round(max(0.0, min(1.0, frac)) * width)
    return "█" * n + DIM + "·" * (width - n) + RESET


def frame(max_rows=None):
    gpus = parse_gpus(smi(f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader,nounits"))
    if not gpus:
        return "nvidia-smi found no GPU (is the NVIDIA driver loaded?)"
    procs = sorted(
        parse_pmon(smi("pmon", "-c", "1", "-s", "um")), key=lambda r: -num(r.get("fb"))
    )
    total = {g["index"]: num(g["memory.total"]) or 1 for g in gpus}
    multi = len(gpus) > 1
    out = []
    for g in gpus:
        tag = f"[{g['index']}] " if multi else ""
        out += [
            (
                f"{BOLD}{tag}{g['name']}{RESET}   {fmt(g['temperature.gpu'], '°C')}   "
                f"{fmt(g['power.draw'], '')}/{fmt(g['power.limit'], ' W')}   "
                f"fan {fmt(g['fan.speed'], '%')}   {fmt(g['clocks.sm'], ' MHz')}"
            ),
            f"GPU  {bar(num(g['utilization.gpu']) / 100)} {fmt(g['utilization.gpu'], ' %'):>7}",
            f"VRAM {bar(num(g['memory.used']) / total[g['index']])} {g['memory.used']:>6} / {g['memory.total']} MiB",
        ]
    out[0] += f"   {DIM}{time.strftime('%H:%M:%S')}{RESET}"

    gcol = 4 if multi else 0
    barw = max(0, min(30, shutil.get_terminal_size().columns - 66 - gcol))
    out += [
        "",
        (
            f"{BOLD}{'PROCESS':<20}{'GPU ' if multi else ''}{'PID':>8} {'TYPE':>4} {'SM%':>4} {'MEM%':>4} "
            f"{'ENC':>4} {'DEC':>4} {'VRAM MiB':>9}{RESET}"
        ),
    ]
    room = len(procs) if max_rows is None else max(1, max_rows - len(out))
    shown = procs if len(procs) <= room else procs[: room - 1]
    for r in shown:
        color = (
            CYAN if "C" in r.get("type", "") else DIM
        )  # compute users stand out, graphics-only dims
        gpu = f"{r.get('gpu', ''):>3} " if multi else ""
        out.append(
            f"{color}{label(r['pid'], r.get('command', '?'))[:19]:<20}{RESET}{gpu}{r['pid']:>8} "
            f"{r.get('type', ''):>4} {r.get('sm', '-'):>4} {r.get('mem', '-'):>4} {r.get('enc', '-'):>4} "
            f"{r.get('dec', '-'):>4} {r.get('fb', '-'):>9}  {bar(num(r.get('fb')) / total.get(r.get('gpu'), 1), barw)}"
        )
    if not procs:
        out.append(
            f"{DIM}(no processes, or `nvidia-smi pmon` is not supported on this GPU){RESET}"
        )
    elif len(shown) < len(procs):
        out.append(f"{DIM}… {len(procs) - len(shown)} more using less VRAM{RESET}")
    return "\n".join(out)


def safe_frame(max_rows=None):
    try:
        return frame(max_rows)
    except (OSError, subprocess.SubprocessError) as e:
        return f"nvidia-smi failed: {e}"


def main():
    global RESET, DIM, BOLD, CYAN
    ap = argparse.ArgumentParser(
        description="Live view of what is on your NVIDIA GPU, with real process names."
    )
    ap.add_argument(
        "-n",
        "--interval",
        type=float,
        default=1.0,
        metavar="SEC",
        help="pause between refreshes (default 1; each sample itself takes about 1 s)",
    )
    ap.add_argument("--once", action="store_true", help="print one snapshot and exit")
    args = ap.parse_args()

    tty = sys.stdout.isatty()
    if os.environ.get("NO_COLOR") or not tty:
        RESET = DIM = BOLD = CYAN = ""
    if args.once or not tty:
        print(safe_frame())
        return

    signal.signal(
        signal.SIGTERM, signal.default_int_handler
    )  # restore the terminal on kill too
    sys.stdout.write(
        "\033[?1049h\033[?25l\033[?7l"
    )  # alternate screen, hide cursor, no line wrap
    try:
        while True:
            text = safe_frame(max_rows=shutil.get_terminal_size().lines - 1)
            sys.stdout.write("\033[H" + text.replace("\n", "\033[K\n") + "\033[K\033[J")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?7h\033[?25h\033[?1049l")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
