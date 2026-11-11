# gpuwho

**See what is on your NVIDIA GPU, with real names instead of a wall of `python`.**

`nvidia-smi` tells you that `python` is using 2.7 GB and another `python` is using 1 GB.
`gpuwho` tells you which is which: it names each process by the systemd service it runs
under, or by its script or `-m` module. It refreshes live in your terminal and works over SSH.

```
NVIDIA GeForce RTX 5070 Ti   44°C   17/300 W   fan 30%   390 MHz   14:02:11
GPU  ······························     0 %
VRAM ██████████····················   5337 / 16303 MiB

PROCESS                  PID TYPE  SM% MEM%  ENC  DEC  VRAM MiB
upload-worker        2843297    C    -    -    -    -      2770  █████·························
pose-api             2831776    C    -    -    -    -      1048  ██····························
sunshine                8134  C+G    -    -    -    -       231  ······························
Hyprland                7180    G    -    -    -    -       167  ······························
Discord              1744933  C+G    -    -    -    -        86  ······························
```

Compare `nvidia-smi` for the same two processes:

```
|    0   N/A  N/A         2831776      C   ...e-3c1f0a92d4b7e815/bin/python       1048MiB |
|    0   N/A  N/A         2843297      C   ...ools/app/.venv/bin/python           2770MiB |
```

## Install

It is one Python file with no dependencies (Python 3.9+, Linux, NVIDIA driver with `nvidia-smi`).

```sh
curl -fsSL https://raw.githubusercontent.com/dishishshawn/gpuwho/main/gpuwho.py -o ~/.local/bin/gpuwho
chmod +x ~/.local/bin/gpuwho
```

## Use

```sh
gpuwho            # live view, Ctrl-C to quit
gpuwho -n 5       # refresh less often
gpuwho --once     # one snapshot, e.g. over ssh or into a file
```

Output that is not a terminal (a pipe or a file) gets one plain snapshot. `NO_COLOR` is respected.

## How names are picked

1. A normal program keeps its own name: `ollama`, `sunshine`, `Xorg`.
2. An interpreter (`python*`, PyTorch's `pt_main_thread`, `node`, `java`) is named by its
   systemd service when it has one: `ollama.service` → `ollama`, `whisper.service` → `whisper`.
   This works for system and `--user` services alike.
3. Otherwise by its `-m` module (`vllm.entrypoints.openai.api_server`) or its script with
   its parent folder (`pose/service.py`).

Compute processes are highlighted, graphics-only ones are dimmed, and the list is sorted
by VRAM. Per-process SM%, memory and encoder/decoder load come from `nvidia-smi pmon`.

## What it does not do

- It is read-only. It never signals, pauses or changes a process or GPU setting.
- Only NVIDIA GPUs. Several GPUs are fine; a GPU column then appears.
- Processes inside containers are named from the host; if their PID is not visible,
  you get the name `nvidia-smi` reports.

## Test

```sh
python3 test_gpuwho.py   # no GPU needed
```

## License

MIT
