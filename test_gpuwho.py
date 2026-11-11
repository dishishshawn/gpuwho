"""No GPU needed. Run: python3 test_gpuwho.py"""

import gpuwho as g

PMON = """\
# gpu         pid   type     sm    mem    enc    dec    jpg    ofa     fb   ccpm    command
# Idx           #    C/G      %      %      %      %      %      %     MB     MB    name
    0     41207     C     63     18      -      -      -      -   6412      0    ollama
    1     52233     C      -      -      -      -      -      -   2770      0    python
    0      7180     G      -      -      -      -      -      -    167      0    Hyprland
"""
rows = g.parse_pmon(PMON)
assert [r["pid"] for r in rows] == ["41207", "52233", "7180"], rows
assert rows[0]["sm"] == "63" and rows[0]["fb"] == "6412" and rows[1]["gpu"] == "1"
assert rows[1]["command"] == "python" and rows[2]["sm"] == "-"
assert g.parse_pmon("Failed to initialize NVML") == []

gpus = g.parse_gpus("0, NVIDIA GeForce RTX 4090, 64, 13410, 24564, 61, 212.40, 450.00, [N/A], 2715\n")
assert gpus[0]["name"] == "NVIDIA GeForce RTX 4090" and gpus[0]["memory.total"] == "24564"
assert g.fmt(gpus[0]["fan.speed"], "%") == "–" and g.fmt(gpus[0]["power.draw"], " W") == "212 W"
assert g.num("-") == 0.0 and g.num("2770") == 2770.0

assert g.name_from("ollama", ["/usr/bin/ollama", "runner", ""], "ollama.service") == "ollama"
assert g.name_from("python", ["/venv/bin/python", "service.py", ""], "pose-api.service") == "pose-api"
assert g.name_from("python3.12", ["python3", "-m", "vllm.entrypoints.api_server", ""], "session-2.scope") == (
    "vllm.entrypoints.api_server"
)
assert g.name_from("pt_main_thread", ["python", "train.py", "-m", "x", ""], "app-kitty-1.scope") == "train.py"
assert g.name_from("python", ["python", "/srv/app/pose/service.py", ""], "tmux-1.scope") == "pose/service.py"
assert g.name_from("python", ["python", ""], "session-2.scope") == "python"
assert g.label(99999999, "fallback") == "fallback"  # above any possible pid_max
print("ok")
