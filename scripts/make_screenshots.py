"""Run real commands and render their output as terminal screenshots for the README.

    python scripts/make_screenshots.py [--out docs/assets]

Nothing is mocked: each image is the captured stdout of the command shown in its
title bar, run against the bundled weights on the current machine.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
BG, BAR, FG, DIM, PROMPT = "#1a1a19", "#2b2b29", "#f0efec", "#a3a29a", "#3987e5"


def run(*args: str, timeout: float = 600) -> str:
    r = subprocess.run(
        args, capture_output=True, text=True, encoding="utf-8", cwd=ROOT, timeout=timeout
    )
    if r.returncode != 0:
        raise SystemExit(f"command failed: {args}\n{r.stdout}\n{r.stderr}")
    return r.stdout + r.stderr


def render(
    title: str, command: str, output: str, path: Path, max_lines: int = 40, width_chars: int = 96
) -> None:
    lines = output.rstrip("\n").splitlines()
    if len(lines) > max_lines:
        keep = max_lines - 1
        lines = [*lines[:keep], f"... ({len(lines) - keep} more lines)"]
    lines = [ln if len(ln) <= width_chars else ln[: width_chars - 3] + "..." for ln in lines]
    line_h, char_w = 0.19, 0.0835  # inches at fontsize 10 monospace
    h = 0.75 + line_h * (len(lines) + 2)
    w = 0.6 + char_w * width_chars
    fig = plt.figure(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, h - 0.42), w, 0.42, boxstyle="square,pad=0", fc=BAR, ec="none"))
    for i, c in enumerate(("#e34948", "#eda100", "#1baf7a")):
        ax.add_patch(Circle((0.25 + i * 0.2, h - 0.21), 0.06, fc=c, ec="none"))
    ax.text(
        w / 2,
        h - 0.21,
        title,
        color=DIM,
        ha="center",
        va="center",
        fontsize=9.5,
        family="DejaVu Sans",
    )
    y = h - 0.42 - 0.3
    ax.text(0.3, y, "$", color=PROMPT, family="DejaVu Sans Mono", fontsize=10, va="center")
    ax.text(0.55, y, command, color=FG, family="DejaVu Sans Mono", fontsize=10, va="center")
    for ln in lines:
        y -= line_h
        ax.text(0.3, y, ln, color=FG, family="DejaVu Sans Mono", fontsize=10, va="center")
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    print(f"wrote {path}")


def server_session() -> tuple[str, str]:
    port = 8765
    proc = subprocess.Popen(
        [PY, "-m", "pagellm", "serve", "--port", str(port), "--log-level", "warning"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(base + "/health", timeout=1)
                break
            except OSError:
                time.sleep(0.1)

        def post(body: dict) -> bytes:
            req = urllib.request.Request(
                base + "/v1/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            return urllib.request.urlopen(req, timeout=60).read()

        out = []
        body = {"prompt": "JULIET:\n", "max_tokens": 60, "temperature": 0.7, "seed": 3}
        out.append(f"$ curl -s {base}/v1/completions -d '{json.dumps(body)}'")
        out.append(json.dumps(json.loads(post(body)), indent=2))
        sbody = {"prompt": "ROMEO:\n", "max_tokens": 4, "stream": True, "seed": 3}
        out.append("")
        out.append(f"$ curl -sN {base}/v1/completions -d '{json.dumps(sbody)}'")
        for chunk in post(sbody).decode().split("\n\n"):
            if chunk.startswith("data: "):
                data = chunk[6:]
                if data != "[DONE]":
                    c = json.loads(data)["choices"][0]
                    data = (
                        f'{{"text": {json.dumps(c["text"])}, '
                        f'"finish_reason": {json.dumps(c["finish_reason"])}, ...}}'
                    )
                out.append("data: " + data)
        out.append("")
        out.append(f"$ curl -s {base}/metrics | grep -E 'tokens_total|kv_blocks'")
        metrics = urllib.request.urlopen(base + "/metrics").read().decode()
        out += [
            ln
            for ln in metrics.splitlines()
            if not ln.startswith("#") and ("tokens_total" in ln or "kv_blocks" in ln)
        ]
        return f"pagellm serve --port {port}", "\n".join(out)
    finally:
        proc.terminate()
        proc.wait(timeout=15)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "docs/assets")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    tests = run(PY, "-m", "pytest", "-p", "no:cacheprovider", "--cov", "--cov-report=term")
    keep = [
        ln
        for ln in tests.splitlines()
        if ln.startswith(("tests", "Name", "-----", "TOTAL", "=")) or "passed" in ln
    ]
    render(
        "pytest: unit, integration and end-to-end",
        "pytest --cov",
        "\n".join(keep),
        a.out / "screenshot-tests.png",
        max_lines=48,
    )

    gen = run(
        PY,
        "-m",
        "pagellm",
        "generate",
        "ROMEO:",
        "--max-tokens",
        "260",
        "--temperature",
        "0.8",
        "--seed",
        "1",
    )
    render(
        "pagellm generate",
        'pagellm generate "ROMEO:" --max-tokens 260 --temperature 0.8 --seed 1',
        gen,
        a.out / "screenshot-generate.png",
    )

    cmd, session = server_session()
    render(
        "OpenAI-compatible server",
        cmd + "   # then, from another shell:",
        session,
        a.out / "screenshot-server.png",
        max_lines=48,
        width_chars=110,
    )

    from pagellm.benchmarks.runner import _server_process

    with _server_process(["--chunked-prefill", "--max-num-batched-tokens", "64"]) as url:
        load = run(
            PY, "-m", "pagellm", "loadtest", "--url", url, "--requests", "64", "--concurrency", "16"
        )
    render(
        "HTTP load test against a chunked-prefill server",
        f"pagellm loadtest --url {url} --requests 64 --concurrency 16",
        load,
        a.out / "screenshot-loadtest.png",
    )

    walk = run(PY, "examples/04_paged_kv_walkthrough.py")
    render(
        "paged KV cache walkthrough",
        "python examples/04_paged_kv_walkthrough.py",
        walk,
        a.out / "screenshot-paged-kv.png",
        width_chars=110,
    )


if __name__ == "__main__":
    main()
