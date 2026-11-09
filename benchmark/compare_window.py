"""Validate and compare the experimental window transport with an archived main.

Run on four GB200 GPUs from this repository, with the pinned BSA dependencies:
python -m benchmark.compare_window --baseline-root /path/to/main --output /path/to/evidence
Individual measurements use benchmark.perf_vsa without modifying its timing.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def source_hashes(root):
    paths = [p for folder in (root, root / "benchmark", root / "tests")
             for p in folder.glob("*.py")]
    paths += list((root / "bsa_cute").rglob("*.py"))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    baseline = args.baseline_root.resolve()
    output = args.output.resolve()
    if not (baseline / "benchmark/perf_vsa.py").is_file():
        parser.error("--baseline-root must contain the archived main benchmark")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"candidate": source_hashes(root), "baseline": source_hashes(baseline),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "steps": []}

    def run(name, command, *, cwd=root, impl="cudnn"):
        env = dict(os.environ, VSA_BSA_IMPL=impl)
        env["PYTHONPATH"] = str(cwd) + os.pathsep + env.get("PYTHONPATH", "")
        print(f"START {name}", flush=True)
        with (output / f"{name}.log").open("w") as log:
            completed = subprocess.run(command, cwd=cwd, env=env, stdout=log,
                                       stderr=subprocess.STDOUT)
        (output / f"{name}.exit").write_text(f"{completed.returncode}\n")
        manifest["steps"].append({"name": name, "command": command, "cwd": str(cwd),
                                  "bsa_impl": impl, "exit_code": completed.returncode})
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"END {name}: exit={completed.returncode}", flush=True)

    run("unit", [sys.executable, "-m", "pytest", "-q", "-m", "not fastvideo", "--disable-warnings"])
    # main contains illustrative fragments as well as two complete programs.
    # Parse every Python block; execute the complete Use and Shared example programs.
    readme = (root / "README.md").read_text()
    for i, block in enumerate(re.findall(r"```python\n(.*?)\n```", readme, re.S)):
        ast.parse(block, filename=f"README block {i}")
    for heading, name in (("## Use", "readme_use"), ("### Shared example", "readme_demo")):
        section = readme.split(heading + "\n", 1)[1]
        code = re.search(r"```python\n(.*?)\n```", section, re.S).group(1)
        path = output / f"{name}.py"
        path.write_text(code)
        run(name, [sys.executable, str(path)])
        if name == "readme_demo":
            for partition in ("cyclic", "contiguous"):
                run(f"{name}_{partition}", ["torchrun", "--standalone", "--nproc-per-node=4",
                    str(path), "--ulysses-size", "1", "--ring-size", "4", "--ring-partition", partition])

    launcher = ["torchrun", "--standalone", "--nproc-per-node=4", "--module", "benchmark.perf_vsa",
                "--ring", "--seed", "937", "--batch-p2p", "--no-profile"]
    for partition in ("cyclic", "contiguous"):
        for transport, impl in (("ring", "cudnn"), ("window", "local")):
            name = f"small_{partition}_{transport}"
            run(name, [*launcher, "--ring-size", "4", "--ring-partition", partition, "--transport", transport,
                       "--heads", "2", "--cases", "ragged", "dense",
                       "single64k", "thd64k", "--warmup", "3", "--repeats", "10",
                       "--output", str(output / f"{name}.json")], impl=impl)
            # This fixture has three global blocks; every participating rank
            # must own a block. Four workers form two interleaved Ring2 groups.
            name = f"partial_{partition}_{transport}"
            run(name, [*launcher, "--ring-size", "2", "--ring-partition", partition,
                       "--transport", transport, "--heads", "2", "--cases", "partial_blocks",
                       "--warmup", "3", "--repeats", "10",
                       "--output", str(output / f"{name}.json")], impl=impl)
        # Compare fixed global work, on the same allocation, with reversed order.
        for variant, round_id in (("main", 0), ("ring", 0), ("window", 0),
                                  ("window", 1), ("ring", 1), ("main", 1)):
            name = f"long_{partition}_{variant}_{round_id}"
            transport_args = [] if variant == "main" else ["--transport", variant]
            run(name, [*launcher, "--ring-size", "4", "--ring-partition", partition, *transport_args,
                       "--heads", "1", "--cases", "thd720p44s", "--input-order", "block",
                       "--bwd-bucket-size", "backend", "--warmup", "10", "--repeats", "50",
                       "--output", str(output / f"{name}.json")],
                cwd=baseline if variant == "main" else root,
                impl="local" if variant == "window" else "cudnn")
    manifest["sources_unchanged"] = (manifest["candidate"] == source_hashes(root)
                                     and manifest["baseline"] == source_hashes(baseline))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    failed = [s["name"] for s in manifest["steps"] if s["exit_code"]]
    print(f"COMPLETE failures={failed}, sources_unchanged={manifest['sources_unchanged']}", flush=True)
    raise SystemExit(int(bool(failed) or not manifest["sources_unchanged"]))


if __name__ == "__main__":
    main()
