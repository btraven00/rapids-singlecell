"""pixi.lock -> conda env YAML pinning every package to version and build (reads `pixi list --json`).

ob copies only the env YAML into out/.envs/, so Snakemake never sees a sibling
*.pin.txt explicit spec. A YAML with every package pinned to name=version=build
leaves the solver no choice, and rebuilds the locked environment.
Usage: pixi list --json --platform linux-64 | python envs/lock.py scanpy > envs/scanpy.linux-64.lock.yml
"""
import json
import sys

name = sys.argv[1]
pkgs = json.load(sys.stdin)
conda = sorted((p for p in pkgs if p["kind"] == "conda"), key=lambda p: p["name"])
pypi = sorted((p for p in pkgs if p["kind"] == "pypi"), key=lambda p: p["name"])
channels = sorted({p["source"] for p in conda}, key=lambda c: "conda-forge" not in c)
print("# generated from pixi.lock by envs/lock.py - do not edit by hand")
print(f"name: {name}")
print("channels:")
for c in channels:
    print(f"- {c.rstrip('/')}")
print("- nodefaults")
print("dependencies:")
for p in conda:
    print(f"- {p['source'].rstrip('/')}::{p['name']}={p['version']}={p['build']}")
if pypi:
    print("- pip:")
    for p in pypi:
        print(f"  - {p['name']}=={p['version']}")
