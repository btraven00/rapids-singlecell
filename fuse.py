#!/usr/bin/env python3
"""Run typed steps in one process, inserting timed host<->device transfers (prototype).

  --steps PCA,NNG,CLUST   fused: intermediates stay where the steps put them
  --steps NNG             split: one stage, load -> (h2d) -> run -> (d2h) -> save

Phases, each closed only after the GPU is idle (artifacts.sync), each carrying its
PCIe bytes (pcie_rx_bytes host->GPU, pcie_tx_bytes GPU->host) in the end event:
  init                  device setup (CUDA context, RMM); zero-cost for a CPU module
  load                  read the chain's external inputs (host)
  h2d:<id> / d2h:<id>   one per boundary crossing, inserted from the type mismatch
  <stage>               one per step: compute only
  write                 save every output (host), after all d2h phases

Module-agnostic apart from the imports from `steps`: a CPU module has no TRANSFER
entries and a no-op sync, and gets the same runner. This is what would move into obkit.
"""

import argparse
import dataclasses
import gc
import os
import signal
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from phases import phase  # noqa: E402
from pcie import counters  # noqa: E402
from obkit.logger import emit, init_logger  # noqa: E402
from steps import IO, STEPS, TRANSFER, sync  # noqa: E402

import anndata as ad  # noqa: E402


def _head(v, n):
    """The first n cells of a loaded input, for the warm-up run."""
    if isinstance(v, ad.AnnData):
        return v[:n].copy()
    def cut(x):
        if isinstance(x, list):
            return x[:n]
        if hasattr(x, "shape") and len(x.shape) == 2 and x.shape[0] == x.shape[1]:
            return x[:n, :n]  # cell x cell graph
        return x[:n] if hasattr(x, "shape") else x
    return dataclasses.replace(v, **{f.name: cut(getattr(v, f.name)) for f in dataclasses.fields(v)})


def _host_of(t):
    """The host type a value of type t is loaded or saved as."""
    if t in IO:
        return t
    hosts = [h for (src, dst) in TRANSFER if dst is t and (h := src) in IO]
    hosts += [h for (src, h) in TRANSFER if src is t and h in IO]
    assert hosts, f"no host form for {t.__name__}"
    return hosts[0]


def plan(stages):
    """Typecheck a chain; return {external input id: type the first consumer wants}."""
    produced, external = {}, {}
    for st in (STEPS[s] for s in stages):
        for k, want in st.inputs.items():
            have = produced.get(k) or _host_of(want)
            assert have is want or (have, want) in TRANSFER, \
                f"{st.stage}: {k} is {have.__name__}, wants {want.__name__}, no transfer"
            if k not in produced:
                external[k] = want
        produced.update(st.outputs)
    return external


@contextmanager
def timed(name):
    """A phase that ends only when the GPU is idle, with its PCIe traffic in the end event:
    pcie_rx_bytes (host -> GPU) and pcie_tx_bytes (GPU -> host), device-wide (src/pcie.py)."""
    sync()
    c0 = counters()
    with phase(name) as attrs:
        yield attrs
        sync()
        c1 = counters()
        if c0 and c1:
            attrs["pcie_tx_bytes"], attrs["pcie_rx_bytes"] = c1[0] - c0[0], c1[1] - c0[1]


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--steps", required=True)
    known, _ = pre.parse_known_args(argv)
    stages = known.steps.split(",")
    external = plan(stages)

    p = argparse.ArgumentParser(description=f"fused steps: {known.steps}")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--steps", required=True)
    p.add_argument("--replicate", type=int, default=0)  # unused; separates same-seed replicate dirs
    # In-process replicates after one warm-up: replicate r writes to rep<r>/ (and r=0 also to
    # the declared outputs), with every *_random_seed + r * seed_stride (0: same-seed replicates).
    p.add_argument("--replicates", type=int, default=1)
    p.add_argument("--seed_stride", type=int, default=0)
    p.add_argument("--threads", type=int, default=0)    # run-wide; used by fuse-prof.sh
    # Run-wide: run the chain once on the first N cells under warmup:* phases and discard
    # it, so JIT/kernel compilation and device init land there, not in the timed phases.
    # N must be large enough to take the same code paths (scanpy kNN: >= 4096 cells).
    p.add_argument("--warmup_cells", type=int, default=0)
    for k in external:
        p.add_argument(f"--{k}", type=Path, required=True)
    # step parameters are namespaced by stage: --pca_dtype, --nng_n_neighbors, --clust_random_seed
    for st in (STEPS[s] for s in stages):
        for k, t in st.params.items():
            p.add_argument(f"--{st.stage.lower()}_{k}", type=t, required=True)
    return p.parse_args(argv), stages, external


def _as(env, k, want, prefix="", move=False):
    """Value k as type `want`, crossing the boundary (timed) if needed; cached.

    move=True replaces env[k] with the transferred value, so the source copy can be
    freed. The runner moves the chain's external inputs: they are never saved and never
    needed on the host again, so after h2d the host X would only hold RAM."""
    have = env[k]
    if type(have) is want:
        return have
    key = (k, want)
    if key not in env:
        f = TRANSFER[(type(have), want)]
        with timed(f"{prefix}{f.__name__}:{k}"):
            env[key] = f(have)
            if move:
                del have
                env[k] = env.pop(key)
                gc.collect()  # drop the host matrix now, not at some later collection
                return env[k]
    return env[key]


def _chain(env, stages, a, prefix=""):
    produced, external = [], set(env)
    for st in (STEPS[s] for s in stages):
        ins = {k: _as(env, k, t, prefix, move=k in external) for k, t in st.inputs.items()}
        with timed(prefix + st.stage.lower()):
            res = st.run(ins, {k: a[f"{st.stage.lower()}_{k}"] for k in st.params})
        env.update(res)
        produced += list(res)
    return {k: _as(env, k, _host_of(type(env[k])), prefix) for k in produced}


def _mem():
    """Current host RSS and device memory in use (device-wide: needs the exclusive GPU), for the
    creep check across replicates."""
    import cupy as cp
    m = {}
    try:
        m["rss_mb"] = round(int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20, 1)
    except OSError:
        pass
    free, total = cp.cuda.runtime.memGetInfo()
    m["gpu_used_mb"], m["gpu_total_mb"] = round((total - free) / 2**20, 1), round(total / 2**20, 1)
    return m


def main(argv=None):
    from gpu import setup_gpu  # this module's RMM settings; a CPU module has none
    args, stages, external = parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    init_logger(str(out))
    a = vars(args)
    state = {"replicate": None, "done": 0}

    # Why we quit, in the event log: ok / sigterm (the runner's time limit) / error. Python runs
    # the handler between bytecodes, so a long kernel or C call delays it; the runner's own
    # record of SIGTERM / kill / OOM is authoritative, this says where the module was.
    def on_term(*_):
        emit("exit", "end", attrs=dict(state, reason="sigterm"))
        sys.exit(128 + signal.SIGTERM)
    signal.signal(signal.SIGTERM, on_term)
    try:
        with timed("init"):  # CUDA context + RMM allocator: a fixed cost every run pays
            setup_gpu()
        with timed("load"):
            env = {k: IO[_host_of(t)].load(a[k]) for k, t in external.items()}
        if args.warmup_cells:
            _chain({k: _head(v, args.warmup_cells) for k, v in env.items()}, stages, a, "warmup:")
        for r in range(args.replicates):
            state["replicate"] = r
            ar = {k: v + r * args.seed_stride if k.endswith("_random_seed") else v for k, v in a.items()}
            if r:  # fresh host inputs (the last replicate moved X to the device) and a collected heap
                env = host = None
                gc.collect()
                with timed("load"):
                    env = {k: IO[_host_of(t)].load(a[k]) for k, t in external.items()}
            with phase("replicate") as attrs:
                attrs.update(replicate=r, seeds={k: v for k, v in ar.items() if k.endswith("_random_seed")})
                host = _chain(env, stages, ar)
                attrs.update(_mem())
            with timed("write"):  # after each replicate, so a SIGTERM keeps the finished ones
                for d in [out / f"rep{r}"] + ([out] if r == 0 else []):
                    d.mkdir(exist_ok=True)
                    for k, v in host.items():
                        IO[type(v)].save(v, d / f"{args.name}{IO[type(v)].suffix}")
            state["done"] = r + 1
    except Exception as e:
        emit("exit", "end", attrs=dict(state, reason="error", error=f"{type(e).__name__}: {e}"))
        raise
    emit("exit", "end", attrs=dict(state, reason="ok"))


if __name__ == "__main__":
    main()
