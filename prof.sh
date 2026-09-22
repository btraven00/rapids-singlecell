#!/bin/sh
# Profile an entrypoint with denet, as an omnibenchmark entrypoint.
#
#   pca-prof: prof.sh pca.py
#
# Three things this works around, all verified against denet 0.6.0:
#
#  1. ob runs `./{entrypoint}` directly, so the conventional `denet pca.py`
#     form resolves to ./denet -- a file that does not exist in the module dir.
#     This wrapper IS the entrypoint, and `prof.sh pca.py` expands correctly.
#  2. denet needs a `run` subcommand and a `--` separator, or it parses the
#     module's own flags (--output_dir, --name) as its options.
#  3. denet cannot profile a shebang script: it execs the script, the kernel
#     swaps in the interpreter, and denet loses the process -- `cmd` comes back
#     empty and NO samples are written, while the script runs fine, so it fails
#     silently. The interpreter has to be named explicitly, so the shebang is
#     resolved here.
#
# FOURTH gotcha, found 2026-09-21: denet's sampling is ADAPTIVE. --interval is
# only the BASE; it backs off toward --max-interval (default 1000ms). Measured
# with --interval 50 alone: intervals ran 346, 233, 255, ... 586, 642ms, a mean
# of 376ms, so a 68ms `write` phase got ZERO samples and a 1669ms `pca` phase
# got one. Both flags must be pinned to get a fixed rate.
#
# Samples land beside the module's obkit-events.jsonl so the two align: denet
# gives RSS/threads/GPU every --interval ms, obkit gives phase boundaries, and
# the intersection is PER-PHASE memory. That is the only way to get compute
# memory uncontaminated by the load phase -- performance.txt reports one peak
# for the whole job, which for a module that materialises a matrix during load
# is the load's peak, not the algorithm's.
set -eu
# Pinned sampling rate; override per-run if a long job makes 50ms too many rows.
: "${DENET_INTERVAL_MS:=50}"
# denet is a single static binary and lives outside every conda env, so a job
# that has activated one may not have it on PATH.
if ! command -v denet >/dev/null 2>&1; then
  for c in "$HOME/.cargo/bin" "$HOME/bin"; do
    [ -x "$c/denet" ] && PATH="$c:$PATH" && export PATH && break
  done
fi
target="$1"
shift

out="."
prev=""
for a in "$@"; do
  [ "$prev" = "--output_dir" ] && out="$a"
  prev="$a"
done
mkdir -p "$out"

# Resolve the shebang: "#!/usr/bin/env python3" -> python3, "#!/usr/bin/Rscript"
# -> /usr/bin/Rscript. Falls back to executing directly if there is no shebang
# (a real binary, which denet handles).
shebang=$(head -1 "$target" | sed -n 's|^#! *||p')
case "$shebang" in
  */env\ *) interp=${shebang#*env } ;;
  "")       interp="" ;;
  *)        interp=$shebang ;;
esac

if [ -n "$interp" ]; then
  exec denet --json --out "$out/denet-samples.jsonl" \
       --interval "$DENET_INTERVAL_MS" --max-interval "$DENET_INTERVAL_MS" \
       --quiet run -- $interp "$target" "$@"
else
  exec denet --json --out "$out/denet-samples.jsonl" \
       --interval "$DENET_INTERVAL_MS" --max-interval "$DENET_INTERVAL_MS" \
       --quiet run -- "./$target" "$@"
fi
