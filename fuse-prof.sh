#!/usr/bin/env bash
# fuse.py under denet. A plain wrapper because ob 0.7.0 resolves an entrypoint
# value like "denet fuse.py" as a file path and ends up running
# `python3 denet fuse.py` (resolver._check_shebang).
# The trace lands next to fuse.py's obkit-events.jsonl, and the two are joined on wall-clock time.
set -euo pipefail
out=""
for ((i = 1; i < $#; i++)); do [[ ${!i} == --output_dir ]] && { j=$((i + 1)); out=${!j}; }; done
[[ -n $out ]] || { echo "error: --output_dir is required" >&2; exit 2; }
mkdir -p "$out"
# Thread limit, N = --threads (0 or missing = every core we are allowed).
# Hard: pin the whole tree (denet included, so its env record shows the pin) to N
# of the inherited cores. Soft: size every pool to N, so they don't oversubscribe
# the pinned cores (OpenBLAS, OpenMP/sklearn, MKL, numba, polars/rayon, numexpr;
# vecLib on macOS). macOS has no affinity, so only the pool sizes apply there.
n=0
for ((i = 1; i < $#; i++)); do [[ ${!i} == --threads ]] && { j=$((i + 1)); n=${!j}; }; done
pin=()
if [[ $(uname) == Linux ]]; then
  # one hardware thread per physical core, from the cores we may use (Slurm/cgroup already narrowed these)
  cores=$(python3 - <<'PY'
import os
seen, pick = set(), []
for c in sorted(os.sched_getaffinity(0)):
    try:
        sib = open(f"/sys/devices/system/cpu/cpu{c}/topology/core_cpus_list").read().strip()
    except OSError:
        sib = str(c)
    if sib not in seen:
        seen.add(sib); pick.append(c)
print(" ".join(map(str, pick)))
PY
)
  read -ra cores <<< "$cores"
  (( n > 0 && n <= ${#cores[@]} )) || n=${#cores[@]}
  pin=(taskset -c "$(IFS=,; echo "${cores[*]:0:n}")")
else
  (( n > 0 )) || n=$(sysctl -n hw.ncpu)
fi
for v in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMBA_NUM_THREADS \
         POLARS_MAX_THREADS RAYON_NUM_THREADS NUMEXPR_MAX_THREADS VECLIB_MAXIMUM_THREADS; do
  export "$v=$n"
done

# --gpu: NVML sampling. Without a usable GPU, denet prints a warning and keeps going (CPU methods).
exec "${pin[@]}" denet --json --quiet --gpu --write-env --out "$out/denet.jsonl" \
  run python3 "$(dirname "$0")/fuse.py" "$@"
