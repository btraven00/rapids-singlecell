from dataclasses import dataclass
from typing import Optional


# Methods whose underlying algorithm has no notion of a random seed.
# Passing --random_seed for any of these is rejected to avoid the false
# impression that the run is seed-controlled.
_UNSEEDED_METHODS = frozenset({
    "rapids-louvain",
    "rapids-hdbscan",
    "rapids-dbscan",
})


@dataclass
class ClusterOptions:
    method: str
    resolution: float
    random_seed: Optional[int]


@dataclass
class ClusterEmbeddingOptions:
    method: str
    random_seed: Optional[int]
    n_clusters: Optional[int] = None
    min_samples: Optional[int] = None
    min_cluster_size: Optional[int] = None
    eps: Optional[float] = None


def _check_seed(method: str, random_seed: Optional[int]) -> None:
    if method in _UNSEEDED_METHODS:
        if random_seed is not None:
            raise ValueError(
                f"--random_seed is not accepted for {method} "
                f"(algorithm has no seed parameter)"
            )
    else:
        if random_seed is None:
            raise ValueError(f"--random_seed is required for {method}")


def build_cluster_opts(args) -> ClusterOptions:
    _check_seed(args.method, args.random_seed)
    return ClusterOptions(
        method=args.method,
        resolution=args.resolution,
        random_seed=args.random_seed,
    )


def build_cluster_embedding_opts(args) -> ClusterEmbeddingOptions:
    if args.method == "rapids-kmeans" and args.n_clusters is None:
        raise ValueError("--n_clusters is required for rapids-kmeans")
    if args.method == "rapids-hdbscan":
        if args.min_samples is None:
            raise ValueError("--min_samples is required for rapids-hdbscan")
        if args.min_cluster_size is None:
            raise ValueError("--min_cluster_size is required for rapids-hdbscan")
    if args.method == "rapids-dbscan":
        if args.eps is None:
            raise ValueError("--eps is required for rapids-dbscan")
        if args.min_samples is None:
            raise ValueError("--min_samples is required for rapids-dbscan")
    _check_seed(args.method, args.random_seed)
    return ClusterEmbeddingOptions(
        method=args.method,
        random_seed=args.random_seed,
        n_clusters=args.n_clusters,
        min_samples=args.min_samples,
        min_cluster_size=args.min_cluster_size,
        eps=args.eps,
    )
