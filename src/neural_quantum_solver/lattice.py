from dataclasses import dataclass


@dataclass(frozen=True)
class Graph:
    num_sites: int
    edges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        for i, j in self.edges:
            if i == j or not (0 <= i < self.num_sites and 0 <= j < self.num_sites):
                raise ValueError(f"invalid edge {(i, j)}")

    @classmethod
    def chain(cls, num_sites: int, *, periodic: bool = False) -> "Graph":
        edges = [(i, i + 1) for i in range(num_sites - 1)]
        if periodic and num_sites > 2:
            edges.append((num_sites - 1, 0))
        return cls(num_sites, tuple(edges))
