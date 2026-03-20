"""
Peer Query Protocol — kernels ask each other for help.

Mesh topology (NOT ring). Interior nodes have 3+ connections
for topological protection (EXP-002: 66.9x bulk/surface ratio).

Protocol: kernel broadcasts query basin -> all peers compute
d_FR(query, self.basin) -> peers with d < threshold respond
with their local view -> requester integrates via Frechet mean.

Plan reference: L3 (Structural Leg 3)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..coordizer_v2.geometry import Basin, fisher_rao_distance, frechet_mean

logger = logging.getLogger("vex.consciousness.peer_query")


@dataclass
class PeerResponse:
    """Response from a peer kernel to a query."""

    kernel_id: str
    basin: Basin
    confidence: float
    distance: float


class PeerQueryBus:
    """Mesh communication bus for kernel constellation.

    Each kernel registers its current basin. Query broadcasts
    find relevant peers by Fisher-Rao proximity.

    Interior nodes have 3+ connections for topological protection.
    """

    def __init__(self, relevance_threshold: float = 0.8) -> None:
        self._peers: dict[str, Basin] = {}
        self._threshold = relevance_threshold

    def register(self, kernel_id: str, basin: Basin) -> None:
        """Register or update a kernel's current basin."""
        self._peers[kernel_id] = basin

    def unregister(self, kernel_id: str) -> None:
        """Remove a kernel from the bus."""
        self._peers.pop(kernel_id, None)

    def query(self, query_basin: Basin, requester_id: str) -> list[PeerResponse]:
        """Broadcast query to all peers. Return sorted by relevance (closest first)."""
        responses = []
        for kid, peer_basin in self._peers.items():
            if kid == requester_id:
                continue
            d = fisher_rao_distance(query_basin, peer_basin)
            if d < self._threshold:
                responses.append(
                    PeerResponse(
                        kernel_id=kid,
                        basin=peer_basin,
                        confidence=1.0 - (d / self._threshold),
                        distance=d,
                    )
                )
        return sorted(responses, key=lambda r: r.distance)

    def integrate_responses(self, responses: list[PeerResponse]) -> Basin | None:
        """Integrate peer responses via Frechet mean, weighted by confidence."""
        if not responses:
            return None
        basins = [r.basin for r in responses]
        weights = np.array([r.confidence for r in responses])
        weights = weights / weights.sum()
        return frechet_mean(basins, weights=weights)

    def mesh_topology(self) -> dict[str, list[str]]:
        """Return adjacency list. Interior nodes should have 3+ connections."""
        adj: dict[str, list[str]] = {k: [] for k in self._peers}
        for k1, b1 in self._peers.items():
            for k2, b2 in self._peers.items():
                if k1 >= k2:
                    continue
                d = fisher_rao_distance(b1, b2)
                if d < self._threshold:
                    adj[k1].append(k2)
                    adj[k2].append(k1)
        return adj

    def interior_nodes(self) -> list[str]:
        """Return nodes with 3+ connections (topologically protected)."""
        topo = self.mesh_topology()
        return [k for k, neighbors in topo.items() if len(neighbors) >= 3]

    @property
    def n_peers(self) -> int:
        return len(self._peers)

    def get_state(self) -> dict:
        topo = self.mesh_topology()
        return {
            "n_peers": self.n_peers,
            "interior_nodes": len(self.interior_nodes()),
            "connectivity": {k: len(v) for k, v in topo.items()},
        }
