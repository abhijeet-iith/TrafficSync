"""Road and communication graphs."""
from .graph import Link, Movement, Network, OPPOSITE
from .spectral import spectral_summary
from .topology import apply_communication, comm_adjacency, complete, corridor, erdos_renyi, grid, k_hop, load_network, ring

__all__ = ["Link", "Movement", "Network", "OPPOSITE", "spectral_summary", "apply_communication", "comm_adjacency",
           "corridor", "grid", "load_network", "erdos_renyi", "ring", "complete", "k_hop"]
