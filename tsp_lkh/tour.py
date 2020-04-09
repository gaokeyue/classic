from typing import Iterable
import numpy as np

class Tour:
    """Array implementation. Leave the fancy 'two-level-tree' for later."""

    def __init__(self, route: Iterable[int]):
        """Accept any iterable of some permutation of range(n). Vertex 0 is fixed as the starting place"""
        self.route = np.asarray(route)
        if self.route[0] != 0:
            raise ValueError(f"Vertex 0 should be fixed as the starting place")

    def two_opt(self):
        pass

