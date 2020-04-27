from typing import Sequence
import numpy as np


class TourDoubleList:
    """Doubly circular linked list implementation of tour."""

    def __init__(self, route: Sequence):
        """Accept any sequence of permutation of range(n). """
        self.size = len(route)
        if sorted(route) != list(range(self.size)):
            raise ValueError(f"Input must be a permutation of {self.size}")
        # convert the sequence to doubly linked list.
        # self.links[i, j] is the predecessor of i if j is 0, successor of i if j is 1
        self.links = np.zeros((self.size, 2))
        last = route[0]
        for curr in route[1:]:
            self.links[last, 1] = curr
            self.links[curr, 0] = last
            last = curr
        self.links[last, 1] = route[0]
        self.links[0, 0] = last

    def succ(self, i):
        return self.links[i, 1]

    def pred(self, i):
        return self.links[i, 0]

    def neighbours(self, i):
        return self.links[i]

    def iter_vertices(self, start=0, reverse=False):
        """return the route as a sequence"""
        yield start
        orientation = 0 if reverse else 1
        next = self.links[start, orientation]
        while next != start:
            yield next

    def iter_links(self, include_reverse=True):
        """return all links in tour"""
        start = 0
        curr = start
        next = self.links[curr, 1]
        while next != start:
            yield curr, next
            if include_reverse:
                yield next, curr
            curr, next = next, self.links[next, 1]

    def check_feasible(self, v):
        """Walk around the tour, O(n) complexity"""
        # step 1, determine the order of v[0], v[1],...,v[2k-1] with orientation v[0]-> v[1] in self.
        p = np.arange(len(v))  # Fix p[0] = 0, and p[1] = 1
        orientation = 1 if self.links[v[0], 1] == v[1] else 0
        q = p.copy()  # the inverse permutation of v, q = [p.index(i) for i in range(2k)]
        incl = p.copy()
        # First jump v[0] to v[2k-1] = v[-1]
        i = q[v[-1]


        
            
            

