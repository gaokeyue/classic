from __future__ import annotations
import numpy as np

from .tour import TourDoubleList as Tour


class TSP_LKH:
    def __init__(self, cost: np.array):
        self.cost = cost  # symmetric n-by-n numpy array
        self.size = len(cost)
        self.max_exchange = 5
        self.candidates = {}  # self.candidates must support __getitem__

    def improve(self, tour: Tour):
        """Improve a tour by a variable-exchange, at most 5-exchange.
        We would like the three variables, i, break_vs and gain, to be shared by all recursion calls,
        so two options are available to deal with the problem that i and gain are immutable:
        (1) place the recursion function inside the improve function (as an inner function) and use the nonlocal
        trick. The nonlocal label is effective throught all recursions;
        (2) place the recursion function outside the improve function, but wrap all the variables inside
        a mutable variable, e.g a dictionary, and then pass this dummy mutable variable into all recursions.
        Approach (2) is the one adopted here, because it's probably more flexible,
        in case the recursion function will be called by another function in addition to the improve function"""
        for v0, v1 in tour.iter_links():
            # mu is a dicionary with fixed keys: i, break_vs and gain.
            # break_vs is vertices to break. In order to conform with LKH paper, 1-based index is used,
            # hence add the sentinel None to break_vs[0].
            mu = {'i': 1, 'break_vs': [v0, v1], 'gain': 0}
            tour = self.dfs_recursion(tour, mu)
            if tour is not None:
                return tour

    def dfs_recursion(self, tour, mu):
        """depth-first-search by recursion called by self.improve.
        If a feasible and profitable tour is found beyond break_vs = [v1, v2,..., v_(2i-1), v_(2i)],
        this function returns the tour. Otherwise return None."""
        i = mu['i']
        # v is an alias for break_vs, the vertices to be broken. It's used for more readable indexing,
        # particularly in self.cost. In other usage, use mu['break_vs'] instead.
        v = mu['break_vs']
        v_2i_1 = v[2 * i - 1]
        gain = mu['gain']
        for v_2i in self.candidates[v_2i_1]:
            if v_2i in v:  # disjunctivity criterion
                continue
            delta_gain = self.cost[v[2 * i - 2], v[2 * i - 1]] - self.cost[v[2 * i - 1], v_2i]
            if gain + delta_gain <= 0:  # positive provisional gain criterion
                continue
            v.append(v_2i)
            mu['gain'] = gain + delta_gain
            # y[i-1] has been found--------------------------------------------------------
            # now search for x[i] = (v_(2i), v_(2i+1))--------------------------------------
            for u in tour.neighbours(v_2i):  # u is v_(2i+1)
                if u in v:  # disjunctivity criterion
                    continue
                mu['break_vs'].append(u)
                # check feasibility immediately
                delta_gain = self.cost[v_2i, u] - self.cost[u, v[0]]
                if mu['gain'] + delta_gain > 0:
                    result = tour.check_feasible(mu['break_vs'])
                    if result is not None:
                        return result
                # This line is reached if closing the break vertices is either unprofitable or infeasible
                if i == self.max_exchange - 1:
                    # maximum exchange is reached, so if the current path is infeasible then fruitless.
                    mu['break_vs'].pop()
                    continue
                mu['i'] += 1
                result = self.dfs_recursion(tour, mu)
                if result is not None:
                    return result
                mu['break_vs'].pop()
                mu['i'] -= 1
            # If this line is reached, then v_2i is fruitless, hence backtrack
            mu['break_vs'].pop()  # i.e break_vs.remove(v_2i)
            mu['gain'] -= self.cost[v[2 * i - 2], v[2 * i - 1]] - self.cost[v[2 * i - 1], v_2i]
        # If this line is reached, then v_(2i-1) is fruitless

    def local_optimum(self, tour: Tour, candidates):
        """improve an initial tour by variable-exchange until local optimum."""
        better = True
        while better:
            tour = self.improve(tour, candidates)
            better = tour is not None
            return tour

    def main(cost_mat):
        """For now, we start from the cost matrix (numpy array).
        todo: more input including capacity, time window in the future.
        1) Transform the cost matrix to "cluster" the vertices adjacent in the optimal tour.
        Return a more suitable cost matrix and the corresponding minimum 1-tree
            Initialize the penalty_vec = 0.
            Loop until stopping criterion:
                a) find the minimum 1-tree of the graph defined by the cost matrix.
                b) Compute the degree for each vertex in the minimum 1-tree, d_vec.
                c) penalty_vec = penalty_vec + step_size * (d_vec - 2) or the equivalent RMSprop style update.
                d) cost_mat[i, j] = cost_mat[i, j] + penalty_vec[i] + penalty_vec[j]
        2) Compute the alpha-nearest value based on the transformed cost matrix,
        probably only need to retain the k alpha-nearest vertices for each vertex, where k in range(5, 20).
        3) Initialize a solution tour.
        4) LK style improvement on the solution tour until local optimum
        """
        cost_mat0 = cost_mat
        cost_mat = cost_mat.copy()
        pass
