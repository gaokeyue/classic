from __future__ import annotations

from .tour import Tour


def lk_step(tour: Tour)->bool:
    pass




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