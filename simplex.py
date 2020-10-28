"""MIT License

Copyright (c) 2020 kevalii

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np


def solve(cost, constraints, x):
    """Solves a standard form linear program given an initial basic feasible solution

    Parameters
        cost: The N-dimensional cost vector
        constraints: The matrix [A b], representing the set of constraints Ax = b
        x: The initial basic feasible solution

    Returns
        The optimal solution to the linear program if it exists, otherwise -np.inf
    """
    M, N = constraints.shape

    A = constraints[:, :-1]
    b = constraints[:, -1]

    # Validate the program and the initial solution
    assert M <= N, "program is infeasible"
    assert (x == 0).sum() != N - M, f"given bfs is not basic!"
    assert (x >= 0).all(), f"given bfs is not feasible!"
    assert (A @ x == b).all(), f"given bfs is not a solution!"

    # Initialize the indices of basic and nonbasic variables
    basis = set(np.flatnonzero(x))

    while (True):
        basis_idxs = list(basis)
        B_inv = np.linalg.inv(A[:, basis_idxs])

        # Calculate reduced costs for all variables
        reduced_costs = np.array(
            [cost[j] - cost[basis_idxs].T @ B_inv @ A[:, j]
                for j in range(N - 1)]
        )
        nonnegative_mask = reduced_costs >= 0

        # Try to find lowest nonbasic index j such that reduced_costs[j] < 0
        # The choice of the lowest is to avoid cycling
        maybe_j = np.where(nonnegative_mask == False)[0]
        if maybe_j.size == 0:
            # No such j exists; then all reduced costs are nonnegative and current solution is optimal
            return x
        j = maybe_j[0]

        # Compute the feasible direction
        u = B_inv @ A[:, j]
        scalars = [(l, x[l] / u[i]) for (i, l) in enumerate(basis) if u[i] > 0]

        # If no component of the feasible direction is positive, optimal cost is
        # negative infinity
        if (not scalars):
            return -np.inf
        l, theta = min(scalars, key=lambda x: x[1])

        # Set new feasible solution
        x[basis_idxs] = x[basis_idxs] - theta * u
        x[j] = theta

        # jth variable enters the basis, lth exits
        basis.add(j)
        basis.remove(l)


if __name__ == '__main__':
    coefficients = np.array([
        [1, 1, 1, 1, 0, 0, 0, 4],
        [1, 0, 0, 0, 1, 0, 0, 2],
        [0, 0, 1, 0, 0, 1, 0, 3],
        [0, 3, 1, 0, 0, 0, 1, 6],
    ])
    cost = np.array([1, 5, -2, 0, 0, 0, 0])
    initial_solution = np.array([2, 0, 2, 0, 0, 1, 4])
    result = solve(cost=cost,
                   constraints=coefficients,
                   x=initial_solution)
    print(f"optimal solution: {result}, optimal cost: {cost @ result}")
