# Linear Programming: An Introduction

This repository contains the Latex source and compiled PDF for the project, as well as a supplementary Python program implementing the simplex method for standard form linear programs given an initial basic feasible solution.

## Running the program

I tested `simplex.py` with CPython 3.7, but any recent and reasonable Python interpreter should work as well.

1. Install the necessary dependencies: `pip install -r requirements.txt`
2. Run `python simplex.py`
3. Play around with the input LPs as you see fit. The program verifies that the given program is feasible and the given initial solution is a bfs before attempting to find an optimal solution.
