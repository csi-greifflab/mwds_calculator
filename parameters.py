# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 15:28:23 2021

@author: Jahn Zhong
"""

# Integer parameters
population_size = 50    # Size of initial solution_list (width)
max_trials = 25         # Depth of exploration of each individual solution
max_iterations = 30     # Iteration number for entire algorithm
onlooker_bees = 20      # Number of bees randomly choosing a solution to explore further
d = 5                   # Number to determine best n solutions (best_n = population_size/d)
best_n = int(round(population_size/d))
# Probability parameters
beta = 0.2              # Aggresiveness of solution mutations
lmbda = 0.6             # Aggresiveness of probability_vector updates
mu1 = 0.5               # Probability of selecting a new completely random solution during scout bee phase
mu2 = 0.5               # Probability of including each node from the best solution in a new solution during scout bee phase
small_value = 0.05      # Initial non-zero value for probability_vector