import os

import matplotlib.pyplot as plt
import numpy as np

from .alexander import evaluate_alexander_polynomial, populate_alexander_matrix
from .generate_chain import generate_closed_chain
from .private.utilities import rot_saw_xy
from .projection import find_reg_project


def basic_monte_carlo_sim(num_nodes, num_chains, table=True, shift=False):
    """return raw_data. Print table and final statistics for monte carlo sim.
    
    We are concerned with the distributions of knot formation as well as
    number of attempts."""
    raw_data = np.zeros((num_chains,2)) # first element is result of is_knotted, second is num attempts
    # if table:
    #     print("+-----------------+---  MONTE CARLO SIMULATION ---+----------------+")
    #     print("|    CHAIN ID     | ALEXANDER POLYNOMIAL (t = -1) |   IS A KNOT?   |")
    #     print("+-----------------+-------------------------------+----------------+")
    
    # run the simulation
    for i in np.arange(num_chains):
        chain, num_attempts = generate_closed_chain(num_nodes, shift)
        print(chain)
        alex_mat = populate_alexander_matrix(rot_saw_xy(chain)[:-1], find_reg_project(chain)[:-1], -1)
        alex_poly = evaluate_alexander_polynomial(alex_mat)
        is_knotted = not (alex_poly == 1)
        if i % 100 == 0:
            print("chain {}".format(i))
        # if table:
            # print("|{:^17}|{:^31}|{:^16}|".format(i+1, alex_poly, is_knotted))
            # print("+-----------------+-------------------------------+----------------+")

        raw_data[i][0] = is_knotted
        raw_data[i][1] = num_attempts

    # analyze the results
    total_knots = len(np.where(raw_data[:,0])[0])
    attempt_stats = np.array([np.mean(raw_data[:,1]), np.std(raw_data[:,1])])
    print("Total knots: {}".format(total_knots))
    print("Mean of number of attempts was {}".format(attempt_stats[0]))
    print("Std dev of number of attempts was {}".format(attempt_stats[1]))

    return raw_data


def cum_monte_carlo_sim():
    """return None. run many monte Carlo simulations with various conditions.
    
    This will take a while (11/07/2021)
    
    """

    simulation_conditions = np.array([[20, 12560], [30, 5950], [40, 3000], [50, 2500], 
        [60, 3450], [70, 1460], [80, 1340], [90, 1120], [100, 910], [110, 400]
        [120, 415], [130, 520], [140, 420]])
    
    probs_of_knot_formation = np.zeros(np.shape(simulation_conditions)[0])
    for condition in simulation_conditions:
        this_run_data = basic_monte_carlo_sim(condition[0], condition[1], table=False)
        # TODO: for now, save png's of num attempts statistics in image folder,
        #       calculate mean and std dev of that dist, and prob of knot formation
    return None
