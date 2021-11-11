import json

import numpy as np

from ..alexander import (evaluate_alexander_polynomial,
                         populate_alexander_matrix)
from ..private.utilities import pre_alexander_compile, rot_saw_xy
from ..projection import find_reg_project

# TODO: change from absolute path (rel to project root) to variable
TEST_CASE_N_18 = "app/tests/test_knots_N_18.json"

def populate_alexander_matrix_unit_test():
    with open(TEST_CASE_N_18) as ifile:
        print("Loading test data from file {}...".format(TEST_CASE_N_18))
        in_data = json.load(ifile)
        test_chains = in_data["tests"]
        num_chains = len(test_chains)
        # validation_list = in_data["validation_list"]
        print("Finished loading test data.")
        print("Printing {} Alexander Polynomials from loaded test data.".format(num_chains))

        for chain in test_chains:
            alex_mat = populate_alexander_matrix(rot_saw_xy(chain),
                                                      find_reg_project(chain),
                                                      -1)
            alex_poly = evaluate_alexander_polynomial(alex_mat)
            if alex_poly == 1:
                print(chain, end='\n')
                print(alex_mat, end='\n\n')

        print("Finished Printing {} Alexander Polynomials from loaded test data.".format(num_chains))
