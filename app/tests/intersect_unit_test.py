import numpy as np

from ..private.utilities import find_intersection_2D


def intersect_unit_test():
    """Return None. Test entire find_intersection_2D function"""
    A = np.array([2.,0.])
    B = np.array([2., 2.])
    C = np.array([0., 2.])
    D = np.array([0., 0.])

    intersection = find_intersection_2D(A, B, C, D)

    if intersection is None:
        print("no intersection or intersection not in segment")
    else:
        print("intersection at ({x}, {y})".format(x=intersection[0], 
              y=intersection[1]))
