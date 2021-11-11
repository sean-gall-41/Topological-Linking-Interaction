import numpy as np

from .private.utilities import rot_saw_xy


def find_reg_project(saw):
    """return regular projection of SAW via rotation by irrational angle.
    
    I like this one. Finds a regular projection of the SAW by rotating 
    the axes CCW (or conversely, by rotating the SAW CW) by an irrational
    angle, thus guaranteeing that no multiple points are triple, and that
    no two vertices are projected to the same point (vertices are at most
    single points).

    (Note that the angle is completely arbitrary, so I chose PI/3)
    (CHANGELOG: added second y-rotation by pi / 6 on 10/04/2021)
    
    argument:
    saw - numpy array of shape (N, 3) - the SAW which we wish to find a com
    
    return value:
    projection - numpy array of shape (N, 2) - the projection of the saw
    """
    rotated_saw = rot_saw_xy(saw)
    projection = []
    for vertex in rotated_saw:
        project_vertex = vertex
        project_vertex[2] = 0.0 # project to xy plane :)
        projection.append(project_vertex)

    projection = np.array(projection)

    return projection
