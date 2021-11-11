import json

import numpy as np

# ============================ CHAIN UTILITIES ============================= #

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self.obj)


# expected input: numpy ndarray
def chain_to_JSON(chain, file_dumps=False):
    data = {"vertices": chain}

    if file_dumps:
        filename = "vertex_array.json"
        print("Serializing NumPy array into {}...".format(filename))
        # write to file 'chain.json'
        with open(filename, "w") as ofile:
            data_to_str = json.dumps(data)
            json.dump(data_to_str, ofile, cls=NumpyArrayEncoder)
        print("Done writing serialized NumPy array into {}.".format(filename))
        return ""

    else:
        print("Dumping NumPy Array into JSON string...")
        return json.dumps(data, cls=NumpyArrayEncoder)

# ========================= PROJECTION UTILITIES =========================== #

def rot_matrix_x(alpha):
    """return rotation matrix about x-axis by angle alpha"""
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(alpha), -np.sin(alpha)],
            [0.0, np.sin(alpha), np.cos(alpha)],
        ]
    )


def rot_matrix_y(alpha):
    """return rotation matrix about Y-axis by angle alpha"""
    return np.array(
        [
            [np.cos(alpha), 0.0, np.sin(alpha)],
            [0.0, 1.0, 0.0],
            [-np.sin(alpha), 0.0, np.cos(alpha)],
        ]
    )


def rot_saw_xy(saw):
    """return rotated saw by angles alpha and beta.

    prepares a saw so that it may be projected into the xy plane via art setting 
    z-comp of every vertex to zero. We don't actually do that here, but instead,
    rotate the saw so that such a projection will be regular.
    
    parameters:
    saw - numpy array of shape (N, 3) - the SAW which we will rotate

    return value:
    rotated_saw numpy array of same shape as saw - the rotated saw
    """
    alpha = -np.pi / 3.0
    beta = np.pi / 6.0
    x_rot = rot_matrix_x(alpha)
    y_rot = rot_matrix_y(beta)
    rotated_saw = []
    for vertex in saw:
        rot_vertex = np.matmul(x_rot, vertex)
        rot_vertex = np.matmul(y_rot, vertex)
        rotated_saw.append(rot_vertex)
    return np.array(rotated_saw)

# ========================== ALEXANDER UTILITIES ============================ #

from numpy.linalg import LinAlgError, norm

EPS = 1.0e-12 # convenient small value for double comparison

def validate_intersect_in_segments(p0, p1, p2, p3, p4):
   """return True if p0 in bounding box of p1p2 AND p3p4, else return False."""
   in_line_1 = min(p1[0], p2[0]) < p0[0] < max(p1[0], p2[0]) and \
          min(p1[1], p2[1]) < p0[1] < max(p1[1], p2[1])
   in_line_2 = min(p3[0], p4[0]) < p0[0] < max(p3[0], p4[0]) and \
          min(p3[1], p4[1]) < p0[1] < max(p3[1], p4[1])
   
   return True if (in_line_1 and in_line_2) else False


# algorithm from: https://www.geeksforgeeks.org/program-for-point-of-intersection-of-two-lines/
def find_intersection_2D(p1, p2, p3, p4):
   """from line segments p1p2 and p3p4, return the intersection if it exists.
   
   The pair of lines is modeled as a linear system of the following form:

      a1*x + b1*y = c1
      a2*x + b2*y = c2

   and is solved via Cramer's Rule

   Here we also validate whether the intersection, if so found, exists within
   both line segments or not.
   
   arguments:
   p1 - numpy array of doubles with shape (2,1) - first point of line segment p1p2
   p2 - numpy array of doubles with shape (2,1) -second point of line segment p1p2
   p3 - numpy array of doubles with shape (2,1) - first point of line segment p3p4
   p4 - numpy array of doubles with shape (2,1) - second point of line segment p3p4
   
   return value(s):
   intersect - numpy array of doubles with shape (2,1) - x is the x-component 
   of the intersection and y is the y-component
   None - if the intersection does not exist or does not lie in segment p1p2
   """
   a1 = p2[1] - p1[1]
   b1 = p1[0] - p2[0]
   c1 = a1*p1[0] + b1*p1[1]

   a2 = p4[1] - p3[1]
   b2 = p3[0] - p4[0]
   c2 = a2*p3[0] + b2*p3[1]

   det = a1*b2 - a2*b1
   if det == 0: # lines are parallel, no intersection
      return None
   else: # Cramer's Rule
      x = (c1*b2 - c2*b1) / det
      y = (a1*c2 - a2*c1) / det

      intersect = np.array([x, y])

      if validate_intersect_in_segments(intersect, p1, p2, p3, p4):
         return intersect
      else: return None # not validating here... do in is_underpass


def find_intersection_2D_vec(p1, p2, p3, p4):
   """from line segments p1p2 and p3p4, return the intersection if it exists.
   
   Each line segment may be represented by the vector eqns:

   x_1 = u_1 - v_1*t
   x_2 = u_2 - v_2*s

   where each underscored variable is a 3-vector, and t and s are real numbers
   lying in some closed interval of the real line such that for min(t) and min(s)
   x_1 = p1 and x_2 = p3 and likewise for the maxima. 

   The point of using this method is so that we return parameters for each
   intersection which gives us information about how far from the starting 
   point the intersection takes place. This information will be used in the 
   case when a single line segment has intersections with multiple other line
   segments. 

   arguments:
   p1 - numpy array of doubles with shape (2,1) - first point of line segment p1p2
   p2 - numpy array of doubles with shape (2,1) -second point of line segment p1p2
   p3 - numpy array of doubles with shape (2,1) - first point of line segment p3p4
   p4 - numpy array of doubles with shape (2,1) - second point of line segment p3p4

   return value(s):
   None - either the linear system is inconsistent or the intersection does not lie
   in both segments' bounding boxes
   np.array([intersect, x[0]]) - a numpy array whose first entry is the coordinates
   of the intersection and the second is the parameter associated with line segment
   p1p2, which serves as the "reference line segment"
   """
   # create 3-vector to solve linear system A*x = b
   b_1 = p1[0] - p3[0]
   b_2 = p1[1] - p3[1]
   b = np.array([b_1,b_2])

   # collect tangent vectors
   v_1 = np.subtract(p2,p1)
   v_2 = np.subtract(p4,p3)

   v_1_normalized = v_1 / norm(v_1)
   v_2_normalized = v_2 / norm(v_2)

   # if the tangent vectors are parallel, impossible for there to be an
   # intersection, given our projection is regular
   if np.array_equal(v_1_normalized, v_2_normalized):
      return None
   
   A = np.array([[-v_1[0], v_2[0]],[-v_1[1], v_2[1]]])
   x = np.empty((2,1))

   try:
      x = np.linalg.solve(A, b)
   except LinAlgError: # A is singular
      return None
   finally:
      intersect = find_intersection_2D(p1, p2, p3, p4)
      if intersect is None: return None # case when have intersect, but no in seg
      else:
         dtype = [('coords', np.float64, (2,)), ('order_param', np.float64)]
         return np.array([(intersect.tolist(), x[0])], dtype=dtype) # return both intersect and param values


def is_underpass(k, j, intersect, saw):
   """return True if segment pkpk_1 is an underpass of pjpj_1, else return False.
   
   This method uses the eqns for a line in 3D found
   here: https://byjus.com/maths/equation-line/ to calculate the z-value
   of the corresponding point in the SAW to determine whether pkpk_1 is
   an underpass or not.
   
   arguments:
   k - int - the index of node pk within the SAW
   j - int the index of node pj within the SAW
   intersect - numpy array with shape (2,1) - the intersection within the 
   projection, with format (x, y)
   saw - numpy array with shape (N, 3) - the SAW where underpasses will be 
   found from
   
   return value:
   boolean - True if pkpk_1 is an underpass, else False
   """
   pk = saw[k]
   pk_1 = saw[k+1]
   pj = saw[j]
   pj_1 = saw[j+1]

   zk = pk[2] + (pk_1[2] - pk[2])*(intersect[0] - pk[0]) / (pk_1[0] - pk[0])
   zj = pj[2] + (pj_1[2] - pj[2])*(intersect[0] - pj[0]) / (pj_1[0] - pj[0])
   return True if (zj - zk > EPS) else False


def order_intersections(intersections):
   """returns a sorted list of intersections according to projection orientation.
   
   Here we assume intersections is a structured numpy array.
   
   arguments:
   intersections - a list of lists, containing the intersection coords and indices, and 
   ordering parameter for each segment the first time it is encountered as an intersection
   
   return value:
   the sorted version of intersections
   """
   # create temp array to be used as the sorting key
   temp = np.zeros(intersections.shape, dtype=[('f0',np.uintc,(2,)),('f1',np.float64)])
   # get first two columns of the indices
   temp['f0'] = intersections['indices'][:,:2]
   temp['f1'] = intersections['order_param']
   # this step is literally magic as far as I know
   return intersections[np.argsort(temp)]


def collect_all_intersections(proj):
   """return structured array of intersection coords and surrounding indices.
   
   Important to note is that this function collects intersections twice, so
   the returned array will always have even shape along the first axis.

   We run a function order_intersections() at the end as our naive alg may
   collect intersections out of order according to the projection's orientation.

   argument:
   proj - numpy array with shape (N, 2) - the regular projection of our saw

   return value:
   return_val - numpy array with shape (I, 3) - the structured array which
   contains the intersection coordinates, the encapsulating indices, and
   the 'order parameter' as explained in find_intersection_2D_vec
   """

   intersections = []
   for k in np.arange(proj.shape[0]-1):
      for j in np.arange(proj.shape[0]-1):
         if j == k-1 or j == k or j == k+1:
            continue
         pk = proj[k][:2]
         pk_1 = proj[k+1][:2]
         pj = proj[j][:2]
         pj_1 = proj[j+1][:2]
         intersection = find_intersection_2D_vec(pk, pk_1, pj, pj_1)
         if intersection is not None:
            this_intersection = (intersection['coords'].tolist()[0], [k,k+1,j,j+1], intersection['order_param'][0])
            intersections.append(this_intersection)
   # dtype to be used for the structured array
   dtype = [('coords', np.float64, (2,)), ('indices', np.uintc, (4,)), ('order_param', np.float64)]
   intersections_as_array = np.array(intersections, dtype=dtype)
   return_val = order_intersections(intersections_as_array)
   return return_val


def get_underpass_indices(proj, saw):
   """return array of underpass indices, in order of occurence.
   
   Returns the 'encapsulating' indices of the nodes surrounding a given
   intersection, as shown below:
   
                       [k]
                     |  |
                     V  |
               [j] -----+----- [j+1]
                        | --->
                        |
                      [k+1]
   
   with the order being [k, k+1, j, j+1].

   parameters:
   saw - numpy array with shape (N, 3) - the SAW we obtain underpasses from
   proj - numpy array with shape (N, 2) - the regular projection of saw

   return value:
   underpass_indices - numpy array with shape (I / 2, 4) - array of encapsulating
   indices of all underpasses, in order of occurence
   """
   intersections = collect_all_intersections(proj)
   intersection_indices = intersections["indices"]
   intersection_coords = intersections["coords"]
   num_intersections = np.shape(intersections)[0]
   num_underpasses = num_intersections // 2

   underpass_indices = np.empty((num_underpasses,4), dtype=np.uintc)
   j = 0

   for i in np.arange(num_intersections):
      if is_underpass(intersection_indices[i,0], intersection_indices[i,2],
                      intersection_coords[i], saw):
         underpass_indices[j] = intersection_indices[i]
         j += 1

   return underpass_indices


def assign_underpass_types(underpasses, proj, underpass_info):
   """return None, only modify elements of underpass_info by assigning underpass type."""
   for l in np.arange(np.shape(underpasses)[0]):
      # collect underpass type
      pk = proj[underpasses[l][1]][:2]
      pk_1 = proj[underpasses[l][2]][:2]
      pj = proj[underpasses[l][3]][:2]
      v1 = np.subtract(pk, pj)
      v2 = np.subtract(pk, pk_1)
      if np.cross(v1, v2) == 1: # Type I
         underpass_info[l,0] = 0
      else: # Type II
         underpass_info[l,0] = 1


def assign_generator_to_underpasses(underpass_indices, intersection_indices,
                                    intersection_coords, underpass_info, saw):
   """return None, modify elements of underpass_info by assigning overpass generators"""
   #using indexes is just..easier
   for i in np.arange(np.shape(underpass_indices)[0]):
      # below finds the index of the current underpass within intersections, then decrements
      j = np.nonzero(np.all((intersection_indices-underpass_indices[i])==0,axis=1))[0][0] - 1
      while True:
         if j < 0:
            j = np.shape(intersection_indices)[0] - 1
         if not is_underpass(intersection_indices[j,0], intersection_indices[j,2],
                             intersection_coords[j], saw):

            underpass_k = np.roll(intersection_indices[j], 2)
            k = np.nonzero(np.all((underpass_indices-underpass_k)==0,axis=1))[0][0]
            underpass_info[k, 1] = i
            j -= 1
         else:
            break


def pre_alexander_compile(saw, proj):
   """return a list of underpass info, including underpass type and generator.
   
   Collects two pieces of information: the types of each underpass (either type
   I or type II, referring to the 'handedness' of the underpass) and the 0'd
   index of the i-th generator for the kth underpass. 
   
   arguments:
   saw - numpy array with shape (N, 3) - the saw which we are running analysis on
   proj - numpy array with shape (N, 2) - the regular projection of saw
   
   return value:
   underpass_info - a numpy array with shape (I, 2) where I is the number of 
   underpasses - the relevant info which will be used to populate alexander matrix
   """
   intersections = collect_all_intersections(proj)
   underpass_indices = get_underpass_indices(proj, saw)

   underpass_info = np.zeros((np.shape(underpass_indices)[0],2),dtype=np.intc) 

   assign_underpass_types(underpass_indices, proj, underpass_info)
   assign_generator_to_underpasses(underpass_indices, intersections["indices"],
                                   intersections["coords"], underpass_info, saw)

   return underpass_info

# ============================== TEST UTILITIES ============================= #

from pathlib import Path

JSON_EXTENSION = ".json"

#https://faculty.washington.edu/cemann/S0218216509007373.pdf
def code_to_chain(code):
   """From codes at above website return json array of code chain rep."""
   directions = np.array([(1,1,1), (-1,-1,-1), (-1,1,1), (1,-1,-1), (-1,-1,1),
                          (1,1,-1), (1,-1,1), (-1,1,-1)])
   chain = np.zeros((len(code)+1,3))
   chain_index = 1
   for number in code:
      previous_node = chain[chain_index-1]
      current_node = np.add(previous_node, directions[int(number)])
      chain[chain_index] = current_node #broadcasting ?
      previous_node = current_node
      chain_index += 1

   return chain
  

def construct_json_validation_from_file(file, validation_list=[]):
   with open(file) as f:
      print("Loading test data from file {}...".format(file))
      file_contents = f.readlines()
      codes, chains = [], []
      for line in file_contents:
         codes = codes + line.split()
      for code in codes:
         chains.append(code_to_chain(code))
      chains = np.array(chains)
      info_as_dict = {"tests": chains.tolist(), "num_segments" : chains.shape[1]-1,
                  "validation_list": validation_list}
      
      out_file = Path(file).stem + JSON_EXTENSION
      with open(out_file, 'w') as json_file:
         json.dump(info_as_dict, json_file)

