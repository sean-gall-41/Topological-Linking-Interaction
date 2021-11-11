
import numpy as np

from .private.utilities import pre_alexander_compile


def populate_alexander_matrix(saw, proj, t):
   """return alexander matrix of the given saw.
   
   We are using the logic from the guiding paper for this project (soon to 
   be in the README!) in order to return the alexander matrix of the given
   saw.
   
   arguments:
   saw - numpy array with shape (N, 3) - the SAW whose alexander matrix we wish
   to calculate
   proj - numpy array with shape (N, 2) - the regular projection of saw
   t - float - the polynomial parameter
   
   return value:
   alex_mat - numpy array with shape (I, I) where I is the number of underpasses.
   """
   underpass_info = pre_alexander_compile(saw, proj)
   I = np.shape(underpass_info)[0]
   alex_mat = np.zeros((I, I))
   
   for k in np.arange(I):
      if underpass_info[k, 1] == k or underpass_info[k, 1] == k+1:
         alex_mat[k, k] = -1
         if k == I-1:
            continue
         else: alex_mat[k, k+1] = 1
      else:
         alex_mat[k, underpass_info[k, 1]] = t - 1
         if underpass_info[k, 0] == 0: # Type I
            alex_mat[k, k] = 1
            if k == I-1:
               continue
            else: alex_mat[k, k+1] = -t
         else: # Type II
            alex_mat[k, k] = -t
            if k == I-1:
               continue
            else: alex_mat[k, k+1] = 1

   return alex_mat


from numpy.linalg import det

from .projection import find_reg_project


def evaluate_alexander_polynomial(alex_mat):
   if np.shape(alex_mat)[0] == 0:
      return 1
   # TODO: focus on edge cases
   minor = np.delete(alex_mat, 0, 0) # delete first row
   minor = np.delete(minor, 0, 1) # delete first colum
   minor_det = det(minor)
   # TODO: figure out power of t we need...or better work around
   final_result = int(np.round(abs(minor_det)))
   return final_result
   # if np.shape(alex_mat)[0] == 0:
   #    print("something is wrong with this chain, or degenerate")
   #    return None
   # elif np.shape(alex_mat)[0] == 1:
   #    return int(np.round(abs(alex_mat[0])))
   # else:
   #    minor = np.delete(alex_mat, 0, 0) # delete first row
   #    minor = np.delete(minor, 0, 1) # delete first colum
   #    minor_det = det(minor)
   #    # TODO: figure out power of t we need...or better work around
   #    final_result = int(np.round(abs(minor_det)))
   #    return None
