import numpy as np

def gen_all_bin_list(N):
    """return numpy array of all binary lists of length N."""
    all_poss = []
    poss = np.empty(N)
    gen_bin_list_helper(N, all_poss, poss, 0)
    return np.array(all_poss)


def gen_bin_list_helper(N, all_poss, poss, i):
    """populates array all_poss with all binary lists of length N
    
    This function implements the recursive algorithm from here:
    https://www.geeksforgeeks.org/generate-all-the-binary-strings-of-n-bits/.

    arguments:
    N - int - the length of each binary list
    all_poss - list - the list to be populated with binary lists
    poss - numpy array - array of possible binary values
    i - int - an index indicating which position in each list we should populate

    return value:
    None
    """
    if i == N:
        # make the necessary transformation (0, 1) -> (-1, 1) because
        # we are working with unit random walks
        poss = 2 * poss - 1
        # once we've reached the end of this possibility, add it to all
        # possible list
        all_poss.append(poss)
        return

    # compute all permutations with a 0 in the ith position
    poss[i] = 0
    gen_bin_list_helper(N, all_poss, poss, i + 1)

    # compute all permutations with a 1 in the ith postion
    poss[i] = 1
    gen_bin_list_helper(N, all_poss, poss, i + 1)
