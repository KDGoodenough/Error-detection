import numpy as np
import scipy.stats
import qutip as qt
from mpmath import *
import pdb

def permute_matrix(c,t,matrix,rho):
    # For a given two-qubit gate in the form, find the matrix of dimension corresponding to rho such that 
    # the two-qubit gate operations on c and t. e.g., permute_matrix(4,1,cnot,rho) will apply a cnot gate
    # between qubits 4 and 1, with 4 the control and 1 the target

    # Get shape and number of qubits
    shape = rho.shape
    n = int(np.log2(shape[0]))

    # Create list in binary from 0 to 2^n-1, corresponding to the orthonormal computational basis
    bitstr_list = []
    for ii in range(0,np.power(2,n)):
        bitstr = bin(ii)[2:].zfill(n)
        bitstr = list(bitstr)
        bitstr_list.append(bitstr)

    # Now swap all entries of the binary list according to the given c and t
    # Notice that we have to be careful of the order we do this in, due to noncommutativity of permutations
    # That's why we have the overabundance of if statements.

    if c != 2 and t != 1:
        for ii in range(0,np.power(2,n)):
            swap1 = bitstr_list[ii][0]
            swap2 = bitstr_list[ii][c-1]
            bitstr_list[ii][0] = swap2
            bitstr_list[ii][c-1] = swap1

        for ii in range(0,np.power(2,n)):
            swap1 = bitstr_list[ii][1]
            swap2 = bitstr_list[ii][t-1]
            bitstr_list[ii][1] = swap2
            bitstr_list[ii][t-1] = swap1
    elif c == 2 and t == 1:
        for ii in range(0,np.power(2,n)):
            swap1 = bitstr_list[ii][0]
            swap2 = bitstr_list[ii][c-1]
            bitstr_list[ii][0] = swap2
            bitstr_list[ii][c-1] = swap1
    elif c == 2 and t != 1:
        for ii in range(0,np.power(2,n)):
            swap1 = bitstr_list[ii][1]
            swap2 = bitstr_list[ii][t-1]
            bitstr_list[ii][1] = swap2
            bitstr_list[ii][t-1] = swap1
        for ii in range(0,np.power(2,n)):
            swap1 = bitstr_list[ii][0]
            swap2 = bitstr_list[ii][c-1]
            bitstr_list[ii][0] = swap2
            bitstr_list[ii][c-1] = swap1
    elif c != 2 and t == 1:
        for ii in range(0,np.power(2,n)):
            swap1 = bitstr_list[ii][0]
            swap2 = bitstr_list[ii][c-1]
            bitstr_list[ii][0] = swap2
            bitstr_list[ii][c-1] = swap1
        for ii in range(0,np.power(2,n)):
            swap1 = bitstr_list[ii][1]
            swap2 = bitstr_list[ii][t-1]
            bitstr_list[ii][1] = swap2
            bitstr_list[ii][t-1] = swap1

    list_of_permutations = np.zeros(np.power(2,n))
    for ii in range(0,np.power(2,n)):
        list_of_permutations[ii] = int(''.join(bitstr_list[ii]),2)

    permutation_matrix = np.zeros((np.power(2,n),np.power(2,n)))
    for ii in range(0,np.power(2,n)):
        for iii in range(0,np.power(2,n)):
            if ii == list_of_permutations[iii]:
                permutation_matrix[ii][iii] = 1

    permutation_matrix = qt.Qobj(permutation_matrix)
    permutation_matrix.dims = rho.dims

    for ii in range(0,n-2):
      matrix = qt.tensor(matrix,qt.identity(2))

    matrix.dims = rho.dims
    # permute matrix
    permuted_matrix = permutation_matrix*matrix*permutation_matrix.trans()
    rho = permuted_matrix*rho*permuted_matrix.dag()
    return rho