# Collection of all error models in the QECC cycle, such as time-dependent errors and noisy cz gates.


import numpy as np
import scipy.stats
import qutip as qt
from mpmath import *
import pdb
from QECC_functions import *

def apply_gate(rho,gate,sel):
    # Apply the operation specified by 'gate' to the selection `sel' of qubits of state rho
    dims = rho.dims
    n = int(np.log2(np.sqrt(np.prod(rho.dims))))# get number of qubits
    if 1 in sel:
        total_gate = gate
    else:
        total_gate = qt.identity(2)

    for ii in range(2,n+1):
        if ii in sel:
            total_gate = qt.tensor(total_gate,gate)
        else:
            total_gate = qt.tensor(total_gate,qt.identity(2))

    rho = total_gate*rho*total_gate.dag()
    rho.dims = dims
    return rho

def amp_damping_single_qubit(rho,gamma,c):
    # apply amp-damping to c'th qubit with parameter gamma of state rho

    # Create Kraus operators
    A_0 = qt.Qobj([[1,                0],
                   [0, np.sqrt(1-gamma)]])
    A_1 = qt.Qobj([[0, np.sqrt(gamma)],
                   [0,              0]])

    # Collect Kraus operators
    kraus_terms = [A_0,A_1]

    # Apply Kraus operators to state rho    
    rho_new = []
    for ii in range(0,len(kraus_terms)):
        rho_new.append(apply_gate(rho,kraus_terms[ii],[c]))

    return sum(rho_new)

def depolarizing_single_qubit(rho,p,c):
    # apply depolarizing to c'th qubit c with parameter p of state rho

    # Create Kraus operators
    A_0 = np.sqrt((1-3*p/4))*qt.identity(2)
    A_1 = np.sqrt(p/4)*qt.Qobj([[0,1],
                 [1,0]])
    A_2 = np.sqrt(p/4)*qt.Qobj([[0,-1j],
                 [1j,0]])
    A_3 = np.sqrt(p/4)*qt.Qobj([[1,0],
                 [0,-1]])

    # Collect Kraus operators
    kraus_terms = [A_0,A_1,A_2,A_3]

    # Apply Kraus operators to state rho
    rho_new = []
    for ii in range(0,len(kraus_terms)):
        rho_new.append(apply_gate(rho,kraus_terms[ii],[c]))

    return sum(rho_new)

def dephasing_single_qubit(rho,p,c):
    # apply dephasing to c'th qubit with parameter p of state rho

    # Create Kraus operators
    A_0 = np.sqrt(1-p)*qt.identity(2)
    A_1 = np.sqrt(p)*qt.Qobj([[1,0],
                                [0,-1]])
    # Collect Kraus operators
    kraus_terms = [A_0,A_1]

    # Apply Kraus operators to state rho
    rho_new = []
    for ii in range(0,len(kraus_terms)):
        rho_new.append(apply_gate(rho,kraus_terms[ii],[c]))

    return sum(rho_new)

def create_noisy_cz(E):
    # create noisy controlled-z gate, as found in Physical Review A, 88(1):012314 and Phys. Rev. A, 87(2):022309
    E1 = (5/4)*E
    phi = 0.0
    delta = np.sqrt((10/3)*E)
    noisy_cz = qt.Qobj([[1, 0,                            0,                          0],
                        [0, np.sqrt(1-E1),                 np.sqrt(E1)*np.exp(1j*phi),  0],
                        [0, -np.sqrt(E1)*np.exp(-1j*phi),  np.sqrt(1-E1),               0],
                        [0, 0,                            0,          -np.exp(delta*1j)]])
    return noisy_cz