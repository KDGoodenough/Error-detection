# Create two mode squeezed vacuum state with variance V
import numpy as np
import scipy.stats
import qutip as qt
import matplotlib.pyplot as plt
import sys
from ErrorModels import *
from mpmath import *
import pdb

def create_projectors():
    # Create all the different projectors that are required for the measurement
    I = qt.identity(2)
    zero = qt.basis(2,0)
    one = qt.basis(2,1)

    proj1 = qt.Qobj([[1,0], # single qubit projector unto 0 subspace
                     [0,0]])
    proj2 = qt.Qobj([[0,0], # single qubit projector unto 1 subspace
                     [0,1]])

    # Create projectors for all measurement steps
    projector1 = qt.tensor(I,I,I,I,proj1,I) # Projector unto 0 for first measurement
    projector2 = qt.tensor(I,I,I,I,proj2,I) # Projector unto 1 for first measurement
    projector3 = qt.tensor(I,I,I,I,I,proj1) # Projector unto 0 for second measurement
    projector4 = qt.tensor(I,I,I,I,I,proj2) # Projector unto 1 for second measurement
    
    projector5 = qt.tensor(proj1,I,I,I) # Projector unto 0 for third measurement
    projector6 = qt.tensor(proj2,I,I,I) # Projector unto 1 for third measurement
    projector7 = qt.tensor(I,I,proj1,I) # Projector unto 0 for fourth measurement
    projector8 = qt.tensor(I,I,proj2,I) # Projector unto 1 for fourth measurement
    
    # Create projector unto codespace
    logicalzero = qt.ket2dm(qt.tensor(zero,zero,zero,zero)+qt.tensor(one,one,one,one)).unit()
    logicalone  = qt.ket2dm(qt.tensor(zero,zero,one,one)+qt.tensor(one,one,zero,zero)).unit()
    projector9 = logicalone+logicalzero

    # Collect all projectors
    projectors = [projector1,projector2,projector3,projector4,projector5,projector6, projector7,projector8,projector9]
    return projectors

def measurement_and_correction(rho,proj,E,gamma,depol,deph):
    X = qt.Qobj([[0,1],
                [1,0]])
    
    rho1 = ((proj[2]*proj[0])*rho*(proj[2]*proj[0]).dag())  # project unto +1 / +1 outcome
    rho2 = ((proj[2]*proj[1])*rho*(proj[2]*proj[1]).dag())  # project unto -1 / +1 outcome
    rho3 = ((proj[3]*proj[0])*rho*(proj[3]*proj[0]).dag())  # project unto +1 / -1 outcome
    rho4 = ((proj[3]*proj[1])*rho*(proj[3]*proj[1]).dag())  # project unto -1 / -1 outcome
    
    # Trace out ancillary system
    rho1 = rho1.ptrace([0,1,2,3])
    rho2 = rho2.ptrace([0,1,2,3])
    rho3 = rho3.ptrace([0,1,2,3])
    rho4 = rho4.ptrace([0,1,2,3])

    # Apply time-dependent noise for t_3 when getting outcomes +/-1 and -/+1
    for iii in range(1,5):
        rho2 = amp_damping_single_qubit(rho2,gamma,iii)
        rho2 = depolarizing_single_qubit(rho2,depol,iii)
        rho2 = dephasing_single_qubit(rho2,deph,iii)

        rho3 = amp_damping_single_qubit(rho3,gamma,iii)
        rho3 = depolarizing_single_qubit(rho3,depol,iii)
        rho3 = dephasing_single_qubit(rho3,deph,iii)

    # Perform measurement on first qubit
    rho2a = proj[4]*rho2*proj[4].dag() # error on 1
    rho2b = proj[5]*rho2*proj[5].dag() # error on 2
    
    # Perform measurement on third qubit
    rho3a = proj[6]*rho3*proj[6].dag() # error on 3
    rho3b = proj[7]*rho3*proj[7].dag() # error on 4


    # Return all possible (unnormalized) rhos corresponding to the possible measurement outcomes.
    list_of_rhos = [rho1,rho2a,rho2b,rho3a,rho3b,rho4]
    return list_of_rhos

def damp_correction(rho,damped_qubit,n,E):
    # Apply error correction as in arXiv:0710.1052 (2007):
    #   Apply a Hadamard gate on the damped qubit.
    #   With damped qubit as the control, apply a CNOT gate to every other qubit
    #   Flip the damped qubit
    # In our case, we apply controlled-z instead of CNOT

    X = qt.Qobj([[0,1],
                 [1,0]])
    rho = apply_gate(rho,qt.snot(),range(1,n+1))

    for ii in range(1,n+1):
        if ii != damped_qubit:
            rho = permute_matrix(damped_qubit,ii,create_noisy_cz(E),rho)

    rho = apply_gate(rho,qt.snot(),range(1,n+1))
    rho = apply_gate(rho,qt.snot(),[damped_qubit])
    rho = apply_gate(rho,X,[damped_qubit])

    return rho

def perfect_decode(rho):
    # Perfectly decode the state from encoded space to single qubit
    cnot = qt.Qobj([[1,0,0,0],
                    [0,1,0,0],
                    [0,0,0,1],
                    [0,0,1,0]])
    rho = permute_matrix(3,1,cnot,rho)
    rho = permute_matrix(4,2,cnot,rho)
    rho = permute_matrix(1,2,cnot,rho)
    rho = rho.ptrace([0])
    return rho
