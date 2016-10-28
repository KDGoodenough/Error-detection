## Calculate entanglement fidelity of a qubit with and without 
# an amplitude-damping (AD) error correction scheme over multiple cycles
# There are multiple error parameters: Dephasing/depolarizing extra AD noise
# , noisy cz gates, noisy ancilla state preparation. The script first calculates the
# superoperators for a single cycle of an encoded and bare qubit. For the encoded qubit
# the state is projected back into the codespace. After the superoperators are found,
# the superoperators are `enlarged' to act on a statespace of two qubits. That is, we
# apply the noise and correction only to the second qubit. By applying this superoperator
# to a maximally entangled state we can get the entanglement fidelities, which are then plotted.

import numpy as np
np.set_printoptions(threshold=np.nan)
import qutip as qt
import matplotlib.pyplot as plt
from QECC_functions import *
from Measurements_and_correction import *
from ErrorModels import *
from Permute_matrix import *

print('-----------------------------------')
print('~---------------------------------~')
print('-----------------------------------')

cycles = 20         # Set the amount of cycles to compute

t_1 = 3           # Set times during which time-dependent noise happens
t_2 = 0.1
t_3 = 0.1
t_total = t_1+t_2+0*t_3                 # Calculate total time to compare with the single qubit. t_3 is excluded since this step doesn't always occur
t = np.array([t_1,t_2,t_3,t_total])     # Create list of all times

T_1 = 50                                # Set T_1 time
T_2 = 90                                # Set T_2 time. Can not be greater than 2*T_2
T_depol = 99999999999999                # Set noise from depolarizing

gamma = 1-np.exp(-t/T_1)                            # Calculate gammas from the waiting times
deph = (1-(np.exp(-t/T_2))/np.sqrt((1-gamma)))/2    # Calculate probability of xflip from the waiting times  https://arxiv.org/pdf/1404.3747v3.pdf
depol = 1-np.exp(-t/T_depol)

ps = 4.4e-5               # Set noise parameter for noisy cz, see Physical Review A, 88(1):012314 and Phys. Rev. A, 87(2):022309
E = (8/5)*ps



state_prep = 0.005         # Amount of depolarizing noise on ancilla qubits
input_state = qt.Qobj([[1/2,1/2],
                       [1/2,1/2]])

proj = create_projectors()      # Create projectors needed for the measurements

# create ancilla state
plus = qt.Qobj([[1/2,1/2],
                [1/2,1/2]])

rho = encoding(input_state)

# Create expected perfect states. Do this by damping a single qubit (or all of them), and then renormalizing the state.
damping = qt.Qobj([[0,1],
                [0,0]])
rho1_perfect  = rho.unit() # no-damping
rho2a_perfect = (apply_gate(rho,damping,[1])).unit() # Damping on first qubit
rho2b_perfect = (apply_gate(rho,damping,[2])).unit() # Damping on second qubit
rho3a_perfect = (apply_gate(rho,damping,[3])).unit() # Damping on third qubit
rho3b_perfect = (apply_gate(rho,damping,[4])).unit() # Damping on fourth qubit
rho4_perfect  = (apply_gate(rho,damping,[4])).unit() # what to do here?!? non-correctable syndrome
list_of_perfect_rhos = [rho1_perfect,rho2a_perfect,rho2b_perfect,rho3a_perfect,rho3b_perfect,rho4_perfect] # create list in order of detected expected states

# create list of fidelities for each measurement outcome 
list_of_fidelities = []


# Start of protocol
# Create encoded state and set dimensions
rho = encoding(input_state)
rho.dims = [[2,2,2,2],[2,2,2,2]]

# Add time-dependent noise for t_1
for ii in range(1,5):
    rho = amp_damping_single_qubit(rho,gamma[0],ii)
    rho = depolarizing_single_qubit(rho,depol[0],ii)
    rho = dephasing_single_qubit(rho,deph[0],ii)

# Append ancilla qubits and add depolarizing noise
rho = qt.tensor(rho,plus,plus)
for ii in range(4,7):
    rho = depolarizing_single_qubit(rho,state_prep,ii)
                
# CZ for parity check Z1Z2
rho = permute_matrix(1,5,create_noisy_cz(E),rho)
rho = permute_matrix(2,5,create_noisy_cz(E),rho)
# CZ for parity check Z3Z4
rho = permute_matrix(3,6,create_noisy_cz(E),rho)
rho = permute_matrix(4,6,create_noisy_cz(E),rho)

# Apply hadamards
rho = apply_gate(rho,qt.snot(),[5,6])

# Add time-dependent noise for t_2
for ii in range(1,7):
    rho = amp_damping_single_qubit(rho,gamma[1],ii)
    rho = depolarizing_single_qubit(rho,depol[1],ii)
    rho = dephasing_single_qubit(rho,deph[1],ii)

# Perform measurements and correction. Supply error parameters for
# the time-dependent noise and projectors for the measurements
list_of_rhos = measurement_and_correction(rho,proj,E,gamma[2],depol[2],deph[2])

# Create empty list of probabilities and fidelities
list_of_probabilities = []
list_of_fidelities = []

# Run over list of rhos to calculate probabilities and corresponding fidelities
for ii in range(0,len(list_of_rhos)):
    list_of_probabilities.append((list_of_rhos[ii]).tr())
    fidelity = qt.fidelity(list_of_perfect_rhos[ii],(list_of_rhos[ii]).unit())
    list_of_fidelities.append(fidelity)

# print results

print("List of probabilities")
print(list_of_probabilities)
print("list of fidelities")
print(list_of_fidelities)
print("average fidelity")
print(np.sum(np.multiply(list_of_fidelities,list_of_probabilities)))
#print(list_of_fidelities)
# Project down into codespace
# rho = proj[8]*rho*proj[8].dag()
# Perform noisefree decoding
            
