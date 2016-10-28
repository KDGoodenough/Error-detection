# Several different functions used in the script

import numpy as np
import scipy.stats
import qutip as qt
from mpmath import *
import pylab
np.set_printoptions(threshold=np.nan)
import csv
from Permute_matrix import *
from ErrorModels import *

def encoding(rho):
    ## Encode qubit perfectly into codespace
    plus = qt.Qobj([[1/2,1/2],
                    [1/2,1/2]])
    # append plus states
    rho = qt.tensor(rho,plus,plus,plus)
    # create controlled-z gate
    rho = permute_matrix(1,2,create_noisy_cz(0),rho)
    rho = apply_gate(rho,qt.snot(),[1])

    for ii in range(1,4):
        rho = permute_matrix(4,ii,create_noisy_cz(0),rho)

    rho = apply_gate(rho,qt.snot(),[1,2,3])
    return rho

def create_basis():
    # Create a list of `computational' basis elements of a qubit density matrix
    rho_basis = []
    rho_basis.append(qt.Qobj([[1,0],
                              [0,0]]))
    rho_basis.append(qt.Qobj([[0,1],
                              [0,0]]))
    rho_basis.append(qt.Qobj([[0,0],
                              [1,0]]))
    rho_basis.append(qt.Qobj([[0,0],
                              [0,1]]))
    return rho_basis


def entanglement_fidelity(superoperator,cycles):
    # Calculate entanglement fidelity over the given amount of cycles by applying the enlarged
    # operator to a maximally entangled state, and calculating the fidelity with the initial state
    # and the state after each cycle

    # `enlarge' superoperator such that it acts only on the second qubit of a two-qubit state
    superoperator = enlargesuperoperator(superoperator)

    # Create maximally entangled states
    state = 1/2*np.array([[1,0,0,1],
                          [0,0,0,0],
                          [0,0,0,0],
                          [1,0,0,1]])
    state_init = qt.Qobj(1/2*np.array([[1,0,0,1],
                          [0,0,0,0],
                          [0,0,0,0],
                          [1,0,0,1]]))
    state.shape = (16,1)

    # Calculate the fidelity for each cycle
    fidelity = np.zeros(cycles)
    for ii in range(0,cycles):
        state = np.dot(superoperator,state)
        qstate = qt.Qobj(np.reshape(state,(4,4)))
        fidelity[ii] = qt.fidelity(qstate,state_init)

    return fidelity

def enlargesuperoperator(superoperator):
    # `enlarge' superoperator such that it acts only on the second qubit of a two-qubit state
    # derivation is given in the .pdf file. This is specifically for the case that n = 2, general 
    # case can be found also in the .pdf file

    superoperator = qt.Qobj(superoperator)  # make Qobj for tensor products
    K = qt.Qobj(np.array([[1,0,0,0],        # create commutation matrix for n = 2, i.e. the swap gate
                          [0,0,1,0],
                          [0,1,0,0],
                          [0,0,0,1]]))
    P = qt.tensor(qt.identity(2),K,qt.identity(2))
    P.dims = [[16],[16]]
    superoperator = qt.tensor(qt.identity(4),superoperator)
    superoperator.dims = [[16],[16]]
    superoperator = P*(superoperator)*P.trans()



    
    return superoperator.full()

def exportdata(list,location):
    # Export data, where list is the data to be stored, location is the name of the txt file
    string_float = [str(x) for x in list]
    text_file = open(location, "w")
    for i in range(0,len(string_float)):
        text_file.write(string_float[i]+"\n")
    text_file.close()

def plot_data(fidelity_enc,fidelity_single_qubit,T_1,T_2,t_1,t_2,t_3,state_prep, ps):
    # plot entanglement fidelities, with the corresponding error parameters
    cycles = len(fidelity_enc)
    cyclerange = np.linspace(1,cycles,cycles)
    pylab.plot(cyclerange, np.power(fidelity_enc,2),'bo',markersize=10,)
    pylab.plot(cyclerange, np.power(fidelity_enc,2),'b',linewidth=5, label='Encoded qubit')
    pylab.plot(cyclerange, np.power(fidelity_single_qubit,2),'go',markersize=10)
    pylab.plot(cyclerange, np.power(fidelity_single_qubit,2),'g--',linewidth=5, label='Single qubit')
    minimum = np.minimum(fidelity_single_qubit,fidelity_enc)
    pylab.axis([0, cycles+1, 0.25, 1])
    pylab.ylim([0.5,1.01])
    pylab.xticks(fontsize = 20)
    pylab.yticks(fontsize = 20)
    pylab.legend(loc='lower left',prop={'size':29.5},markerscale = 10)
    pylab.grid()

    fig = pylab.figure(1, figsize=(8,8))
    bbox_props = dict(boxstyle="round4,pad=0.3", fc="lightgray", ec="darkgray", lw=2)
    ax = fig.add_subplot(111)
    pylab.xlabel(r'Cycles',fontsize=21)
    pylab.ylabel(r'Entanglement fidelity',fontsize=21)
    t = ax.text(3.8, 0.49+0.188, '$T_1 =\/%s,\/T_2 =\/%s $\n $t_1 =%s,\/t_2 = \/%s,\/t_3=\/%s$\n Prep$\/=\/%s,\/$ps$ =\/%s$' % (T_1, np.round(T_2),t_1,t_2,t_3,state_prep,ps), ha="center", va="center", rotation=0,
            size=21,
            bbox=bbox_props)
    mng = pylab.get_current_fig_manager()
    mng.window.showMaximized()
    pylab.show()

    return