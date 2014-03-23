"""
General code pertaining to creating, updating, and propagating Greens functions.
"""

import numpy

from collections import deque
from scipy import det, inv

from helper import grouper,maximumDegeneracy
from math_functions import determinantPhase, RDU, UDR

__all__ = ['initGreens','makeGreensParts','updateGreensV','wrapGreens']

def initGreens(getPhase,paramDict,expVs,sliceGroups): # {{{
    """
    This function initializes and returns a “state” dictionary, which contains the
    initial Green's function (i.e. the produt of the timeslices form 1 to L), the
    partial matrix products entering the Green's function, and the exponential of
    the Ising-field matrix.

    For the partial matrix products, see [1].

    1: Loh, J.E.Y., Gubernatis, J.E., Scalapino, D.J., Sugar, R.L. & White, S.R.
    Proceedings of the Los Alamos Conference on Quantum Simulation 156–167 (1990)
    """
    det = 0
    L = paramDict['L']
    N = paramDict['N']
    expK = paramDict['expK']
    ones = numpy.eye(N,dtype=numpy.complex128)
    U = numpy.copy(ones)
    D = numpy.copy(ones)
    R = numpy.copy(ones)

    no_slices = len(sliceGroups)

# Create the partial matrix products and store them away (in reversed order)
    B_left  = numpy.empty((no_slices,3,N,N),dtype=numpy.complex128)
    B_right = numpy.empty((no_slices,3,N,N),dtype=numpy.complex128)

    chunkCount = no_slices - 1
    for chunk in [sg[::-1] for sg in sliceGroups[::-1]]:
        B = multiplySlicesStart(N,expK,expVs,chunk)
        tmpMatrix = numpy.dot(D,numpy.dot(U,B))
        r,D,U = RDU(tmpMatrix)
        R = numpy.dot(R,r)
        B_left[chunkCount,0] = R
        B_left[chunkCount,1] = D
        B_left[chunkCount,2] = U
        chunkCount -= 1
    Uinv = inv(U)
    Rinv = inv(R)
    tmpMatrix = numpy.dot(Rinv,Uinv) + D
    r,D,u = RDU(tmpMatrix)
    U = numpy.dot(u,U)
    R = numpy.dot(R,r)

    Rinv = inv(R)
    Dinv = inv(D)
    Uinv = inv(U)
    G = numpy.dot(Uinv,numpy.dot(Dinv,Rinv))

    state = {'G': G
                    ,'B_left': B_left
                    ,'B_right': B_right
                    ,'expVs': expVs}

    if getPhase:
        detR,phaseR = determinantPhase(R)
        detU,phaseU = determinantPhase(U)
        detD,phaseD = determinantPhase(D)
        phase       = phaseR*phaseU*phaseD

    return phase,state #}}}

def makeGreensParts(getDeterminant,paramDict,state,sliceCount,sliceGroups): # {{{
    """
    Updates the state dictionary with a new Greens function by calculating it using
    the stored UDR/RDU multiplication groups. Also stores the new UDR
    multiplication group entering this Greens function in the state.
    """
    N = paramDict['N']
    expK = paramDict['expK']

    det = 0
    ones = numpy.eye(N)

    no_slices = len(sliceGroups)
    sliceGroup = sliceGroups[sliceCount-1]

    # Factors for RDU factorization
    if sliceCount < no_slices:
        RL = state['B_left'][sliceCount,0]
        DL = state['B_left'][sliceCount,1]
        UL = state['B_left'][sliceCount,2]
    else: # At the last slice there is no RDU partial product which can be used.
        RL = numpy.copy(ones)
        DL = numpy.copy(ones)
        UL = numpy.copy(ones)

    if sliceCount == 0:
        UR = numpy.copy(ones)
        DR = numpy.copy(ones)
        RR = numpy.copy(ones)
        RLinv = inv(RL)
        ULinv = inv(UL)
        tmpMatrix = numpy.dot(RLinv,ULinv) + DL
        rL,DL,uL = RDU(tmpMatrix)
        UL = numpy.dot(uL,UL)
        RL = numpy.dot(RL,rL)
        RLinv = inv(RL)
        DLinv = inv(DL)
        ULinv = inv(UL)
        state['G'] = numpy.dot(ULinv,numpy.dot(DLinv,RLinv))
    else:
        B = multiplySlicesEnd(N,expK,state['expVs'],sliceGroup)
        if sliceCount == 1:
            UR,DR,RR = UDR(B)
        else:
            UR = state['B_right'][sliceCount-1,0]
            DR = state['B_right'][sliceCount-1,1]
            RR = state['B_right'][sliceCount-1,2]
            tmpMatrix = numpy.dot(numpy.dot(B,UR),DR)
            UR,DR,r = UDR(tmpMatrix)
            RR = numpy.dot(r,RR)

        URinv = inv(UR)
        ULinv = inv(UL)
        tmpMatrix = numpy.dot(URinv,ULinv) + numpy.dot(DR,numpy.dot(RR,numpy.dot(RL,DL)))
        U,D,R = UDR(tmpMatrix)

        Rinv = inv(R)
        Dinv = inv(D)
        Uinv = inv(U)
        state['G'] = numpy.dot(ULinv,numpy.dot(Rinv,numpy.dot(Dinv,numpy.dot(Uinv,URinv))))

    state['B_right'][sliceCount,0] = UR
    state['B_right'][sliceCount,1] = DR
    state['B_right'][sliceCount,2] = RR

    if getDeterminant:
        if sliceCount == 0:
            detUL = det(UL)
            detDL = det(DL)
            detRL = det(RL)
            det   = detUL*detDL*detRL
        else:
            detUL = det(UL)
            detUR = det(UR)
            detD  = det(D)
            detU  = det(U)
            detR  = det(R)
            det   = detUL*detUR*detR*detD*detU

    return det #}}}

def updateGreensV(i,paramDict,state,weightValues): #{{{
    """
    Performs a Sherman-Morrison update of the Green's function in vectorized
    form to allow Numpy to use a C for-loop.
    """
    N        = paramDict['N']
    gamma    = weightValues['gamma']
    det      = weightValues['det']
    newG     = numpy.zeros((N,N),dtype=numpy.complex128)
    coeff    = gamma/det
    delta    = numpy.zeros(N)
    delta[i] = 1
    G        = state['G']
    newG     = G + G[i,:] * (G[:,i,numpy.newaxis] - delta[:,numpy.newaxis]) * coeff
    return newG #}}}

def wrapGreens(expK,l,state): # {{{
    """
    Propagates the Green's function to the next time slice using “wrapping”.
    """
    B    = numpy.dot(expK,state['expVs'][l])
    Binv = inv(B)
    newG = numpy.dot(numpy.dot(B,state['G']),Binv)
    return newG #}}}

def greensDegeneracy(degeneracyDict,A,B): #{{{
    """
    Calculates the degeneracy between two matrices A, B by looking at the element
    with the largest relative difference. It then compares it to the degeneracy of
    a former test stored in a dictionary, and returns the larger degeneracy together
    with the elements it was calculated from.
    """
    ix,degeneracy_new = maximumDegeneracy(A,B)
    if degeneracy_new > degeneracyDict['value']
        return {'value': degeneracy_new, 'old element': A[ix], 'new element': B[ix]}
    else:
        return degeneracyDict #}}}

"""
The below functions are kept for reference, but are not used in the simulation, because
they are surpassed by faster/optimized methods.
"""

def makeGreensUDR(getDeterminant,L,N,expK,expVs,i,m): # Returns a Green's function and the sign of the associated determinant {{{
    det = 0
    order = deque(range(L))
    order.rotate(i)
    order.reverse() # Reverse the order so all the elements get multiplied from the right first
    orderChunks = grouper(order,m)
    I = numpy.eye(N,dtype=numpy.complex128)
    U = numpy.copy(I)
    D = numpy.copy(I)
    R = numpy.copy(I)
    for chunk in orderChunks:
        B = multiplySlicesEnd(N,expK,expVs,chunk)
        tmpMatrix = numpy.dot(numpy.dot(B,U),D)
        U,D,r = UDR(tmpMatrix)
        R = numpy.dot(r,R)
    Uinv = inv(U)
    Rinv = inv(R)
    tmpMatrix = numpy.dot(Uinv,Rinv) + D
    u,D,r = UDR(tmpMatrix)
    U = numpy.dot(U,u)
    R = numpy.dot(r,R)
    if getDeterminant:
        detR=det(R)
        detU=det(U)
        detD=det(D)
        det = detR*detD*detU
    Rinv = s.linv(R)
    Dinv = s.linv(D)
    Uinv = s.linv(U)
    G = numpy.dot(numpy.dot(Rinv,Dinv),Uinv)
    return det,G #}}}

def makeGreensRDU(getDeterminant,L,N,expK,expVs,i,m): # Returns a Green's function and the sign of the associated determinant {{{
    det = 0
    order = deque(range(L))
    order.rotate(i)
    orderChunks = grouper(order,m)

    orderChunks = list(orderChunks)
    numChunks = len(orderChunks)

    I = numpy.eye(N,dtype=numpy.complex128)
    U = numpy.copy(I)
    D = numpy.copy(I)
    R = numpy.copy(I)

    calcChunks = 0

    for chunk in orderChunks:
        B = multiplySlicesStart(N,expK,expVs,chunk)
        tmpMatrix = numpy.dot(D,numpy.dot(U,B))
        r,D,U = RDU(tmpMatrix)
        R = numpy.dot(R,r)
    Uinv = inv(U)
    Rinv = inv(R)
    tmpMatrix = numpy.dot(Rinv,Uinv) + D
    r,D,u = RDU(tmpMatrix)
    U = numpy.dot(u,U)
    R = numpy.dot(R,r)
    if getDeterminant:
        detR = det(R)
        detU = det(U)
        detD = det(D)
        det  = detR*detD*detU
    Rinv = inv(R)
    Dinv = inv(D)
    Uinv = inv(U)
    G = numpy.dot(Uinv,numpy.dot(Dinv,Rinv))
    return det,G #}}}

def makeGreensNaive(getDeterminant,L,N,expK,expVs,i): # As makeGreensUDR, but without stabilization through UDR decomposition {{{
    det = 0
    order = deque(range(L))
    order.rotate(i)
    I = numpy.eye(N,dtype=numpy.complex128)
    A = numpy.eye(N,dtype=numpy.complex128)
    for o in order:
        B = numpy.dot(expK,expVs[o])
        A = numpy.dot(A,B)
    O = I + A
    if getDeterminant:
        det = det(O)
    G = inv(O)
    return det,G #}}}

def updateGreensL(i,paramDict,state,weightValues): #{{{
    """
    Performs a Sherman-Morrison update of the Green's function
    using a (slow) Python for-loop.
    """
    N = paramDict['N']
    gamma = weightValues['gamma']
    det = weightValues['det']
    newG = numpy.zeros((N,N),dtype=numpy.complex128)
    coeff = gamma/det
    G = state['G']
    for j in range(N):
        for k in range(N):
            delta = 1 if j==i else 0
            newG[j,k] = G[j,k] + G[i,k] * (G[j,i] - delta) * coeff
    return newG #}}}

