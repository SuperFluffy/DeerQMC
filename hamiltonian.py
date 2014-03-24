"""
This module contains all functions pertaining to the construction of the
Hamiltonian describing the lattice.
"""
import numpy

from numpy.random import choice
from scipy.linalg import expm2

def makeField(L,N,spinsSample=None): #{{{
    if sample == None:
        spinsSample = [-1,+1]
    randarray = choice(spinsSample,size=N*L)
# Store array as (L,N), since the Python stores in row-major form -> faster access
    spacetime = randarray.reshape(L,N)
    return spacetime #}}}

def makeHopp1D(n,k): # {{{
    K  = numpy.eye(n,k=+k,dtype=numpy.float64)
    K += numpy.eye(n,k=-k,dtype=numpy.float64)
# Set the matrix elements to fulfil the PBC by hand. Has no effect on a 2-site chain
    if n>k:
        K += numpy.eye(n,k=(n-k))
        K += numpy.eye(n,k=-(n-k))
    return K #}}}

def makeHopp2D(nx,ny,k): # 2D hopping matrix for symmetric, square lattices {{{
    Kx = makeHopp1D(nx,k)
    Ix = numpy.eye(nx,dtype=numpy.float64)

    Ky = makeHopp1D(ny,k)
    Iy = numpy.eye(ny,dtype=numpy.float64)

    K = numpy.kron(Iy,Kx) + numpy.kron(Ky,Ix)
    return K #}}}

def makePotential(paramDict,C,M): #{{{
    L = paramDict['L']
    N = paramDict['N']
    lambda1_general    = paramDict['lambda1 general']
    lambda2_general    = paramDict['lambda2 general']
    lambda1_domainWall = paramDict['lambda1 domainWall']
    lambda2_domainWall = paramDict['lambda2 domainWall']
    spinUp = paramDict['spinUp']
    spinDn = paramDict['spinDn']

    spinUp_other = paramDict['spinUp_other']
    spinDn_other = paramDict['spinDn_other']

    spacetime_1 = makeField(L,N)
    spacetime_2 = makeField(L,N)

    lattice_general = paramDict['lattice general']
    lattice_domainWall = paramDict['lattice domainWall']

    V1  = lambda1_general    * numpy.array([numpy.diag(space) for space in (lattice_general * spacetime_1)],dtype=numpy.complex128)
    V1 += lambda1_domainWall * numpy.array([numpy.diag(space) for space in (lattice_domainWall * spacetime_1)],dtype=numpy.complex128)

    V2  = lambda2_general    * numpy.array([numpy.diag(space) for space in (lattice_general * spacetime_2)],dtype=numpy.complex128)
    V2 += lambda2_domainWall * numpy.array([numpy.diag(space) for space in (lattice_domainWall * spacetime_2)],dtype=numpy.complex128)

    expVs_up = numpy.array([expm2(spinUp*v1 + spinUp_other * v2 + C + M) for (v1,v2) in zip(V1,V2)])
    expVs_dn = numpy.array([expm2(spinDn*v1 + spinDn_other * v2 + C - M) for (v1,v2) in zip(V1,V2)])

    return spacetime_1,spacetime_2,expVs_up,expVs_dn
# }}}

def makeHamiltonian(paramDict): #{{{
    """
    This function constructs all the Hamiltonian describing the system from the
    quantities stored in the parameter dictionary.
    """
    edgeLength_x = paramDict['edgeLength x']
    edgeLength_y = paramDict['edgeLength y']
    N = paramDict['N']
    tn = paramDict['tn']
    tnn = paramDict['tnn']
    mu = paramDict['mu']
    B = paramDict['B']
    dtau = paramDict['dtau']

    lambda1_general = paramDict['lambda1 general']
    lambda2_general = paramDict['lambda2 general']

    lambda1_domainWall = paramDict['lambda1 domainWall']
    lambda2_domainWall = paramDict['lambda2 domainWall']

    spinUp = paramDict['spinUp']
    spinDn = paramDict['spinDn']

    spinUp_other = paramDict['spinUp_other']
    spinDn_other = paramDict['spinDn_other']

    Kn  = (-dtau*tn) *  makeHopp2D(edgeLength_x,edgeLength_y,1)
    Knn = (-dtau*tnn) * makeHopp2D(edgeLength_x,edgeLength_y,2)
    K = Kn + Knn
    expK = expm2(-1*K)

    C = (dtau*mu) * numpy.eye(N,dtype=numpy.float64)
    M = (dtau*B)  * numpy.eye(N,dtype=numpy.float64)

    spacetime_1,spacetime_2,expVs_up,expVs_dn = makePotential(paramDict,C,M)

    return expK, spacetime_1, spacetime_2, expVs_up, expVs_dn #}}}
