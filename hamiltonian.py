"""
This module contains all functions pertaining to the construction of the
Hamiltonian describing the lattice.
"""
import numpy

from numpy.random import choice
from scipy.linalg import expm2

def make_field(L,N,spinsSample=None): #{{{
    if sample == None:
        spinsSample = [-1,+1]
    randarray = choice(spinsSample,size=N*L)
# Store array as (L,N), since the Python stores in row-major form -> faster access
    spacetime = randarray.reshape(L,N)
    return spacetime #}}}

def shift_matrix(size,offset=1,periodic=True,antiperiodic=False,dtype=float64): #{{{
    """
    Returns a shift matrix with super- and subdiagonals with a distance
    <offset> from the principal diagonal. When <periodic> is set, extra
    diagonals are inserted, but counted from farthest (diagonal) distance from
    the principal. This is useful when generating hopping terms which “wrap
    around” the edge of the lattice.
    Note: The shift matrix S = U + L as found in literature, with superdiagonal
    U and subdiagonal L, corresponds to the settings offset=0, periodic=False.
    Parameters
    ----------
    size : int
        The order n of a n×n square matrix.
    offset : int, optional
        The distance from the principal diagonal for the super- and
        subdiagonals, as measured on the counterdiagonal.
        If 1 (default), the superdiagonal U and subdiagonal L are created
        (as per the actual definition of super- and subdiagonal).
    periodic : bool, optional
        If set, two extra super- and subdiagonals are inserted, but
        at a distance of (size - offset), i.e. placed not w.r.t. to the
        diagonal, but w.r.t. to the fartest point from the diagonal.
    antiperiodic :  bool, optional.
        If set, the periodic diagonals are multiplied by (-1).
        This setting is ignored if periodic is not set.
    dtype : data-type, optional
        Desired output data type.

    Returns
    -------
    k : ndarray, shape (size×size)

    """
    k  = eye(size,k=+offset,dtype=dtype)
    if offset != 0:
        k += eye(size,k=-offset,dtype=dtype)

    if periodic and size>offset:
        u = eye(size,k=(size-offset))
        l = eye(size,k=-(size-offset))
        if antiperiodic:
            u *= -1
            l *= -1
        k += u
        k += l

    return k #}}}

def neighbour_hopping(x,y=1,z=1,distance=1,shift=0,periodic=True,antiperiodic=False,dtype=float64): #{{{
    """
    Returns the matrix with entries describing the hopping between neighbouring
    sites in a square lattice. The degree of the hopping (e.g.
    nearest-neighbour, next-to-nearest-neighbour) is defined by the set of
    vectors with entries consisting of permutations of d, <distance>, and s,
    <shift>.
    For example: in a 2D lattice, these vectors would be {(±d, ±s),(±s,±d)}.
    Nearest-neighbour hopping in this case would be completely described by
    the four vectors {(±1,0),(0,±1)}.
    Next-to-nearest-neighbour hopping would be completely described by
    {(±1,±1)}.

    Parameters
    ----------
    x : integer
        Length/number of nodes in x direction.
    y : integer, optional
        Length/number of nodes in y direction.
    z : integer, optional
        Length/number of nodes in z direction.
        Currently a dummy variable without effect.
    distance : integer, optional
        The component d in the (d,s) vector tuple.
        If 1 (default), generates nearest-neighbour hopping
        together with the default value for <shift> (0).
    shift : integer, optional
        The component s in the (d,s) vector tuple.
        If 0 (default), generates nearest-neighbour hopping
        together with the default value for <distance> (1).
    periodic : bool, optional
        If set, periodic boundaries at the lattice edges are set, so
        that hoppings between “start” and “end” of the lattice can occur.
        diagonal, but w.r.t. to the fartest point from the diagonal.
    antiperiodic :  bool, optional.
        If set, the the boundary hopping is anti-periodic (i.e. multiplied
        by the factor (-1).
    dtype : data-type, optional
        Desired output data type.

    Returns
    -------
    k : ndarray, shape s×s, where s = (x*y*z) (at the moment, always z=1)
    """

    with_distance = lambda dim: shift_matrix(dim,offset=distance,periodic=periodic,antiperiodic=antiperiodic,dtype=dtype)
    with_shift = lambda dim: shift_matrix(dim,offset=shift,periodic=periodic,antiperiodic=antiperiodic,dtype=dtype)

    k_x = with_distance(x)
    s_y = with_shift((y)

    k = kron(s_y,k_x)

    if distance != shift:
        k_y = with_distance(y)
        s_x = with_shift(x)
        k += kron(k_y,s_x)
    return k #}}}

def hopping_matrix(x,y=1,z=1,neighbours=1,periodic=True,antiperiodic=False,dtype=float64): #{{{
    """
    Returns the hopping matrix, i.e. the matrix in kinetic tight-binding term,
    for a lattice with dimensions (x,y,z) (at the moment z is meaningless).
    Interactions for arbitrary neighbour-degrees are possible, where e.g. <neighbour>=1
    describes nearest-neighbour hopping, <neighbour>=2 describes nearest-neighbour and
    next-to-nearest-neighbour hopping, etc.

    Parameters
    ----------

    x : integer
        Length/number of nodes in x direction.
    y : integer, optional
        Length/number of nodes in y direction.
    z : integer, optional
        Length/number of nodes in z direction.
        Currently a dummy variable without effect.
    neighbours : integer, optional
        Up to which neighbour-degree hopping should be included. E.g.,
        a value of 2 includes nearest- and next-to-nearest-neighbour
        hopping.
        If 1 (default), only nearest-neighbour hopping is included.
    periodic : bool, optional
        If set, periodic boundaries at the lattice edges are set, so
        that hoppings between “start” and “end” of the lattice can occur.
        diagonal, but w.r.t. to the fartest point from the diagonal.
    antiperiodic :  bool, optional.
        If set, the the boundary hopping is anti-periodic (i.e. multiplied
        by the factor (-1).
    dtype : data-type, optional
        Desired output data type.

    Returns
    -------
    k : ndarray, shape s×s, where s = (x*y*z) (at the moment, always z=1)
    """

    edge_length = x*y
    k = zeros((edge_length,edge_length))

    distance = 1
    shift = 0

    #for n in range(neighbours):
    while neighbours > 0:
        k += neighbour_hopping(x,y=y,z=z,distance=distance,shift=shift,periodic=periodic,antiperiodic=antiperiodic,dtype=dtype)
        neighbours -= 1
        if shift < distance:
            shift += 1
        else:
            distance += 1
            shift = 0

    return k #}}}

def potential_matrix(paramDict,C,M): #{{{
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

def make_hamiltonian(paramDict): #{{{
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

    C = (dtau*mu) * eye(N,dtype=float64)
    M = (dtau*B)  * eye(N,dtype=float64)

    spacetime_1,spacetime_2,expVs_up,expVs_dn = potential_matrix(paramDict,C,M)

    return expK, spacetime_1, spacetime_2, expVs_up, expVs_dn #}}}
