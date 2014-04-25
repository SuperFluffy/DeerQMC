"""
This module contains all functions pertaining to the construction of the
Hamiltonian describing the lattice.
"""
from numpy import array, copy, diag, empty, eye, exp, complex128, float64, kron, size, zeros

from numpy.random import choice
from scipy.linalg import expm2

# Default values which are used in more than one function.
defaults = {'spin species': [-1,+1]}

def shift_matrix(size,offset=1,periodic=True,period=1.0,dtype=float64): #{{{
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
    size : integer
        The order n of a n×n square matrix.
    offset : integer, optional
        The distance from the principal diagonal for the super- and
        subdiagonals, as measured on the counterdiagonal.
        If 1 (default), the superdiagonal U and subdiagonal L are created
        (as per the actual definition of super- and subdiagonal).
    periodic : bool, optional
        If set, two extra super- and subdiagonals are inserted, but
        at a distance of (size - offset), i.e. placed not w.r.t. to the
        diagonal, but w.r.t. to the fartest point from the diagonal.
    period :  float, optional.
        The strength p of the periodic boundary. If p > 0, the boundary
        is periodic; if p < 0, the boundary is antiperiodic.
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
        u *= period
        l *= period
        k += u
        k += l

    return k #}}}

def neighbour_hopping(x,y=1,z=1,distance=1,shift=0,periodic=True,period=1.0,dtype=float64): #{{{
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
    period :  float, optional.
        The strength p of the periodic boundary. If p > 0, the boundary
        is periodic; if p < 0, the boundary is antiperiodic.
    dtype : data-type, optional
        Desired output data type.

    Returns
    -------
    k : ndarray, shape s×s, where s = (x*y*z) (at the moment, always z=1)
    """

    with_distance = lambda dim: shift_matrix(dim,offset=distance,periodic=periodic,period=period,dtype=dtype)
    with_shift = lambda dim: shift_matrix(dim,offset=shift,periodic=periodic,period=period,dtype=dtype)

    k_x = with_distance(x)
    s_y = with_shift(y)

    k = kron(s_y,k_x)

    if distance != shift:
        k_y = with_distance(y)
        s_x = with_shift(x)
        k += kron(k_y,s_x)
    return k #}}}

def hopping_matrix(x,y=1,z=1,couplings=None,periodic=True,period=1.0,dtype=float64): #{{{
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
    couplings : list of numbers (objects that are instances of numbers.Number), optional
        The cardinality of the list fixes the highest degree of
        neighbour-hopping included in the calculation of the kinetic term.  The
        values in the list are the coupling constants belonging to each degree,
        starting with the lowest (i.e. next-neighbour hopping).
        E.g.: [1,0.5] creates a hopping term up to next-to-nearest-neighbour
        (NNN) hopping, with t=1 for NN and t'=0.5 for NNN.  If None, all matrix
        entries are set to 0 (i.e., no hopping occurs).
    periodic : bool, optional
        If set, periodic boundaries at the lattice edges are set, so that
        hoppings between “start” and “end” of the lattice can occur.
    period :  float, optional.
        The strength p of the periodic boundary. If p > 0, the boundary is
        periodic; if p < 0, the boundary is antiperiodic.
    dtype : data-type, optional
        Desired output data type.

    Returns
    -------
    k : ndarray, shape s×s, where s = (x*y*z) (at the moment, always z=1)

    Raises
    ------
    ValueError
        If there are so many entries in neighbour_couplings that the hopping
        vector connecting two nodes wraps around the lattice (e.g. in cases
        like a hopping occuring between a node and itself).
    """

    edge_length = x*y
    k = zeros((edge_length,edge_length))

    if couplings is not None:
        d = min(x,y) - 1
        maximum_neighbours = (d**2 + 3*d)//2
        neighbours = len(couplings)

        if neighbours > maximum_degree:
            error_message = """Couplings for a hopping degree of up to {0} found, \
                but only a degree of {1} stays within the lattice (dimensions (x,y) \
                = ({2},{3})).""".format(neighbours, maximum_degree, x, y)

            raise ValueError(error_message)
        else:
            distance = 0
            shift = 0
            for t in couplings:
                if shift < distance:
                    shift += 1
                else:
                    distance += 1
                    shift = 0
                k += -t * neighbour_hopping(x,y=y,z=z,distance=distance,shift=shift,periodic=periodic,period=period,dtype=dtype)
    return k #}}}

def auxiliary_field(timeslices,lattice_sites,spin_species=None): #{{{
    """
    Returns a randomly generated configuration of Ising spins for the
    space-time lattice of the decoupled electron-electron interaction.

    Parameters
    ----------
    timeslices : integer
        The number of timeslices.
    lattice_sites : integer
        The number of sites on the lattice.
    spin_species : list, optional
        The set of values the auxiliary spins can take. If None (default),
        then only Ising spins (i.e. {↑,↓} represented as the list [+1,-1])
        are used.

    Returns
    -------
    spacetime_field : ndarray, shape (timeslices×lattice_sites)
    """
    if spin_species is None:
        spin_species = defaults['spin species']
    randarray = choice(spin_species,size=N*L)
    spacetime = randarray.reshape(L,N)
    return spacetime_field #}}}

def possible_values(lambda_lattice,spin_species=None): #{{{
    """
    Returns an array containing all possible values the decoupled potential
    term can take on each lattice site depending on the auxiliary Ising.
    Note: at the moment this assumes an isotropic distribution of the λ
    coupling parameters in time direction.

    Parameters
    -----------
    lambda_lattice : ndarray, shape n
        An one-dimensional array of shape n (the number of lattice sites), which
        consists of the decoupling parameter λ_i on each lattice site.
    spin_species : list, optional
        The set of values the auxiliary spins can take. If None (default),
        then only Ising spins (i.e. {↑,↓} represented as the list [+1,-1])
        are used.

    Returns
    -------
    a : ndarray, shape s×n, where s = len(spin_species)
    """

    if spin_species is None:
        spin_species = defaults['spin species']

    a = empty((len(spin_species), lambda_lattice.size))

    for i,s in enumerate(spin_species):
        a[i] = s * lambda_lattice

    return a #}}}

def decoupled_potential(timeslices,decoupling_terms,spin_species=None,dtype=float64): #{{{
    """
    Returns a random configuration of the auxiliary Ising spins for each of the
    terms entering the decoupled potential, as well as the set of all possible
    values that the decoupled terms can take on each lattice site.

    Parameters
    ----------
    timeslices : integer
        The number of timeslices.
    decoupling_terms : list of ndarrays
        A list containing a one-dimensional array of shape n with λ_{k,i}
        values for each of the k decoupling terms on every space-site with
        index i, where n is the total number of lattice sites.
    spin_species : list, optional
        The set of values the auxiliary spins can take. If None (default),
        then only Ising spins (i.e. {↑,↓} represented as the list [+1,-1])
        are used.

    Returns
    -------
    fields : ndarray, shape (d×l×n), where
        d = number of decoupling terms
        l = number of timeslices
        n = number of lattice sites

    values : ndarray, shape (d×s×n), where s = len(spin_species)
    """

    lattice_sites = size(decoupling_terms[0])
    d = len(decoupling_terms)

    fields = empty((d,timeslices,lattice_sites),dtype=dtype)
    values = empty((d,len(spin_species),lattice_sites),dtype=dtype1)

    for i,lambda_lattice in enumerate(decoupling_terms):
        fields[i] = auxiliary_fields(timeslices,lattice_sites,spin_species)
        values[i] = possible_values(lambda_lattice,spin_spcies)

    return fields,values #}}}

def hamiltonian(parameter_dictionary): #{{{
    """
    Glues together the construction of all terms appearing in the Hamiltonian,
    i.e. the kinetic, chemical, magnetic terms k, c, and m, respectively; also
    includes the radom generation of a configuration for each of the
    auxiliary fields decoupling the potential term, and the possible values the
    decoupled terms can take on each lattice site.

    Parameters
    ----------
    parameter_dictionary : dictionary
        The dictionary containing all parameters describing the simulation.

    Returns
    -------
    k : ndarray, shape (n×n), where
        n = number of lattice sites
    c : ndarray, shape (n×n)
    m : ndarray, shape (n×n)
    fields : ndarray, shape (d×l×n), where
        d = the number of discretization terms
        l = the number of time slices
    values : ndarray, shape (d×s×n), where
        s = the number of spin species/channels in the original Hamiltonian
    """
    x = parameter_dictionary['x']
    y = parameter_dictionary['y']
    z = parameter_dictionary['z']

    t = parameter_dictionary['t']
    mu = parameter_dictionary['mu']
    b = parameter_dictionary['b']

    decoupling_terms = parameter_dictionary['decoupling terms']
    periodic = parameter_dictionary['periodic']
    period = parameter_dictionary['period']

    spin_species = parameter_dictionary['spin species']

    lattice_sites = x*y*z

    dtype = decoupling_terms[0].dtype

    k = hopping_matrix(x=x,y=y,z=1,neighbour_couplings=t,periodic=periodic,period=period,dtype=dtype)
    c = (-1) *  mu * eye(lattice_sites,dtype=dtype)
    m = (-1) *  b  * eye(lattice_sites,dtype=dtype)

    fields,values = decoupled_potential(timeslices,decoupling_terms,spin_species)

    return k, c, m, fields, values #}}}

def discretized_exponentials(dtau,k,c,m,values): #{{{
    """
    Returns a dictionary of the discretized exponentials of the constants
    Hamiltonian terms, as well as the exponentials of all values the decoupled
    potential terms can take.
    """

    exp_dictionary['k'] = expm2(-dtau * k)
    exp_dictionary['c'] = expm2(-dtau * c)
    exp_dictionary['m'] = expm2(-dtau * m)

    exp_dictionary['values'] = exp(values)

    return exp_dictionary #}}}

# {{{ Stuff that was refactored away 
    #V1  = lambda1_general    * array([diag(space) for space in (lattice_general * spacetime_1)],dtype=float64)
    #V1 += lambda1_domainWall * array([diag(space) for space in (lattice_domainWall * spacetime_1)],dtype=float64)

    #V2  = lambda2_general    * array([diag(space) for space in (lattice_general * spacetime_2)],dtype=float64)
    #V2 += lambda2_domainWall * array([diag(space) for space in (lattice_domainWall * spacetime_2)],dtype=float64)

    #expVs_up = array([expm2(spinUp*v1 + spinUp_other * v2 + C + M) for (v1,v2) in zip(V1,V2)])
    #expVs_dn = array([expm2(spinDn*v1 + spinDn_other * v2 + C - M) for (v1,v2) in zip(V1,V2)])

    #lambda1_general = paramDict['lambda1 general']
    #lambda2_general = paramDict['lambda2 general']

    #lambda1_domainWall = paramDict['lambda1 domainWall']
    #lambda2_domainWall = paramDict['lambda2 domainWall']
#}}}


