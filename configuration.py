from numpy import array, deg2rad, empty, cos, sin, unique, where
from ast import literal_eval
from collections import defaultdict, Counter
from numbers import Number
from yaml import load
from re import match

import os.path

from itertools import chain

__all__=['read_config']

def process_config(path,config): #{{{
    """
    Reads and processes the simulation configuration file, filling a dictionary of
    all parameters relevant to the simulation.
    """
    sysConf = config['system']
    simConf = config['simulation']

    paramDict = {'beta':                    sysConf['beta']
                ,'idtau':                   sysConf['idtau']
                ,'t':                       sysConf['t']
                ,'periodic':                sysConf['periodic']
                ,'period':                  sysConf['period']
                ,'u':                       sysConf['u']
                ,'reset_factor':            simConf['reset_factor']
                ,'thermalization_steps':    simConf['steps']['thermalization']
                ,'measurement_steps':       simConf['steps']['measurements']
                }

    muConf = sysConf['mu']
    muU    = sysConf['u'] if muConf['type'] == 'units of u' else 1
    mu     = muU * muConf['value']

    BConf  = sysConf['b']
    BU     = sysConf['u'] if BConf['type'] == 'units of u' else 1
    B      = BU * BConf['value']

    lambda2_list = list(read_complex(sysConf['lambda2']['values']))
# Make all entries in the list real if all elements have no imaginary part
    if all(l == 0.0 for l in lambda2_list):
        lambda2_list = [l.real for l in lambda2_list]

    lambda2_values = array(lambda2_list)

    paramDict['x'],paramDict['y'],paramDict['nodes'] = process_lattice(path,sysConf['lattice'])

    paramDict['N'] = paramDict['x'] * paramDict['y']
    paramDict['lambda2 values'] = lambda2_values

    return paramDict #}}}

def read_complex(config_values): #{{{
    '''
    Transforms a list of values from a Yaml configuration file to a list of
    complex numbers, e.g.:
    ['1+2j', (1,90), (1,45), '3+3j'] -> [(1+2j), (0+1j), (0.707+0.707j), (3+3j)]
    '''
    for v in config_values:
        v = literal_eval(v)
        if isinstance(v,Number):
            yield v
        else:
            yield translate_complex(v) #}}}

def read_config(configuration_file): #{{{
    with open(configuration_file, 'r') as configuration_handle:
        try:
            config = load(configuration_handle)
        except:
            raise
        else:
            path,basename = os.path.split(configuration_file)
            configDict = process_config(path,config)
            return configDict #}}}

def lattice_from_file(path_to_lattice): #{{{
    lattice = array(read_lattice(path_to_lattice))
    lattice_flat = lattice.reshape(lattice.size)
    y, x = lattice.shape
    unique_elements = dict(enumerate(unique(lattice)))
    lambda_lattice = empty(y*x,dtype='u2')
    for k,v in unique_elements.items():
        lambda_lattice[where(lattice_flat == v)] = k
    return x,y,lambda_lattice #}}}

def lattice_from_parameter(lattice_dictionary): #{{{
    xdim = lattice_dictionary['edgeLength']['x']
    ydim = lattice_dictionary['edgeLength']['y']

    l2_at_index = list(map(literal_eval,lattice_dictionary['nodes']))

    indices = list(zip(*l2_at_index))[1]
    out_of_lattice = list(filter(lambda xy: xy[0] >= xdim or xy[1] >= ydim, indices))

    if out_of_lattice:
        raise ValueError("Lattice dimensions {0} exceeded by node coordinates: {1}".format((xdim,ydim),", ".join(map(str,out_of_lattice))))

    node_dictionary = defaultdict(list)

    for i,(x,y) in l2_at_index:
        node_dictionary[i].append( y * ydim + x )

    idx_list = list(chain(*node_dictionary.values()))
    idx_set = set(idx_list)

    if len(idx_list) != len(idx_set):
        dups = [i for i,j in Counter(idx_list).items() if j>1]
        raise ValueError("Ambiguous λ₂ for lattice indices: {0}.".format(", ".join(map(str,dups))))

    non_default_keys = set(node_dictionary.keys()) - set([0])
    default_indices = set(range(xdim*ydim))

    for k,v in node_dictionary.items():
        if k in non_default_keys:
            default_indices -= set(v)

    node_dictionary[0] = list(default_indices)
    node_dictionary = dict([k,array(v)] for k,v in node_dictionary.items())

    lambda_lattice = empty(xdim *ydim,dtype='u2')
    for l,idx in node_dictionary.items():
        lambda_lattice[idx] = l

    return xdim,ydim,lambda_lattice #}}}

def process_lattice(path,lattice_dictionary): #{{{
    lattice_input = lattice_dictionary['type']
    if lattice_input == 'parameter':
        return lattice_from_parameter(lattice_dictionary)
    elif lattice_input == 'file':
        path_to_lattice = os.path.join(path,lattice_dictionary['file'])
        return lattice_from_file(path_to_lattice)
    else:
        raise ValueError("No such input type for lattice: {0}".format(lattice_input)) #}}}

def read_lattice(lattice_file): #{{{
    with open(lattice_file, 'r') as lh:
        raw = lh.read()
        rgx = r'^([\n\s0-9]+|[\n\sa-z]+)$'
        if match(rgx, raw) == None:
            raise ValueError("Lattice file should contain only [a-z] OR [0-9]")
        else:
            return [line.split(' ') for line in raw.splitlines()] #}}}

def translate_complex(num): #{{{
    '''
    Tries converting a tuple like (1,45) into a complex number. The tuple is
    given in spherical form, z = r·exp(iφ); e.g. above, r = 1, φ=45.
    '''
    if len(num) != 2:
        raise ValueError("Value not given as a tuple: {0}. Spherical form of complex number cannot be parsed.".format(num))
    elif not all(isinstance(x,Number) for x in num):
        raise ValueError("Tuple parts not instances of Number: {0}.".format(num))
    else:
        rad = deg2rad(num[1])
        return num[0] * complex(cos(rad), sin(rad)) #}}}
