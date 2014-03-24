import numpy
from ast import literal_eval
from collections import defaultdict
from numbers import Number
from yaml import load

__all__=['getConfig']

def translateComplex(num): #{{{
    '''
    Tries converting a tuple like (1,45) into a complex number. The tuple is
    given in spherical form, z = r·exp(iφ); e.g. above, r = 1, φ=45.
    '''
    if len(num) != 2:
        raise ValueError("Value not given as a tuple: {0}. Spherical form of complex number cannot be parsed.".format(num))
    elif not all(isinstance(x,Number) for x in num):
        raise ValueError("Tuple parts not instances of Number: {0}.".format(num))
    else:
        rad = numpy.deg2rad(num[1])
        return num[0] * complex(numpy.cos(rad), numpy.sin(rad)) #}}}

def readComplex(config_values): #{{{
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
            yield translateComplex(v) #}}}

def getConfig(inputName): #{{{
    inputHandle = open(inputName)
    try:
        config = yaml.load(inputHandle)
    except:
        inputHandle.close()
        raise
    else:
        inputHandle.close()
        configDict = processConfig(config)
        return configDict #}}}

def processConfig(config): #{{{
    """
    Reads and processes the simulation configuration file, filling a dictionary of
    all parameters relevant to the simulation.
    """
    sysConf = config['system']
    simConf = config['simulation']

    paramDict = {'beta':                sysConf['beta']
                ,'idtau':               sysConf['idtau']
                ,'tn':                  sysConf['tn']
                ,'tnn':                 sysConf['tnn']
                ,'U':                   sysConf['U']
                ,'thermalizationSteps': simConf['steps']['thermalization']
                ,'measurementSteps':    simConf['steps']['measurements']
                }

    muConf = sysConf['mu']
    muU    = sysConf['U'] if muConf['type'] == 'units of U' else 1
    mu     = muU * muConf['value']

    BConf  = sysConf['B']
    BU     = sysConf['U'] if BConf['type'] == 'units of U' else 1
    B      = BU * BConf['type']

    lambda2_gen = readComplex(sysConf['lambda2']['values'])
    lambda2_dict = dict(enumerate(lambda2_gen))

    paramDict['domainWall indices'] = defaultdict(list)

    lattice_type = sysConf['lattice']['type'] 

    # Calculate the indices of the domain wall nodes
    if lattice_type == 'parameter':
        paramDict['x'] = sysConf['lattice']['edgeLength']['x']
        paramDict['y'] = sysConf['lattice']['edgeLength']['y']
        paramDict['domainWall'] = list(map(literal_eval,sysConf['lattice']['domainWall']))
        for i,(x,y) in paramDict['domainWall']:
            if x > paramDict['x']:
                raise ValueError("Coordinate {0} in x direction exceeds lattice length {1}".format(x,paramDict['x']))
            elif y > paramDict['y']:
                raise ValueError("Coordinate {0} in y direction exceeds lattice length {1}".format(y,paramDict['y']))
            else:
                paramDict['domainWall indices'][i].append( paramDict['y'] * y + x )
    elif lattice_type == 'file':
        print("This needs more work!")
    else:
        raise ValueError("No such option for lattice: {0}".format(lattice_type))

    paramDict['N'] = paramDict['x'] * paramDict['y']
    paramDict['lambda2 dictionary'] = lambda2_dict

    return paramDict #}}}
