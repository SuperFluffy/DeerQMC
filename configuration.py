import numpy
from numbers import Number

def translateComplex(num):
    '''
    Tries converting a string like '1+1j' or a tuple like (1,45) into a complex
    number. The tuple is given in spherical form, z = r·exp(iφ); e.g. above, 
    r = 1, φ=45.
    '''
    try:
        return complex(num)
    except (ValueError, TypeError):
        if len(num) != 2:
            raise ValueError("Value not given as a tuple: {0}. Spherical form of complex number cannot be parsed.".format(num))
        elif not all(isinstance(x,Number) for x in num):
            raise ValueError("Tuple not an instance of Number: {0}.".format(num))
        else:
            rad = numpy.deg2rad(num[1])
            return num[0] * complex(numpy.cos(rad), numpy.sin(rad))

def readComplex(config_values):
    '''
    Transforms a list of values from a Yaml configuration file to a list of
    complex numbers, e.g.:
    ['1+2j', (1,90), (1,45), '3+3j'] -> [(1+2j), (0+1j), (0.707+0.707j), (3+3j)]
    '''
    for v in config_values:
        yield translateComplex(v)

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
                ,'edgeLength x':        sysConf['lattice']['edgeLength']['x']
                ,'edgeLength y':        sysConf['lattice']['edgeLength']['y']
                ,'useLambda2':          simConf['useLambda2']
                ,'thermalizationSteps': simConf['steps']['thermalization']
                ,'measurementSteps':    simConf['steps']['measurements']
                }

    paramDict['N'] = paramDict['edgeLength x'] * paramDict['edgeLength y']

    muConf = sysConf['mu']
    muU    = sysConf['U'] if muConf['type'] == 'units of U' else 1
    mu     = muU * muConf['value']

    BConf  = sysConf['B']
    BU     = sysConf['U'] if BConf['type'] == 'units of U' else 1
    B      = BU * BConf['type']

    paramDict['domainWall'] = numpy.array( list( map( ast.literal_eval, sysConf['lattice']['domainWall'] ) ) )
    paramDict['domainWall indices'] = []

    # Calculate the indices of the domain wall nodes
    for (x,y) in paramDict['domainWall']:
        if x > paramDict['edgeLength x']:
            raise ValueError("Coordinate {0} in x direction exceeds lattice length {1}".format(x,paramDict['edgeLength y']))
        elif y > paramDict['edgeLength y']:
            raise ValueError("Coordinate {0} in y direction exceeds lattice length {1}".format(y,paramDict['edgeLength y']))
        else:
            paramDict['domainWall indices'].append( paramDict['edgeLength y'] * y + x )

    lambda2_general    = sysConf['lambda2']['general']
    lambda2_domainWall = sysConf['lambda2']['domainWall']

    if sysConf['lambda2']['general']['complexForm'] == 'polar':
        angle = sysConf['lambda2']['general']['angle']
        rad = numpy.deg2rad(angle)
        lambda2_general = lambda2_general * complex(numpy.cos(rad),numpy.sin(rad))
    elif sysConf['lambda2']['general']['complexForm'] == 'rectangular':
        lambda2_general = complex(lambda2_general)
    else:
        raise ValueError("Option '{}' for 'complexForm' in 'lambda2' not recognized.".format(sysConf['lambda2']['general']['complexForm']))

    lambda2_gen = readComplex(paramDict['system']['lambda2']['values'])
    lambda2_dict = dict(enumerate(lambda2_gen))

    paramDict['lambda2 dictionary'] = lambda2_dict

    return paramDict #}}}
