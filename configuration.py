import numpy

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

    if sysConf['lambda2']['domainWall']['complexForm'] == 'polar':
        angle = sysConf['lambda2']['domainWall']['angle']
        rad = numpy.deg2rad(angle)
        lambda2_domainWall = lambda2_domainWall * complex(numpy.cos(rad),numpy.sin(rad))
    elif sysConf['lambda2']['domainWall']['complexForm'] == 'rectangular':
        lambda2_domainWall = complex(lambda2_domainWall)
    else:
        raise ValueError("Option '{}' for 'complexForm' in 'lambda2' not recognized.".format(sysConf['lambda2']['domainWall']['complexForm']))

    paramDict['lambda2 general'] = lambda2_general
    paramDict['lambda2 domainWall'] = lambda2_domainWall

    return paramDict #}}}
