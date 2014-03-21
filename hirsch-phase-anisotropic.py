#!/usr/bin/python

import numpy 

from itertools    import count
from math         import floor
from scipy.linalg import LinAlgError

import logging
import yaml
import argparse

import time
from datetime import timedelta

import os.path as osp
import h5py

import ast

from hamiltonian import makeHamiltonian
from helper import grouper
from support import *

# Construct dtype dtypes {{{
def construct_parameter_dtype(domainWall):
  parameter_dtype = numpy.dtype([('beta',numpy.float64)
                                 ,('tn',numpy.float64)
                                 ,('tnn',numpy.float64)
                                 ,('U',numpy.float64)
                                 ,('mu',numpy.float64)
                                 ,('B',numpy.float64)
                                 ,('dtau',numpy.float64)
                                 ,('lambda1 general',numpy.complex128)
                                 ,('lambda2 general',numpy.complex128)
                                 ,('lambda1 domainWall',numpy.complex128)
                                 ,('lambda2 domainWall',numpy.complex128)
                                 ,('timeSlices',numpy.int64)
                                 ,('latticePoints',numpy.int64)
                                 ,('groupSize',numpy.int64)
                                 ,('edgeLength x', numpy.int64)
                                 ,('edgeLength y', numpy.int64)
                                 ,('domainWall', numpy.int64)
                                 ])
#}}}

def calcRatios(l,i,spacetime,paramDict,upState,downState,which): #{{{
  s = spacetime[l,i]
  gup = upState['G'][i,i]
  gdn = downState['G'][i,i]
  if  which == 'main':
    spinUp = paramDict['spinUp']
    spinDn = paramDict['spinDn']
    if i in paramDict['domainWall indices']:
      lmbd = paramDict['lambda1 domainWall']
    else:
      lmbd = paramDict['lambda1 general']
  elif which == 'other':
    spinUp = paramDict['spinUp_other']
    spinDn = paramDict['spinDn_other']
    if i in paramDict['domainWall indices']:
      lmbd = paramDict['lambda2 domainWall']
    else:
      lmbd = paramDict['lambda2 general']
  else: 
    raise ParameterError("Unknown parameter: {0}".format(which))

  val_up = {}
  val_up['delta'] = numpy.exp(-2*(spinUp)*lmbd*s)
  val_up['gamma'] = val_up['delta'] - 1
  val_up['det'] = 1 + (1 - gup) * val_up['gamma']

  val_dn = {}
  val_dn['delta'] = numpy.exp(-2*(spinDn)*lmbd*s)
  val_dn['gamma'] = val_dn['delta'] - 1
  val_dn['det'] = 1 + (1 - gdn) * val_dn['gamma']

  #detTot = val_up['det'] * val_dn['det'] * numpy.exp(-1*(spinUp+spinDn) * s * lmbd)
  detTot = val_up['det'] * val_dn['det'] * numpy.exp((spinUp+spinDn) * s * lmbd)
  return val_up, val_dn, detTot #}}}

def sweep(paramDict,sliceGroups,spacetime_1,spacetime_2,weightPhase,upState,downState): # Sweep and measure all time slices, return measurements and new sign/Greens' functions {{{
# Sweep over time slices and lattice sites.
# Time slices are iterated in the direction [L,L-1,...,1],
# as per M = 1 + B(1) B(2) ... B(L-1) B(L).

  degeneracy = {'up': {'value': 0.0, 'old element': 0.0, 'new element': 0}
               ,'down': {'value' 0.0, 'old element': 0.0, 'new element': 0}

# Unwrap the necessary parameters
  L = paramDict['L']
  N = paramDict['N']

  m = paramDict['m']
  useLambda2 = paramDict['useLambda2']

  spinUp = paramDict['spinUp']
  spinDn = paramDict['spinDn']

  spinUp_other = paramDict['spinUp_other']
  spinDn_other = paramDict['spinDn_other']

  expK = paramDict['expK']

  sliceCount = 0
  no_slices = len(sliceGroups)

  if useLambda2:
    phases = numpy.empty(2*L*N,dtype=numpy.complex128)
  else:
    phases = numpy.empty(L*N,dtype=numpy.complex128)
  countPhase = 0

  accepted = {"field 1": 0, "field 2": 0}

  for sliceGroup in sliceGroups:
    for l in sliceGroup:
      for i in range(N):
        # Sweep over the first Ising field {{{
        val_up_1, val_dn_1, detTot_1 = calcRatios( l, i, spacetime_1, paramDict, upState, downState, 'main' )
        #saveGup = numpy.copy(Gup)
        if checkFlip(abs(detTot_1)):
          spacetime_1[l,i] *= -1
          upState['expVs'][l,i,i]   *= val_up_1['delta']
          downState['expVs'][l,i,i] *= val_dn_1['delta']
          upState['G']   = updateGreensV(i,paramDict,upState,val_up_1)
          downState['G'] = updateGreensV(i,paramDict,downState,val_dn_1)
          weightPhase *= phase(detTot_1)
          accepted["field 1"] += 1
        phases[countPhase] = weightPhase
        countPhase += 1 #}}}

#        detUp = initGreens(True,paramDict,upState['expVs'],sliceGroups)[0]
#        detDn = initGreens(True,paramDict,downState['expVs'],sliceGroups)[0]
#        newWeight = detUp*detDn*numpy.exp((-1)*lambda2*numpy.sum(spacetime_2))
#        print("Track: {0}; reset: {1}".format(weight,newWeight))

        if useLambda2: # Sweep over the second Ising field {{{
          val_up_2, val_dn_2, detTot_2 = calcRatios( l, i, spacetime_2, paramDict, upState, downState, 'other' )
          if checkFlip(abs(detTot_2)):
            spacetime_2[l,i] *= -1
            upState['expVs'][l,i,i]   *= val_up_2['delta']
            downState['expVs'][l,i,i] *= val_dn_2['delta']
            upState['G']   = updateGreensV(i,paramDict,upState,val_up_2)
            downState['G'] = updateGreensV(i,paramDict,downState,val_dn_2)
            weightPhase *= phase(detTot_2)
            accepted["field 2"] += 1
          phases[countPhase] = weightPhase
          countPhase += 1 #}}}

#          detUp = initGreens(True,paramDict,upState['expVs'],sliceGroups)[0]
#          detDn = initGreens(True,paramDict,downState['expVs'],sliceGroups)[0]
#          newWeight = detUp*detDn*numpy.exp((-1)*lambda2*numpy.sum(spacetime_2))
#          print("Track: {0}; reset: {1}".format(weight,newWeight))

      upState['G']   = wrapGreens(expK,l,upState)
      downState['G'] = wrapGreens(expK,l,downState)

    sliceCount += 1
    if sliceCount < no_slices: # Functions terminates after the last sliceGroup has been treated
      Gup_old = numpy.copy(upState['G'])
      Gdn_old = numpy.copy(downState['G'])
      makeGreensParts(False,paramDict,upState,sliceCount,sliceGroups)
      makeGreensParts(False,paramDict,downState,sliceCount,sliceGroups)

      #degUp,gUp_old,gUp_new = getGreensMaximumDegeneracy(degUp,gUp_old,gUp_new,upState['G'],Gup_old)
      #degDn,gDn_old,gDn_new = getGreensMaximumDegeneracy(degDn,gDn_old,gDn_new,downState['G'],Gdn_old)
      degeneracy['up']   =  greensDegeneracy(degeneracy['up'], Gup_old, upState['G'])
      degeneracy['down'] =  greensDegeneracy(degeneracy['down'], Gdn_old, downState['G'])
  return degeneracy,phases,accepted
#}}}

def makeLoggingFile(outputName): #{{{
  saveName = outputName
  path,basename = osp.split(outputName)
  head,tail = osp.splitext(basename)
  triedNames = []
  for n in count():
    try:
      basename = "{0}-{1}.log".format(head,n)
      outputName = osp.join(path,basename)
      outputHandle = open(outputName, 'x')
    except OSError as oe: 
      triedNames.append(basename)
      pass
    else:
      outputHandle.close()
      break
  return outputName,triedNames #}}}

def makeOutputFile(outputName): #{{{
  saveName = outputName
  path,basename = osp.split(outputName)
  head,tail = osp.splitext(basename)
  triedNames = []
  for n in count():
    try:
      basename = "{0}-{1}.h5".format(head,n)
      outputName = osp.join(path,basename)
      outputHandle = h5py.File(outputName, 'w-')
    except OSError as oe: 
      triedNames.append(basename)
      pass
    else:
      if not len(triedNames) == 0:
        logging.debug("Output file(s) exist: {0}".format(triedNames))
      logging.info("Output file name: {0}".format(outputName))
      break
  return outputHandle #}}}

def finalizeSimulation(paramDict,outputName,record_phases,record_field_1,record_field_2): #{{{
  measurementSteps = paramDict['measurementSteps']
  beta = paramDict['beta']
  tn = paramDict['tn']
  tnn = paramDict['tnn']
  U = paramDict['U']
  mu = paramDict['mu']
  B = paramDict['B']
  dtau = paramDict['dtau']
  lambda1_general = paramDict['lambda1 general']
  lambda2_general = paramDict['lambda2 general']
  lambda1_domainWall = paramDict['lambda1 domainWall']
  lambda2_domainWall = paramDict['lambda2 domainWall']
  edgeLength_x = paramDict['edgeLength x']
  edgeLength_y = paramDict['edgeLength y']
  m = paramDict['m']

  N = paramDict['N']
  L = paramDict['L']

  parameters = numpy.array( (beta,t,U,mu,B,dtau
                            ,lambda1_general,lambda2_general
                            ,lambda1_domainWall,lambda2_domainWall
                            ,L,N,m
                            ,edgeLength_x, edgeLength_y
                            ,domainWall
                            ), dtype=parameter_dtype)

  #logging.info("Parameters: {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}".format(beta, t, U, mu, B, dtau, lambda1, lambda2))

  logging.info("Simulation parameters:")
  for field in parameters.dtype.names:
    if type(parameters[field][()]) == numpy.bytes_ :
      logging.info("{0}: {1}".format(field, parameters[field][()].decode() ))
    else:
      logging.info("{0}: {1}".format(field, parameters[field][()] ))

  outputFile = makeOutputFile(outputName)

  outputFile.create_dataset("parameters", data=parameters)
  outputFile.create_dataset("phases", compression='lzf', data=record_phases)
  outputFile.create_dataset("field 1", compression='lzf', data=record_field_1)
  outputFile.create_dataset("field 2", compression='lzf', data=record_field_2)

  outputFile.close()
  #}}}

def runSimulation(paramDict, sliceGroups, spacetime_1, spacetime_2, weightPhase, upState, downState): #{{{
# Allocate space for the measurements
  no_therm = paramDict['thermalizationSteps']
  no_meas = paramDict['measurementSteps']
  L = paramDict['L']
  N = paramDict['N']
  useLambda2 = paramDict['useLambda2']

  #measurements = numpy.empty(no_meas,dtype=measurement_dtype)
 
  sweepLength = N*L
  if paramDict['useLambda2']:
    sweepLength *= 2

  record_phases = numpy.empty(no_meas*sweepLength, dtype=numpy.complex128)
  record_field_1 = numpy.empty((no_meas,L,N), dtype=numpy.float64)
  record_field_2 = numpy.empty((no_meas,L,N), dtype=numpy.float64)

  degUp_save = 0.0
  degDn_save = 0.0
  gUp_old_save = 0.0
  gUp_new_save = 0.0
  gDn_old_save = 0.0
  gDn_new_save = 0.0

  totalAccepted = {'field 1':0, 'field 2':0}

  for i in range(no_therm):
    degeneracies,phases,accepted = sweep(paramDict,sliceGroups,spacetime_1,spacetime_2,weightPhase,upState,downState)
    upState   = initGreens(False,paramDict,upState['expVs'],sliceGroups)[1]
    downState = initGreens(False,paramDict,downState['expVs'],sliceGroups)[1]
    weightPhase = phases[-1]
    if degeneracies['up']['value'] > degUp_save:
      degUp_save = degeneracies['up']['value']
      gUp_old_save = degeneracies['up']['old element']
      gUp_new_save = degeneracies['up']['new element']
    if degeneracies['down']['value'] > degDn_save:
      degDn_save = degeneracies['down']['value']
      gDn_old_save = degeneracies['down']['old element']
      gDn_new_save = degeneracies['down']['new element']

# Ensure there is at least one step grouped
  tenth = int(no_meas/10)
  if tenth < 1:
    tenth = 1
  measGroups = list(grouper(tenth,range(no_meas)))

  startTime = time.time()
  startTime_format = time.localtime(startTime)

  logging.info("Measurement sweeps started on {0}.".format(time.strftime("%d %b %Y at %H:%M:%S, %z",startTime_format)))
  
  lastTime = startTime
  percentDone = 0

  for ms in measGroups: #{{{ Measurement sweeps

    for i in ms:
      degUp,gUp_old,gUp_new,degDn,gDn_old,gDn_new,phases,accepted = sweep(paramDict,sliceGroups,spacetime_1,spacetime_2,weightPhase,upState,downState)
      totalAccepted['field 1'] += accepted['field 1']
      totalAccepted['field 2'] += accepted['field 2']
      if degUp > degUp_save:
        degUp_save   = degUp
        gUp_old_save = gUp_old
        gUp_new_save = gUp_new

      if degDn > degDn_save:
        degDn_save   = degDn
        gDn_old_save = gDn_old
        gDn_new_save = gDn_new

      # Reset Green's functions and weights
      phaseUp,upState   = initGreens(True,paramDict,upState['expVs'],sliceGroups)
      phaseDn,downState = initGreens(True,paramDict,downState['expVs'],sliceGroups)
      weightPhase       = phases[-1]

      expFactor_general    = numpy.exp((-1) * paramDict['lambda2 general'] * numpy.sum( paramDict['lattice general'] * spacetime_2))
      expFactor_domainWall = numpy.exp((-1) * paramDict['lambda2 domainWall'] * numpy.sum( paramDict['lattice domainWall'] * spacetime_2))
      newWeightPhase       = phaseUp * phaseDn * phase( expFactor_general * expFactor_domainWall )

      #newWeightPhase    = phaseUp * phaseDn * phase(numpy.exp((-1)*paramDict['lambda2']*numpy.sum(spacetime_2)))

      record_phases[i*sweepLength:(i+1)*sweepLength] = phases
      record_field_1[i] = spacetime_1
      record_field_2[i] = spacetime_2

      relative_real = (newWeightPhase.real - weightPhase.real) / newWeightPhase.real * 100
      relative_imag = (newWeightPhase.imag - weightPhase.imag) / newWeightPhase.imag * 100
      if relative_real > 1 or relative_imag > 1:
        logging.warning("At the end of sweep {0}: phases from sweep and Green's reset differ by more than 1%.".format(i))
        logging.warning("Last weight in sweep: {0}.".format(weightPhase))
        logging.warning("From Green's reset:   {0}.".format(newWeightPhase))

      weightPhase = newWeightPhase

    percentDone += 1

    if percentDone > 0 and percentDone < 10:
      currentTime = time.time()
      delta_with_start = currentTime - startTime

      delta_with_last = currentTime - lastTime
      lastTime = currentTime

      remaining = 10 - percentDone
      delta_estimate = remaining * delta_with_last

      dtDone = str(timedelta(seconds=delta_with_start))
      dtTodo = str(timedelta(seconds=delta_estimate))

      logging.info("{0}% of sweeps completed in {1}, est. time remaining: {2}".format(percentDone*10,dtDone,dtTodo))

  #}}}

  endTime = time.time()
  endTime_format = time.localtime(endTime)
  deltaT = endTime - startTime

  #logging.info("Sweeps ended on {0}.".format(time.strftime("%d %b %Y at %H:%M:%S, %z",endTime_format)))
  logging.info("Monte Carlo sweeps finished in: {0}.".format(str(timedelta(seconds=deltaT))))
  logging.info("Average time per sweep: {0}.".format(str(timedelta(seconds=deltaT/no_meas))))
  totalSteps = no_meas*L*N
  #if useLambda2: # This was needed when the flip-count was bunching the 2 fields together
  #  totalSteps *= 2
  logging.info("Total number of accepted flips:")
  logging.info("Ising field 1: {0}. Number of tries: {1}. Rate: {2}".format(totalAccepted['field 1'],totalSteps,totalAccepted['field 1']/totalSteps))
  logging.info("Ising field 2: {0}. Number of tries: {1}. Rate: {2}".format(totalAccepted['field 2'],totalSteps,totalAccepted['field 2']/totalSteps))

  formDegUp = numpy.around(degUp_save*100,decimals=4)
  formDegDn = numpy.around(degDn_save*100,decimals=4)
  if degUp_save*100 > 0.1 or degDn_save*100 > 0.1:
    logging.warning("Maximum degeneracy in the Green's functions exceeded 0.1%.")
    logging.warning("Maximum degeneracy in Gup: {0} with elements (Gwrapped, Grecalc) = ({1}, {2}).".format(formDegUp,gUp_old_save,gUp_new_save))
    logging.warning("Maximum degeneracy in Gdn: {0} with elements (Gwrapped, Grecalc) = ({1}, {2}).".format(formDegDn,gDn_old_save,gDn_new_save))
  return record_phases, record_field_1, record_field_2
#}}}

def setupSimulation(configDict): # Fill the simulation parameter dictionary and construct all matrices {{{
  paramDict = configDict.copy()

  U = paramDict['U']
  idtau = paramDict['idtau']
  beta = paramDict['beta']
  lambda2_general = paramDict['lambda2 general']
  lambda2_domainWall = paramDict['lambda2 domainWall']

  dtau = 1/idtau
  m = floor(1.2*idtau)

  L = beta * idtau

  acosh_argument_general    = numpy.cosh(lambda2_general)    * numpy.exp(numpy.absolute(U) * dtau/2)
  acosh_argument_domainWall = numpy.cosh(lambda2_domainWall) * numpy.exp(numpy.absolute(U) * dtau/2)

  if acosh_argument_general < 1.0 and lambda2_general.real != 0:
    raise ValueError("For purely imaginary λ₂, general lattice: argument to arccosh in calculation of λ₁ smaller than 1.0: {}.".format(acosh_argument_general))
  else:
    lambda1_general = numpy.arccosh( acosh_argument_general )

  if acosh_argument_domainWall < 1.0 and lambda2_domainWall.real != 0:
    raise ValueError("For purely imaginary λ₂, domain wall: argument to arccosh in calculation of λ₁ smaller than 1.0: {}.".format(acosh_argument_domainWall))
  else:
    lambda1_domainWall = numpy.arccosh( acosh_argument_domainWall )

# Update the parameter dictionary
  paramDict['dtau'] = dtau
  paramDict['m'] = m
  paramDict['L'] = L
  paramDict['lambda1 general']    = lambda1_general
  paramDict['lambda1 domainWall'] = lambda1_domainWall

  if U < 0:
    paramDict['spinUp'] = +1
    paramDict['spinDn'] = +1
    paramDict['spinUp_other'] = +1
    paramDict['spinDn_other'] = -1
  else:
    paramDict['spinUp'] = +1
    paramDict['spinDn'] = -1
    paramDict['spinUp_other'] = +1
    paramDict['spinDn_other'] = +1

  sliceGroups = list(grouper(m,range(L)[::-1]))

  lattice_domainWall = [0] * N
  for i in paramDict['domainWall indices']:
    lattice_domainWall[i] = 1

  lattice_general = numpy.array([x^1 for x in lattice_domainWall])
  lattice_domainWall = numpy.array(lattice_domainWall)

  paramDict['lattice general']    = lattice_general
  paramDict['lattice domainWall'] = lattice_domainWall

  #spacetime_1,spacetime_2,weightPhase,upState,downState = makeHamiltonian(paramDict,sliceGroups)
  expK, spacetime_1,spacetime_2,expVs_up, expVs_dn = makeHamiltonian(paramDict)

  paramDict['expK'] = expK

  phaseUp,upState   = initGreens(True,paramDict,expVs_up,sliceGroups)
  phaseDn,downState = initGreens(True,paramDict,expVs_dn,sliceGroups)

  expFactor_general    = numpy.exp((-1) * lambda2_general    * numpy.sum( paramDict['lattice general'] * spacetime_2))
  expFactor_domainWall = numpy.exp((-1) * lambda2_domainWall * numpy.sum( paramDict['lattice domainWall'] * spacetime_2))

  weightPhase = phaseUp * phaseDn * phase( expFactor_general * expFactor_domainWall )

  logging.info("Maximum number of grouped/wrapped slices m = {0}.".format(m))
  return paramDict,sliceGroups,spacetime_1,spacetime_2,weightPhase,upState,downState #}}}

def startSimulation(configDict,outputName): #{{{
# Get all the relevenat values out of the dictionary
  try:
    paramDict,sliceGroups,spacetime_1,spacetime_2,weightPhase,upState,downState = setupSimulation(configDict)
  except ParameterError as perr:
    logging.error(perr)
  except ValueError as verr:
    logging.error(verr)
  else:
    try:
      record_phases,record_field_1,record_field_2 = runSimulation(paramDict,sliceGroups,spacetime_1,spacetime_2,weightPhase,upState,downState)
    except LinAlgError as lae:
      logging.error(lae)
    else:
      finalizeSimulation(paramDict,outputName,record_phases,record_field_1,record_field_2)
# }}}

def processConfig(config): #{{{ Process the configuration file
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

def getConfig(inputHandle):
  try:
    config = yaml.load(inputHandle)
  except yaml.YAMLError as exc:
    logging.error("Error in configuration file: {0}".format(exc))
    if hasattr(exc, 'problem_mark'):
      mark = exc.problem_mark
      logging.error("Error position: ({0}:{1})".format(mark.line-1, mark.column-1))
  else:
    inputHandle.close()
    configDict = processConfig(config)
    return configDict

def main(inputName,outputName): # Controls the entire simulation {{{
  #r.seed(439451005467937294)
  #npr.seed(8119087183917565057)
  try:
    inputHandle = open(inputName)
  except IOError as ioe:
    print(ioe)
  else:
    if outputName == None:
      outputName = inputName

    loggingFile,triedNames = makeLoggingFile(outputName) # If opening the file causes no problem, then the path should exist.
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(asctime)s - %(message)s',
                        filename=loggingFile,
                        filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(asctime)s - %(message)s')

    console.setFormatter(console_formatter)
    logging.getLogger().addHandler(console)

    logging.info("Path to configuration file: {0}".format(inputName))

    if not len(triedNames) == 0:
      logging.debug("Logging file(s) exist: {0}".format(triedNames))
    logging.info("Logging file name: {0}".format(loggingFile))

    try:
      configDict = getConfig(inputHandle)
    except IOError as ioe:
      logging.error(ioe)
    else:
      logging.info("Sets of simulation parameters found in configuration file: {0}".format(len(configDicts)))
      startSimulation(configDict,outputName)
#}}}

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("input", help="The simulation configuration input", type=str)
  parser.add_argument("-o", "--output", help="The output file; creates an output file based on the input file's name if not specified.", type=str)
  args = parser.parse_args()
  main(args.input, args.output)
  #main()
