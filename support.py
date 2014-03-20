import numpy 

from itertools import islice
from numpy.random import choice
from scipy.linalg import expm2

from collections import deque

from math-functions import *
from helper import grouper

__all__ = ['checkFlip', 'compress_array', 'makeField',
           'makeHopp1D', 'makeHopp2D', 'makePotential',
           'multiplySlicesStart', 'multiplySlicesEnd', 'thermalized',
           'ParameterError']

class ParameterError(Exception): # Custom exception {{{
  pass #}}}

def compress_array(A): # Only chains of up to 64 elements {{{
  spinSpecies = [-1,+1] # Only two spin-species
  testSample = numpy.invert(numpy.in1d(A,spinsSample))
  ixWrong = numpy.where(testsample)[0]
  if A.size > 64:
    raise ValueError("Size of {0} of input array too large for compression.".format(A.size))
  elif ixWrong.size > 0:
    raise ValueError("Unknown spin species' in lattice: {0}".format(', '.join(map(', ', numpy.unique(A[ixWrong])))))
  else:
    B = numpy.copy(A)
    B[ A==(-1) ] = 0
    C = numpy.zeros(64,dtype='u8')
    C[0:(B.size)] = B
    D = numpy.packbits(C)
  return D[0]
#}}}

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

  lattice_domainWall = [0] * N
  for i in paramDict['domainWall indices']:
    lattice_domainWall[i] = 1

  lattice_general = numpy.array([x^1 for x in lattice_domainWall])
  lattice_domainWall = numpy.array(lattice_domainWall)

  V1  = lambda1_general    * numpy.array([numpy.diag(space) for space in (lattice_general * spacetime_1)],dtype=numpy.complex128)
  V1 += lambda1_domainWall * numpy.array([numpy.diag(space) for space in (lattice_domainWall * spacetime_1)],dtype=numpy.complex128)

  V2  = lambda2_general    * numpy.array([numpy.diag(space) for space in (lattice_general * spacetime_2)],dtype=numpy.complex128)
  V2 += lambda2_domainWall * numpy.array([numpy.diag(space) for space in (lattice_domainWall * spacetime_2)],dtype=numpy.complex128)

  expVs_up = numpy.array([expm2(spinUp*v1 + spinUp_other * v2 + C + M) for (v1,v2) in zip(V1,V2)])
  expVs_dn = numpy.array([expm2(spinDn*v1 + spinDn_other * v2 + C - M) for (v1,v2) in zip(V1,V2)])

  paramDict['lattice general']    = lattice_general
  paramDict['lattice domainWall'] = lattice_domainWall

  return spacetime_1,spacetime_2,expVs_up,expVs_dn
# }}}

def constructSystem(paramDict,sliceGroups): #{{{
  edgeLength_x = paramDict['edgeLength x']
  edgeLength_y = paramDict['edgeLength y']
  tn = paramDict['tn']
  tnn = paramDict['tnn']
  U = paramDict['U']
  mu = paramDict['mu']
  B = paramDict['B']
  dtau = paramDict['dtau']
  L = paramDict['L']
  m = paramDict['m']

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

  N = edgeLength_x * edgeLength_y

  C = (dtau*mu) * numpy.eye(N,dtype=numpy.float64)
  M = (dtau*B)  * numpy.eye(N,dtype=numpy.float64)

  paramDict['N'] = N
  paramDict['expK'] = expK

  spacetime_1,spacetime_2,expVs_up,expVs_dn = makePotential(paramDict,C,M)

  phaseUp,upState   = initGreens(True,paramDict,expVs_up,sliceGroups)
  phaseDn,downState = initGreens(True,paramDict,expVs_dn,sliceGroups)

  expFactor_general    = numpy.exp((-1) * lambda2_general    * numpy.sum( paramDict['lattice general'] * spacetime_2))
  expFactor_domainWall = numpy.exp((-1) * lambda2_domainWall * numpy.sum( paramDict['lattice domainWall'] * spacetime_2))

  weightPhase = phaseUp * phaseDn * phase( expFactor_general * expFactor_domainWall )

  return spacetime_1,spacetime_2,weightPhase,upState,downState #}}}

def multiplySlicesStart(N,expK,expVs,order): # Multiplies “B_i”s in a given order from the head. {{{
  B = numpy.eye(N,dtype=numpy.complex128)
  for l in order:
    B = numpy.dot(B,numpy.dot(expK,expVs[l]))
  return B #}}}

def multiplySlicesEnd(N,expK,expVs,order): # Multiplies “B_i”s in a given order from the tail. {{{
  B = numpy.eye(N,dtype=numpy.complex128)
  for l in order:
    B = numpy.dot(numpy.dot(expK,expVs[l]),B)
  return B #}}}

def checkFlip(p,gamma=None): #{{{
  r = p / (1+p)
  q = random()
  flip = False
  if q < r:
    flip = True
  return flip #}}}

def thermalized(measurements, tolerance=0.01): #{{{ Calculate whether thermalization was reached up to a certain tolerance
  results = [];
  totMean = measurements.mean();
  measIndex = linspace(0, measurements.size-1, measurements.size)

  fitted = polyval(polyfit(measIndex, measurements, 1), measIndex)  # fit measurements
  percentChange = (fitted[-1] - fitted[0])/totMean

  return abs(percentChange) < tolerance #}}}
