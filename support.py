import numpy 

from itertools import islice
from numpy.random import choice
from scipy.linalg import det, expm2, inv, qr, rq

from collections import deque

from math-functions import *
from helper import grouper

__all__ = ['checkFlip', 'compress_array', 'initGreens', 'makeField',
           'makeGreensParts', 'makeGreensRDU', 'makeGreensUDR', 'makeGreensNaive',
           'makeHopp1D', 'makeHopp2D', 'makePotential',
           'multiplySlicesStart', 'multiplySlicesEnd', 'thermalized',
           'updateGreensL', 'updateGreensV'
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

def makeGreensUDR(getDeterminant,L,N,expK,expVs,i,m): # Returns a Green's function and the sign of the associated determinant {{{
  det = 0
  order = deque(range(L))
  order.rotate(i)
  order.reverse() # Reverse the order so all the elements get multiplied from the right first
  orderChunks = grouper(m,order)
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
  orderChunks = grouper(m,order)

  orderChunks = list(orderChunks)
  numChunks = len(orderChunks)

  I = numpy.eye(N,dtype=numpy.complex128)
  U = numpy.copy(I)
  D = numpy.copy(I)
  R = numpy.copy(I)

  calcChunks = 0

# Create the partial matrix products and store them away (in reversed order)
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

def makeGreensNaive(getDeterminant,L,N,expK,expVs,i): # As makeGreensUDR, but without UDR decomposition {{{
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

def makeGreensParts(getDeterminant,paramDict,state,sliceCount,sliceGroups): # Updates the state of the simulation and returns the determinant {{{
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

def initGreens(getPhase,paramDict,expVs,sliceGroups): # Returns a Green's function and the sign of the associated determinant {{{
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

def updateGreensL(i,paramDict,state,weightValues): #{{{
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

def updateGreensV(i,paramDict,state,weightValues): #{{{
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

def wrapGreens(expK,l,state): # Propagate the Green's function to the next time slice {{{
  B    = numpy.dot(expK,state['expVs'][l])
  Binv = inv(B)
  newG = numpy.dot(numpy.dot(B,state['G']),Binv)
  return newG #}}}

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
