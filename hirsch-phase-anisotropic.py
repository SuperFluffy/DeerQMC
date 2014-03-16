#!/usr/bin/python

import math
import numpy 
import scipy
import numpy.random as nr
import collections as coll
import itertools as itools
import scipy.linalg as sl

import sys
import logging
import yaml
import argparse
import time
import datetime

import os.path
import h5py

import itertools

import ast

def timing(func):
    def wrapper(*arg,**kw):
        t1 = time.time()
        res = func(*arg,**kw)
        t2 = time.time()
        return (t2-t1),res
    return wrapper

class ParameterError(Exception): # Custom exception {{{
  pass #}}}

# Construct dtype dtypes {{{
def construct_parameter_dtype(domainWall):
  parameter_dtype = numpy.dtype([('beta',numpy.float64)
                                 ,('t',numpy.float64)
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

#def grouper(n, iterable, fillvalue=None): # Group/slice iterables {{{
## Taken from http://docs.python.org/3.3/library/itertools.html#itertools-recipes
#    "Collect data into fixed-length chunks or blocks"
#    # grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
#    args = [iter(iterable)] * n
#    return zip_longest(*args, fillvalue=fillvalue)

def grouper(n,iterable): 
# From http://stackoverflow.com/questions/12185952/python-optimize-grouper-function-to-avoid-none-elements
  it = iter(iterable)
  return iter(lambda: tuple(itools.islice(it,n)), ()) #}}}

def compress_array(A): # Only chains of up to 64 elements
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

def UDR(A): # Calculate the UDR decomposition of a matrix {{{
  U,r = sl.qr(A)
  d = numpy.diagonal(r)
  R = r / d[:,numpy.newaxis]
  D = numpy.diagflat(d)
  return U,D,R #}}}

def RDU(A): # Calculate the RDU decomposition of a matrix {{{
  r,U = sl.rq(A)
  d = numpy.diagonal(r)
  R = r / d[numpy.newaxis,:]
  D = numpy.diagflat(d)
  return R,D,U #}}}

def calcSign(M): #{{{
  detM = sl.det(M)
  return numpy.sign(detM) #}}}

# Phase calculations {{{
def phase(z):
  return z/numpy.absolute(z)

def calcDeterminantPhase(M):
  detM = sl.det(M)
  return numpy.absolute(detM),phase(detM)
#}}}

# K, V initialization {{{
# Store array as (L,N), since the Python stores in row-major form -> faster access
def makeField(L,N,spinsSample=None):
  if sample == None:
    spinsSample = [-1,+1]
  randarray = nr.choice(spinsSample,size=N*L)
  spacetime = randarray.reshape(L,N)
  return spacetime

# 1D case
def makeKin1D(N):
  K = numpy.eye(N,k=1,dtype=numpy.float64)
  K += numpy.eye(N,k=-1,dtype=numpy.float64)
# Set the matrix elements to fulfil the PBC by hand. Has no effect on a 2-site chain
  if N>1:
    K[0,N-1] = 1
    K[N-1,0] = 1
  return K

# 2D case for symmetric square matrices
def makeKin2D(Nx,Ny):
  Kx = makeKin1D(Nx)
  Ix = numpy.eye(Nx,dtype=numpy.float64)

  Ky = makeKin1D(Ny)
  Iy = numpy.eye(Ny,dtype=numpy.float64)

  K = numpy.kron(Iy,Kx) + numpy.kron(Ky,Ix)
  return K

def makeDiagBilinear(N,c):
  D = c * numpy.eye(N,dtype=numpy.float64)
  return D

def makePotential(paramDict,C,M):
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

  expVs_up = numpy.array([sl.expm2(spinUp*v1 + spinUp_other * v2 + C + M) for (v1,v2) in zip(V1,V2)])
  expVs_dn = numpy.array([sl.expm2(spinDn*v1 + spinDn_other * v2 + C - M) for (v1,v2) in zip(V1,V2)])

  paramDict['lattice general']    = lattice_general
  paramDict['lattice domainWall'] = lattice_domainWall

  return spacetime_1,spacetime_2,expVs_up,expVs_dn
# }}}

def multiplySlicesStart(N,expK,expVs,order): # Multiplies “B_i”s in a given order from the head. {{{
  I = numpy.eye(N,dtype=numpy.complex128)
  B = numpy.eye(N,dtype=numpy.complex128)
  for l in order:
    B = numpy.dot(B,numpy.dot(expK,expVs[l]))
  return B #}}}

def multiplySlicesEnd(N,expK,expVs,order): # Multiplies “B_i”s in a given order from the tail. {{{
  I = numpy.eye(N,dtype=numpy.complex128)
  B = numpy.eye(N,dtype=numpy.complex128)
  for l in order:
    B = numpy.dot(numpy.dot(expK,expVs[l]),B)
  return B #}}}

def makeGreensUDR(getDeterminant,L,N,expK,expVs,i,m): # Returns a Green's function and the sign of the associated determinant {{{
  det = 0
  order = coll.deque(range(L))
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
  Uinv = sl.inv(U)
  Rinv = sl.inv(R)
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
  order = coll.deque(range(L))
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
  Uinv = sl.inv(U)
  Rinv = sl.inv(R)
  tmpMatrix = numpy.dot(Rinv,Uinv) + D
  r,D,u = RDU(tmpMatrix)
  U = numpy.dot(u,U)
  R = numpy.dot(R,r)
  if getDeterminant:
    detR = sl.det(R)
    detU = sl.det(U)
    detD = sl.det(D)
    det  = detR*detD*detU
  Rinv = sl.inv(R)
  Dinv = sl.inv(D)
  Uinv = sl.inv(U)
  G = numpy.dot(Uinv,numpy.dot(Dinv,Rinv))
  return det,G #}}}

def makeGreensNaive(getDeterminant,L,N,expK,expVs,i): # As makeGreensUDR, but without UDR decomposition {{{
  det = 0
  order = coll.deque(range(L))
  order.rotate(i)
  I = numpy.eye(N,dtype=numpy.complex128)
  A = numpy.eye(N,dtype=numpy.complex128)
  for o in order:
    B = numpy.dot(expK,expVs[o])
    A = numpy.dot(A,B)
  O = I + A
  if getDeterminant:
    det = sl.det(O)
  G = sl.inv(O)
  return det,G #}}}

def makeGreensParts(getDeterminant,paramDict,state,sliceCount,sliceGroups): #{{{
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
    RLinv = sl.inv(RL)
    ULinv = sl.inv(UL)
    tmpMatrix = numpy.dot(RLinv,ULinv) + DL
    rL,DL,uL = RDU(tmpMatrix)
    UL = numpy.dot(uL,UL)
    RL = numpy.dot(RL,rL)
    RLinv = sl.inv(RL)
    DLinv = sl.inv(DL)
    ULinv = sl.inv(UL)
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

    URinv = sl.inv(UR)
    ULinv = sl.inv(UL)
    tmpMatrix = numpy.dot(URinv,ULinv) + numpy.dot(DR,numpy.dot(RR,numpy.dot(RL,DL)))
    U,D,R = UDR(tmpMatrix)

    Rinv = sl.inv(R)
    Dinv = sl.inv(D)
    Uinv = sl.inv(U)
    state['G'] = numpy.dot(ULinv,numpy.dot(Rinv,numpy.dot(Dinv,numpy.dot(Uinv,URinv))))

  state['B_right'][sliceCount,0] = UR
  state['B_right'][sliceCount,1] = DR
  state['B_right'][sliceCount,2] = RR

  if getDeterminant:
    if sliceCount == 0:
      detUL = sl.det(UL)
      detDL = sl.det(DL)
      detRL = sl.det(RL)
      det   = detUL*detDL*detRL
    else:
      detUL = sl.det(UL)
      detUR = sl.det(UR)
      detD  = sl.det(D)
      detU  = sl.det(U)
      detR  = sl.det(R)
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
  Uinv = sl.inv(U)
  Rinv = sl.inv(R)
  tmpMatrix = numpy.dot(Rinv,Uinv) + D
  r,D,u = RDU(tmpMatrix)
  U = numpy.dot(u,U)
  R = numpy.dot(R,r)

  Rinv = sl.inv(R)
  Dinv = sl.inv(D)
  Uinv = sl.inv(U)
  G = numpy.dot(Uinv,numpy.dot(Dinv,Rinv))

  state = {'G': G
          ,'B_left': B_left
          ,'B_right': B_right
          ,'expVs': expVs}

  if getPhase:
    detR,phaseR = calcDeterminantPhase(R)
    detU,phaseU = calcDeterminantPhase(U)
    detD,phaseD = calcDeterminantPhase(D)
    phase       = phaseR*phaseU*phaseD

  return phase,state #}}}

def updateGreensLoop(i,paramDict,state,weightValues): #{{{
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

def updateGreensVect(i,paramDict,state,weightValues): #{{{
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

def updateGreens(i,paramDict,state,weightValues): #{{{ Update scheme chooser
  return updateGreensVect(i,paramDict,state,weightValues) #}}}

def wrapGreens(expK,l,state): # Propagate the Green's function to the next time slice {{{
  B    = numpy.dot(expK,state['expVs'][l])
  Binv = sl.inv(B)
  newG = numpy.dot(numpy.dot(B,state['G']),Binv)
  return newG #}}}

def checkFlip(detTot,gamma=None): #{{{
  detTot_abs = numpy.absolute(detTot)
  r = detTot_abs / (1+detTot_abs)
  p = nr.random()
  flip = False
  if p < r:
    flip = True
  return flip #}}}

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

def constructSystem(paramDict,sliceGroups): #{{{
  edgeLength_x = paramDict['edgeLength x']
  edgeLength_y = paramDict['edgeLength y']
  t = paramDict['t']
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

  K = makeKin2D(edgeLength_x,edgeLength_y)

  N = edgeLength_x * edgeLength_y

  paramDict['N'] = N

  C = makeDiagBilinear(N,dtau*mu)
  M = makeDiagBilinear(N,dtau*B)
  K *= (-dtau*t)
  expK = sl.expm2(-1*K)

  paramDict['expK'] = expK

  spacetime_1,spacetime_2,expVs_up,expVs_dn = makePotential(paramDict,C,M)

  phaseUp,upState   = initGreens(True,paramDict,expVs_up,sliceGroups)
  phaseDn,downState = initGreens(True,paramDict,expVs_dn,sliceGroups)

  expFactor_general    = numpy.exp((-1) * lambda2_general    * numpy.sum( paramDict['lattice general'] * spacetime_2))
  expFactor_domainWall = numpy.exp((-1) * lambda2_domainWall * numpy.sum( paramDict['lattice domainWall'] * spacetime_2))

  weightPhase = phaseUp * phaseDn * phase( expFactor_general * expFactor_domainWall )

  return spacetime_1,spacetime_2,weightPhase,upState,downState #}}}

def getGreensMaximumDegeneracy(deg,g_old,g_new,G1,G2): #{{{ Highest difference of an element in the last wrapped Green's function compared to the one calculated from scratch
  dims = G1.shape
  degMatrix = numpy.absolute((G2-G1)/G1)
  deg_index = numpy.argmax(degMatrix)
  deg_index = numpy.unravel_index(deg_index,dims)

  deg_new = degMatrix[deg_index]
  if deg_new > deg:
    return deg_new,G2[deg_index],G1[deg_index]
  else:
    return deg,g_old,g_new #}}}

def sweep(paramDict,sliceGroups,spacetime_1,spacetime_2,weightPhase,upState,downState): # Sweep and measure all time slices, return measurements and new sign/Greens' functions {{{
# Sweep over time slices and lattice sites.
# Time slices are iterated in the direction [L,L-1,...,1],
# as per M = 1 + B(1) B(2) ... B(L-1) B(L).
  degUp   = 0.0
  degDn   = 0.0
  gUp_old = 0.0
  gUp_new = 0.0
  gDn_old = 0.0
  gDn_new = 0.0

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
        if checkFlip(detTot_1):
          spacetime_1[l,i] *= -1
          upState['expVs'][l,i,i]   *= val_up_1['delta']
          downState['expVs'][l,i,i] *= val_dn_1['delta']
          upState['G']   = updateGreens(i,paramDict,upState,val_up_1)
          downState['G'] = updateGreens(i,paramDict,downState,val_dn_1)
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
          if checkFlip(detTot_2):
            spacetime_2[l,i] *= -1
            upState['expVs'][l,i,i]   *= val_up_2['delta']
            downState['expVs'][l,i,i] *= val_dn_2['delta']
            upState['G']   = updateGreens(i,paramDict,upState,val_up_2)
            downState['G'] = updateGreens(i,paramDict,downState,val_dn_2)
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

      degUp,gUp_old,gUp_new = getGreensMaximumDegeneracy(degUp,gUp_old,gUp_new,upState['G'],Gup_old)
      degDn,gDn_old,gDn_new = getGreensMaximumDegeneracy(degDn,gDn_old,gDn_new,downState['G'],Gdn_old)
  return degUp,gUp_old,gUp_new,degDn,gDn_old,gDn_new,phases,accepted
#}}}

def thermalized(measurements, tolerance=0.01): #{{{ Calculate whether thermalization was reached up to a certain tolerance
  results = [];
  totMean = measurements.mean();
  measIndex = scipy.linspace(0, measurements.size-1, measurements.size)

  fitted = scipy.polyval(scipy.polyfit(measIndex, measurements, 1), measIndex)  # fit measurements
  percentChange = (fitted[-1] - fitted[0])/totMean

  print(abs(percentChange))
  return abs(percentChange) < tolerance #}}}

def makeLoggingFile(outputName): #{{{
  saveName = outputName
  path,basename = os.path.split(outputName)
  head,tail = os.path.splitext(basename)
  triedNames = []
  for n in itertools.count():
    try:
      basename = "{0}-{1}.log".format(head,n)
      outputName = os.path.join(path,basename)
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
  path,basename = os.path.split(outputName)
  head,tail = os.path.splitext(basename)
  triedNames = []
  for n in itertools.count():
    try:
      basename = "{0}-{1}.h5".format(head,n)
      outputName = os.path.join(path,basename)
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

def finalizeSimulation(paramNo,paramDict,outputName,record_phases,record_field_1,record_field_2): #{{{
  measurementSteps = paramDict['measurementSteps']
  beta = paramDict['beta']
  t = paramDict['t']
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

  simGroup = outputFile.create_group(str(paramNo))
  simGroup.create_dataset("parameters", data=parameters)
  simGroup.create_dataset("phases", compression='lzf', data=record_phases)
  simGroup.create_dataset("field 1", compression='lzf', data=record_field_1)
  simGroup.create_dataset("field 2", compression='lzf', data=record_field_2)

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
    degUp,gUp_old,gUp_new,degDn,gDn_old,gDn_new,phases,accepted = sweep(paramDict,sliceGroups,spacetime_1,spacetime_2,weightPhase,upState,downState)
    upState = initGreens(False,paramDict,upState['expVs'],sliceGroups)[1]
    downState = initGreens(False,paramDict,downState['expVs'],sliceGroups)[1]
    weightPhase = phases[-1]
    if degUp > degUp_save:
      degUp_save = degUp
      gUp_old_save = gUp_old
      gUp_new_save = gUp_new
    if degDn > degDn_save:
      degDn_save = degDn
      gDn_old_save = gDn_old
      gDn_new_save = gDn_new

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

      dtDone = str(datetime.timedelta(seconds=delta_with_start))
      dtTodo = str(datetime.timedelta(seconds=delta_estimate))

      logging.info("{0}% of sweeps completed in {1}, est. time remaining: {2}".format(percentDone*10,dtDone,dtTodo))

  #}}}

  endTime = time.time()
  endTime_format = time.localtime(endTime)
  deltaT = endTime - startTime

  #logging.info("Sweeps ended on {0}.".format(time.strftime("%d %b %Y at %H:%M:%S, %z",endTime_format)))
  logging.info("Monte Carlo sweeps finished in: {0}.".format(str(datetime.timedelta(seconds=deltaT))))
  logging.info("Average time per sweep: {0}.".format(str(datetime.timedelta(seconds=deltaT/no_meas))))
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

def setupSimulation(configDict): # Fill the simulation parameter dictionary and construct all matrices
  paramDict = configDict.copy()

  U = paramDict['U']
  idtau = paramDict['idtau']
  beta = paramDict['beta']
  lambda2_general = paramDict['lambda2 general']
  lambda2_domainWall = paramDict['lambda2 domainWall']

  dtau = 1/idtau
  m = math.floor(1.2*idtau)

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

  spacetime_1,spacetime_2,weightPhase,upState,downState = constructSystem(paramDict,sliceGroups)

  logging.info("Maximum number of grouped/wrapped slices m = {0}.".format(m))
  return paramDict,sliceGroups,spacetime_1,spacetime_2,weightPhase,upState,downState

def startSimulation(paramNo,configDict,outputName): #{{{
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
    except sl.LinAlgError as lae:
      logging.error(lae)
    else:
      finalizeSimulation(paramNo,paramDict,outputName,record_phases,record_field_1,record_field_2)
# }}}

def makeParamElems(elemName, pDict, elemMult): #{{{ Create list of one parameter from configuration
  if pDict['valueType'] == 'range':
    start = pDict['start']
    end = pDict['end']
    step = pDict['step']
    res = pDict['resolution']
    elems = [step/res*elemMult for step in range(start, end+step,step)]
  elif pDict['valueType'] == 'list':
    elems = [step*elemMult for step in pDict['list'] ]
  else:
    logging.error('Not recognized value in {0}.'.format(elemName))
  return elems #}}}

def processConfig(config): #{{{ Process the configuration file
  sysConf = config['system']
  simConf = config['simulation']

  paramDict = {'t':                   sysConf['t']
              ,'U':                   sysConf['U']
              ,'edgeLength x':        sysConf['lattice']['edgeLength']['x']
              ,'edgeLength y':        sysConf['lattice']['edgeLength']['y']
              ,'useLambda2':          simConf['useLambda2']
              ,'thermalizationSteps': simConf['steps']['thermalization']
              ,'measurementSteps':    simConf['steps']['measurements']
              }

  muConf = sysConf['mu']
  muU    = sysConf['U'] if muConf['type'] == 'units of U' else 1
  mus    = makeParamElems('mu', muConf, muU)

  BConf  = sysConf['B']
  BU     = sysConf['U'] if BConf['type'] == 'units of U' else 1
  Bs     = makeParamElems('B', BConf, BU)

  betas  = makeParamElems('beta', sysConf['beta'], 1)
  idtaus = makeParamElems('idtau', sysConf['idtau'], 1)

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

  lambda2s_general_raw    = makeParamElems('lambda2 general', sysConf['lambda2']['general'], 1)
  lambda2s_domainWall_raw = makeParamElems('lambda2 domainWall', sysConf['lambda2']['domainWall'], 1)

  if sysConf['lambda2']['general']['complexForm'] == 'polar':
    angle = sysConf['lambda2']['general']['angle']
    rad = numpy.deg2rad(angle)
    lambda2s_general = [r * complex(numpy.cos(rad),numpy.sin(rad)) for r in lambda2s_general_raw]
  elif sysConf['lambda2']['general']['complexForm'] == 'rectangular':
    lambda2s_general = [complex(z) for z in lambda2s_general_raw]
  else:
    raise ValueError("Option '{}' for 'complexForm' in 'lambda2' not recognized.".format(sysConf['lambda2']['general']['complexForm']))

  if sysConf['lambda2']['domainWall']['complexForm'] == 'polar':
    angle = sysConf['lambda2']['domainWall']['angle']
    rad = numpy.deg2rad(angle)
    lambda2s_domainWall = [r * complex(numpy.cos(rad),numpy.sin(rad)) for r in lambda2s_domainWall_raw]
  elif sysConf['lambda2']['domainWall']['complexForm'] == 'rectangular':
    lambda2s_domainWall = [complex(z) for z in lambda2s_domainWall_raw]
  else:
    raise ValueError("Option '{}' for 'complexForm' in 'lambda2' not recognized.".format(sysConf['lambda2']['domainWall']['complexForm']))

  paramDicts = []
  for mu, B, beta, idtau, lambda2_general, lambda2_domainWall in itools.product(mus, Bs, betas, idtaus, lambda2s_general, lambda2s_domainWall):
    newpd                       = paramDict.copy()
    newpd['mu']                 = mu
    newpd['B']                  = B
    newpd['beta']               = beta
    newpd['idtau']              = idtau
    newpd['lambda2 general']    = lambda2_general
    newpd['lambda2 domainWall'] = lambda2_domainWall
    paramDicts.append(newpd)


  return paramDicts #}}}

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
    configDicts = processConfig(config)
    return configDicts

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
      configDicts = getConfig(inputHandle)
    except IOError as ioe:
      logging.error(ioe)
    else:
      paramNo = 1
      logging.info("Sets of simulation parameters found in configuration file: {0}".format(len(configDicts)))
      for configDict in configDicts:
        startSimulation(paramNo,configDict,outputName)
        paramNo += 1
#}}}

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("input", help="The simulation configuration input", type=str)
  parser.add_argument("-o", "--output", help="The output file; creates an output file based on the input file's name if not specified.", type=str)
  args = parser.parse_args()
  main(args.input, args.output)
  #main()
