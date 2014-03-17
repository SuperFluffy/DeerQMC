import numpy 

from itertools import islice
from numpy.random import choice
from scipy.linalg import det, expm2, inv, qr, rq,

from collections import deque

__all__ = ['grouper', 'compress_array', 'UDR', 'RDU', 'calcSign', 'phase'
          ,'calcDeterminantPhase', 'makeField', 'makeKin1D', 'makeKin2D'
          ,'makeDiagBilinear', 'makePotential', 'multiplySlicesStart'
          ,'multiplySlicesEnd', 'makeGreensUDR', 'makeGreensRDU',
          ,'makeGreensNaive', 'timing', 'ParameterError']

class ParameterError(Exception): # Custom exception {{{
  pass #}}}

def timing(func):
    def wrapper(*arg,**kw):
        t1 = time.time()
        res = func(*arg,**kw)
        t2 = time.time()
        return (t2-t1),res
    return wrapper

def grouper(n,iterable): #{{{ Slices an iterable object into objects of size n
# From http://stackoverflow.com/questions/12185952/python-optimize-grouper-function-to-avoid-none-elements
  it = iter(iterable)
  return iter(lambda: tuple(islice(it,n)), ()) #}}}

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

def UDR(A): # Calculate the UDR decomposition of a matrix {{{
  U,r = qr(A)
  d = numpy.diagonal(r)
  R = r / d[:,numpy.newaxis]
  D = numpy.diagflat(d)
  return U,D,R #}}}

def RDU(A): # Calculate the RDU decomposition of a matrix {{{
  r,U = rq(A)
  d = numpy.diagonal(r)
  R = r / d[numpy.newaxis,:]
  D = numpy.diagflat(d)
  return R,D,U #}}}

def calcSign(M): #{{{
  detM = det(M)
  return numpy.sign(detM) #}}}

def phase(z): #{{{
  return z/numpy.absolute(z) #}}}

def calcDeterminantPhase(M): #{{{
  detM = det(M)
  return numpy.absolute(detM),phase(detM) #}}}

def makeField(L,N,spinsSample=None): #{{{
  if sample == None:
    spinsSample = [-1,+1]
  randarray = choice(spinsSample,size=N*L)
# Store array as (L,N), since the Python stores in row-major form -> faster access
  spacetime = randarray.reshape(L,N)
  return spacetime #}}}

def makeKin1D(n,k): # {{{
  K  = numpy.eye(n,k=+k,dtype=numpy.float64)
  K += numpy.eye(n,k=-k,dtype=numpy.float64)
# Set the matrix elements to fulfil the PBC by hand. Has no effect on a 2-site chain
  if n>k:
    K += numpy.eye(n,k=(n-k))
    K += numpy.eye(n,k=-(n-k))
  return K #}}}

def makeKin2D(nx,ny): # 2D hopping matrix for symmetric, square lattices {{{
  Kx = makeKin1D(nx)
  Ix = numpy.eye(nx,dtype=numpy.float64)

  Ky = makeKin1D(ny)
  Iy = numpy.eye(ny,dtype=numpy.float64)

  K = numpy.kron(Iy,Kx) + numpy.kron(Ky,Ix)
  return K #}}}

def makeDiagBilinear(N,c): #{{{
  D = c * numpy.eye(N,dtype=numpy.float64)
  return D #}}}

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
