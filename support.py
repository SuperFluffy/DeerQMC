import numpy 

from math-functions import *

__all__ = ['checkFlip', 'compress_array',
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
