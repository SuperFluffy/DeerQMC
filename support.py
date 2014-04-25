from numpy import dot,eye,float64,packbits,unique,zeros

from math import ceil
from math_functions import *

__all__ = ['checkFlip', 'compress_array',
           'multiply_slices_start', 'multiply_slices_end', 'thermalized',
           'ParameterError']

class ParameterError(Exception): # Custom exception {{{
  pass #}}}

def compress_array(A): # {{{
    """
    Compresses a binary array like [0,1,0,1] or ['a','b','a','b'] into an
    unsigned integer with padding, like 01010000, by setting the appropriate
    bits.
    """
    uniqueElements = unique(A)
    if uniqueElements.size > 2:
        raise ValueError("Array contains more than two unique elements: {0}".format(', '.join(map(str, uniqueElements))))
    else:
        needed_uints = ceil(A.size/64)
        B = zeros(needed_uints * 64,dtype='u1')
        B[ A==uniqueElements[0] ] = 0 # Translate the two unique elements in the
        B[ A==uniqueElements[1] ] = 1 # argument to 0, 1.
        D = packbits(B)
    return D.view('u8'), {uniqueElements[0]: 0, uniqueElements[1]: 1} #}}}

def multiply_slices_start(N,expK,expVs,order,dtype=float64): # Multiplies “B_i”s in a given order from the head. {{{
    B = eye(N,dtype=dtype)
    for l in order:
        B = dot(B,dot(expK,expVs[l]))
    return B #}}}

def multiply_slices_end(N,expK,expVs,order,dtype=float64): # Multiplies “B_i”s in a given order from the tail. {{{
    B = eye(N,dtype=dtype)
    for l in order:
        B = dot(dot(expK,expVs[l]),B)
    return B #}}}

def check_flip(p,gamma=None): #{{{
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
