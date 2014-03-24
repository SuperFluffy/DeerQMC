"""
This module defines some useful and frequently used mathematical/linear algebra
methods, which are used throughout the simulation.
"""

from scipy.linalg import qr, rq
from numpy import absolute, diagonal, diagflat, newaxis

__all__ = ['phase','determinant_phase', 'UDR', 'RDU']

def phase(z): #{{{
    return z/absolute(z) #}}}

def determinant_phase(M): # Return determinant and phase of a matrix M {{{
    """
    Efficient form of returning the absolute and phase of a matrix' determinant
    to avoid repeated calculation of the absolute in the phase-function.
    """
    detM = det(M)
    abs_detM = absolute(detM)
    return abs_detM, detM/abs_detM #}}}

def UDR(A): # Calculate the UDR decomposition of a matrix {{{
    U,r = qr(A)
    d = numpy.diagonal(r)
    R = r / d[:,newaxis]
    D = diagflat(d)
    return U,D,R #}}}

def RDU(A): # Calculate the RDU decomposition of a matrix {{{
    r,U = rq(A)
    d = numpy.diagonal(r)
    R = r / d[newaxis,:]
    D = diagflat(d)
    return R,D,U #}}}
