"""
Some useful general functions.
"""

import numpy
import time
from itertools import islice, zip_longest

__all__ = ['timing', 'grouper', 'grouper_padded']

def timing(func): # Timing decorator {{{
    def wrapper(*arg,**kw):
        t1 = time.time()
        res = func(*arg,**kw)
        t2 = time.time()
        return (t2-t1),res
    return wrapper #}}}

def grouper(iterable,n): #{{{
   '''
    Collect data into fixed-length chunks or blocks
    grouper_padded('ABCDEFG', 3) --> ABC DEF G
    From http://stackoverflow.com/questions/12185952/python-optimize-grouper-function-to-avoid-none-elements
    '''
    it = iter(iterable)
    return iter(lambda: tuple(islice(it,n)), ()) #}}}

def grouper_padded(iterable, n, fillvalue=None): #{{{
    '''
    Collect data into fixed-length chunks or blocks
    grouper_padded('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    '''
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue) #}}

def maximumDegeneracy(A,B): # {{{
    """
    Calculates the element-wise degeneracy (relative difference) between elements
    of two matrices A, B and returns the index of the elements with largest degeneracy
    and the degeneracy.
    """
# Need to add some test between the types and shape of A, B.
    relativeMatrix = numpy.absolute((B-A)/A)
    ix = numpy.argmax(relativeMatrix)
    ix = numpy.unravel_index(deg_index,A.shape)
    return ix,relativeMatrix[ix] #}}}
