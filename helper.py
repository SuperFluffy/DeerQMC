"""
Some useful general functions.
"""

import numpy
import time
from itertools import islice

__all__ = ['timing', 'grouper']

def timing(func): # Timing decorator {{{
    def wrapper(*arg,**kw):
        t1 = time.time()
        res = func(*arg,**kw)
        t2 = time.time()
        return (t2-t1),res
    return wrapper #}}}

def grouper(n,iterable): #{{{ Slices an iterable object into objects of size n
# From http://stackoverflow.com/questions/12185952/python-optimize-grouper-function-to-avoid-none-elements
  it = iter(iterable)
  return iter(lambda: tuple(islice(it,n)), ()) #}}}
