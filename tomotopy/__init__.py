"""
Python package `tomotopy` provides types and functions for various Topic Model 
including LDA, DMR, HDP, MG-LDA, PA and HPA. It is written in C++ for speed and provides Python extension.

.. include:: ./documentation.md
"""
from enum import IntEnum

class TermWeight(IntEnum):
    """
    This enumeration is for Term Weighting Scheme and it is based on following paper:
    
    > * Wilson, A. T., & Chew, P. A. (2010, June). Term weighting schemes for latent dirichlet allocation. In human language technologies: The 2010 annual conference of the North American Chapter of the Association for Computational Linguistics (pp. 465-473). Association for Computational Linguistics.
    
    There are three options for term weighting and the basic one is ONE. The others also can be applied for every topic model in `tomotopy`. 
    """

    ONE = 0
    """ Consider every term equal (default)"""

    IDF = 1
    """ 
    Use Inverse Document Frequency term weighting.
    
    Thus, a term occurring at almost every document has very low weighting
    and a term occurring at a few document has high weighting. 
    """

    PMI = 2
    """
    Use Pointwise Mutual Information term weighting.
    """

isa = ''
"""
Indicate which SIMD instruction set is used for acceleration.
It can be one of `'avx2'`, `'avx'`, `'sse2'` and `'none'`.
"""

# This code is an autocomplete-hint for IDE.
# The object imported here will be overwritten by _load() function.
try: from _tomotopy import *
except: pass

def _load():
    import importlib, os
    from cpuinfo import get_cpu_info
    flags = get_cpu_info()['flags']
    env_setting = os.environ.get('TOMOTOPY_ISA', '').split(',')
    if not env_setting[0]: env_setting = []
    isas = ['avx2', 'avx', 'sse2', 'none']
    isas = [isa for isa in isas if (env_setting and isa in env_setting) or (not env_setting and (isa in flags or isa == 'none'))]
    if not isas: raise RuntimeError("No isa option for " + str(env_setting))
    for isa in isas:
        try:
            mod_name = '_tomotopy' + ('_' + isa if isa != 'none' else '')
            globals().update({k:v for k, v in vars(importlib.import_module(mod_name)).items() if not k.startswith('_')})
            return
        except:
            if isa == isas[-1]: raise
_load()
del _load, IntEnum
