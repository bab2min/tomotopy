"""
Python package `tomotopy` provides types and functions for various topic models
including LDA, DMR, HDP, MG-LDA, PA and HPA. It is written in C++ for speed and provides a Python extension.

.. include:: ./documentation.rst
"""
from tomotopy._version import __version__
from enum import IntEnum

class TermWeight(IntEnum):
    """
    This enumeration is for Term Weighting Scheme and it is based on the following paper:
    
    > * Wilson, A. T., & Chew, P. A. (2010, June). Term weighting schemes for latent dirichlet allocation. In human language technologies: The 2010 annual conference of the North American Chapter of the Association for Computational Linguistics (pp. 465-473). Association for Computational Linguistics.
    
    There are three options for term weighting and the basic one is ONE. The others can also be applied to all topic models in `tomotopy`. 
    """

    ONE = 0
    """ Consider every term equal (default)"""

    IDF = 1
    """ 
    Use Inverse Document Frequency term weighting.
    
    Thus, a term occurring in almost every document has very low weighting
    and a term occurring in a few documents has high weighting. 
    """

    PMI = 2
    """
    Use Pointwise Mutual Information term weighting.
    """

class ParallelScheme(IntEnum):
    """
    This enumeration is for Parallelizing Scheme:
    There are three options for parallelizing and the basic one is DEFAULT. Not all models support all options. 
    """

    DEFAULT = 0
    """tomotopy chooses the best available parallelism scheme for your model"""

    NONE = 1
    """ 
    Turn off multi-threading for Gibbs sampling at training or inference. Operations other than Gibbs sampling may use multithreading.
    """

    COPY_MERGE = 2
    """
    Use Copy and Merge algorithm from AD-LDA. It consumes RAM in proportion to the number of workers. 
    This has advantages when you have a small number of workers and a small number of topics and vocabulary sizes in the model.
    Prior to version 0.5, all models used this algorithm by default. 
    
    > * Newman, D., Asuncion, A., Smyth, P., & Welling, M. (2009). Distributed algorithms for topic models. Journal of Machine Learning Research, 10(Aug), 1801-1828.
    """

    PARTITION = 3
    """
    Use Partitioning algorithm from PCGS. It consumes only twice as much RAM as a single-threaded algorithm, regardless of the number of workers.
    This has advantages when you have a large number of workers or a large number of topics and vocabulary sizes in the model.
    
    > * Yan, F., Xu, N., & Qi, Y. (2009). Parallel inference for latent dirichlet allocation on graphics processing units. In Advances in neural information processing systems (pp. 2134-2142).
    """

import tomotopy.models as models

try:
    from _tomotopy import isa
except ImportError:
    isa = ''
"""
Indicates which SIMD instruction set is used for acceleration.
It can be one of `'avx512'`, `'avx2'`, `'sse2'` and `'none'`.
"""
import tomotopy.utils as utils
import tomotopy.coherence as coherence
import tomotopy.label as label
import tomotopy.viewer as viewer

from tomotopy.models import (
    LDAModel,
    DMRModel,
    GDMRModel,
    HDPModel,
    MGLDAModel,
    PAModel,
    HPAModel,
    CTModel,
    SLDAModel,
    LLDAModel,
    PLDAModel,
    HLDAModel,
    DTModel,
    PTModel,
)

def _get_all_model_types():
    types = []
    for name in dir(models):
        if name.endswith('Model') and not name.startswith('_'):
            types.append(getattr(models, name))
    return types

def load_model(path:str) -> 'LDAModel':
    '''
..versionadded:: 0.13.0

Load any topic model from the given file path.    

Parameters
----------
path : str
    The file path to load the model from.

Returns
-------
model : LDAModel or its subclass
    '''
    model_types = _get_all_model_types()
    for model_type in model_types:
        try:
            return model_type.load(path)
        except:
            pass
    raise ValueError(f'Cannot load model from {path}')

def loads_model(data:bytes) -> 'LDAModel':
    '''
..versionadded:: 0.13.0

Load any topic model from the given bytes data.

Parameters
----------
data : bytes
    The bytes data to load the model from.

Returns
-------
model : LDAModel or its subclass
    '''
    model_types = _get_all_model_types()
    for model_type in model_types:
        try:
            return model_type.loads(data)
        except:
            pass
    raise ValueError(f'Cannot load model from the given data')

import os as _os
if _os.environ.get('TOMOTOPY_LANG') == 'kr':
    from tomotopy._docs_ko import apply as _apply_ko
    _apply_ko()
    del _apply_ko
del IntEnum, _os
