"""
Python package `tomotopy` provides types and functions for various Topic Model 
including LDA, DMR, HDP, MG-LDA, PA and HPA. It is written in C++ for speed and provides Python extension.

.. include:: ./documentation.rst
"""
from enum import IntEnum

class TermWeight(IntEnum):
    """
    This enumeration is for Term Weighting Scheme and it is based on following paper:
    
    > * Wilson, A. T., & Chew, P. A. (2010, June). Term weighting schemes for latent dirichlet allocation. In human language technologies: The 2010 annual conference of the North American Chapter of the Association for Computational Linguistics (pp. 465-473). Association for Computational Linguistics.
    
    There are three options for term weighting and the basic one is ONE. The others also can be applied for all topic models in `tomotopy`. 
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
import os
if os.environ.get('TOMOTOPY_LANG') == 'kr':
    __doc__ = """`tomotopy` 패키지는 Python에서 사용가능한 다양한 토픽 모델링 타입과 함수를 제공합니다.
이 모듈은 c++로 작성되어 컴파일되기 때문에 빠른 속도를 자랑합니다.

.. include:: ./documentation.kr.rst
"""
    __pdoc__ = {}
    __pdoc__['isa'] = """현재 로드된 모듈이 어떤 SIMD 명령어 세트를 사용하는지 표시합니다. 
이 값은 `'avx2'`, `'avx'`, `'sse2'`, `'none'` 중 하나입니다."""
    __pdoc__['TermWeight'] = """용어 가중치 기법을 선택하는 데에 사용되는 열거형입니다. 여기에 제시된 용어 가중치 기법들은 다음 논문을 바탕으로 하였습니다:
    
> * Wilson, A. T., & Chew, P. A. (2010, June). Term weighting schemes for latent dirichlet allocation. In human language technologies: The 2010 annual conference of the North American Chapter of the Association for Computational Linguistics (pp. 465-473). Association for Computational Linguistics.

총 3가지 가중치 기법을 사용할 수 있으며 기본값은 ONE입니다. 기본값뿐만 아니라 다른 모든 기법들도 `tomotopy`의 모든 토픽 모델에 사용할 수 있습니다. """
    __pdoc__['TermWeight.ONE'] = """모든 용어를 동일하게 간주합니다. (기본값)"""
    __pdoc__['TermWeight.IDF'] = """역문헌빈도(IDF)를 가중치로 사용합니다.

따라서 모든 문헌에 거의 골고루 등장하는 용어의 경우 낮은 가중치를 가지게 되며, 
소수의 특정 문헌에만 집중적으로 등장하는 용어의 경우 높은 가중치를 가지게 됩니다."""
    __pdoc__['TermWeight.PMI'] = """점별 상호정보량(PMI)을 가중치로 사용합니다."""
del _load, IntEnum, os
