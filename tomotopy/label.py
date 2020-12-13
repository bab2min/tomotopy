"""
Submodule `tomotopy.label` provides automatic topic labeling techniques.
You can label topics automatically with simple code like below. The results are attached to the bottom of the code.

.. include:: ./auto_labeling_code.rst
"""

def _load():
    import importlib, os
    env_setting = os.environ.get('TOMOTOPY_ISA', '').split(',')
    if not env_setting[0]: env_setting = []
    if len(env_setting) == 0 or len(env_setting) > 1:
        from cpuinfo import get_cpu_info
        flags = get_cpu_info()['flags']
    else:
        flags = []
    isas = ['avx2', 'avx', 'sse2', 'none']
    isas = [isa for isa in isas if (env_setting and isa in env_setting) or (not env_setting and (isa in flags or isa == 'none'))]
    if not isas: raise RuntimeError("No isa option for " + str(env_setting))
    for isa in isas:
        try:
            mod_name = '_tomotopy' + ('_' + isa if isa != 'none' else '')
            globals().update({k:v for k, v in vars(importlib.import_module(mod_name)).items() if k.startswith('_Label')})
            return
        except:
            if isa == isas[-1]: raise
_load()
del _load

Candidate = _LabelCandidate
PMIExtractor = _LabelPMIExtractor
FoRelevance = _LabelFoRelevance

import os
if os.environ.get('TOMOTOPY_LANG') == 'kr':
    __doc__ = """
`tomotopy.label` 서브모듈은 자동 토픽 라벨링 기법을 제공합니다.
아래에 나온 코드처럼 간단한 작업을 통해 토픽 모델의 결과에 이름을 붙일 수 있습니다. 그 결과는 코드 하단에 첨부되어 있습니다.

.. include:: ./auto_labeling_code.rst
"""
del os
