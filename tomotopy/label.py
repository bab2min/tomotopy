"""
Submodule `tomotopy.label` provides automatic topic labeling techniques.
You can label topics automatically with simple code like below. The results are attached to the bottom of the code.

.. include:: ./auto_labeling_code.rst
"""

from _tomotopy import (_LabelCandidate, _LabelPMIExtractor, _LabelFoRelevance)

Candidate = _LabelCandidate
PMIExtractor = _LabelPMIExtractor
FoRelevance = _LabelFoRelevance
'''end of copy from pyc'''

import os
if os.environ.get('TOMOTOPY_LANG') == 'kr':
    __doc__ = """
`tomotopy.label` 서브모듈은 자동 토픽 라벨링 기법을 제공합니다.
아래에 나온 코드처럼 간단한 작업을 통해 토픽 모델의 결과에 이름을 붙일 수 있습니다. 그 결과는 코드 하단에 첨부되어 있습니다.

.. include:: ./auto_labeling_code.rst
"""
del os
