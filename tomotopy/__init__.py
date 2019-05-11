import importlib
from cpuinfo import get_cpu_info
flags = get_cpu_info()['flags']
for isa in ['avx2', 'avx', 'sse2', '']:
  if not isa:
    from _tomotopy import *
    del importlib, get_cpu_info, flags
    break
  if isa in flags:
    try:
      vars().update({k:v for k, v in vars(importlib.import_module('_tomotopy_' + isa)).items() if not k.startswith('_')})
      del importlib, get_cpu_info, flags
      break
    except:
      pass
