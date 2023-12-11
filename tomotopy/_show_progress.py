from tqdm import tqdm
from ._call_utils import call_method_bound

_tqdm_objs = {}

def init_tqdm_LDAModel(mdl, current_iteration:int, total_iteration:int):
    _tqdm_objs[mdl] = tqdm(total=total_iteration, desc='Iteration')

def close_tqdm_LDAModel(mdl, current_iteration:int, total_iteration:int):
    obj:tqdm = _tqdm_objs[mdl]
    obj.update(current_iteration - obj.n)
    obj.close()
    del _tqdm_objs[mdl]

def progress_LDAModel(mdl, current_iteration:int, total_iteration:int):
    obj:tqdm = _tqdm_objs[mdl]
    obj.set_postfix_str(f'LLPW: {mdl.ll_per_word:.6f}')
    obj.update(current_iteration - obj.n)

def progress_HDPModel(mdl, current_iteration:int, total_iteration:int):
    obj:tqdm = _tqdm_objs[mdl]
    obj.set_postfix_str(f'# Topics: {mdl.live_k}, LLPW: {mdl.ll_per_word:.6f}')
    obj.update(current_iteration - obj.n)

def progress_HLDAModel(mdl, current_iteration:int, total_iteration:int):
    obj:tqdm = _tqdm_objs[mdl]
    obj.set_postfix_str(f'# Topics: {mdl.live_k}, LLPW: {mdl.ll_per_word:.6f}')
    obj.update(current_iteration - obj.n)

def show_progress(mdl, current_iteration:int, total_iteration:int):
    if current_iteration == 0:
        call_method_bound(mdl, 'init_tqdm', globals(), current_iteration, total_iteration)
    elif current_iteration == total_iteration:
        call_method_bound(mdl, 'close_tqdm', globals(), current_iteration, total_iteration)
    else:
        call_method_bound(mdl, 'progress', globals(), current_iteration, total_iteration)
