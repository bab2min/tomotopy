
def call_method_bound(mdl, method:str, global_methods:dict, *args, **kwargs):
    for c in type(mdl).mro()[:-1]:
        cname = c.__name__
        try:
            return global_methods[method + '_' + cname](mdl, *args, **kwargs)
        except KeyError:
            pass
    raise KeyError(method + '_' + cname)
