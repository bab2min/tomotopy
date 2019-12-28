import tomotopy as tp

model_cases = [
    (tp.LDAModel, 'test/sample.txt', 0, None, {'k':10}, None),
    (tp.LLDAModel, 'test/sample_with_md.txt', 0, None, {'k':5}, None),
    (tp.PLDAModel, 'test/sample_with_md.txt', 0, None, {'latent_topics':2, 'topics_per_label':2}, None),
	(tp.PLDAModel, 'test/sample_with_md.txt', 1, lambda x:x, {'latent_topics':2, 'topics_per_label':2}, None),
    (tp.HLDAModel, 'test/sample.txt', 0, None, {'depth':3}, [tp.ParallelScheme.NONE]),
    (tp.CTModel, 'test/sample.txt', 0, None, {'k':10}, None),
    (tp.HDPModel, 'test/sample.txt', 0, None, {'initial_k':10}, [tp.ParallelScheme.COPY_MERGE]),
    (tp.MGLDAModel, 'test/sample.txt', 0, None, {'k_g':5, 'k_l':5}, None),
    (tp.PAModel, 'test/sample.txt', 0, None, {'k1':5, 'k2':10}, [tp.ParallelScheme.COPY_MERGE]),
    (tp.HPAModel, 'test/sample.txt', 0, None, {'k1':5, 'k2':10}, [tp.ParallelScheme.COPY_MERGE]),
    (tp.DMRModel, 'test/sample_with_md.txt', 1, lambda x:'_'.join(x), {'k':10}, None),
    (tp.SLDAModel, 'test/sample_with_md.txt', 1, lambda x:list(map(float, x)), {'k':10, 'vars':'b'}, None),
]

def train1(cls, inputFile, mdFields, f, kargs, ps):
    print('Test train')
    tw = 0
    print('Initialize model %s with TW=%s ...' % (str(cls), ['one', 'idf', 'pmi'][tw]))
    mdl = cls(tw=tw, min_cf=2, rm_top=2, **kargs)
    print('Adding docs...')
    for n, line in enumerate(open(inputFile, encoding='utf-8')):
        ch = line.strip().split()
        if len(ch) < mdFields + 1: continue
        if mdFields: mdl.add_doc(ch[mdFields:], f(ch[:mdFields]))
        else: mdl.add_doc(ch)
    mdl.train(200, workers=1, parallel=ps)

def train4(cls, inputFile, mdFields, f, kargs, ps):
    print('Test train')
    tw = 0
    print('Initialize model %s with TW=%s ...' % (str(cls), ['one', 'idf', 'pmi'][tw]))
    mdl = cls(tw=tw, min_cf=2, rm_top=2, **kargs)
    print('Adding docs...')
    for n, line in enumerate(open(inputFile, encoding='utf-8')):
        ch = line.strip().split()
        if len(ch) < mdFields + 1: continue
        if mdFields: mdl.add_doc(ch[mdFields:], f(ch[:mdFields]))
        else: mdl.add_doc(ch)
    mdl.train(200, workers=4, parallel=ps)

def train0(cls, inputFile, mdFields, f, kargs, ps):
    print('Test train')
    tw = 0
    print('Initialize model %s with TW=%s ...' % (str(cls), ['one', 'idf', 'pmi'][tw]))
    mdl = cls(tw=tw, min_cf=2, rm_top=2, **kargs)
    print('Adding docs...')
    for n, line in enumerate(open(inputFile, encoding='utf-8')):
        ch = line.strip().split()
        if len(ch) < mdFields + 1: continue
        if mdFields: mdl.add_doc(ch[mdFields:], f(ch[:mdFields]))
        else: mdl.add_doc(ch)
    mdl.train(200, parallel=ps)

def save_and_load(cls, inputFile, mdFields, f, kargs, ps):
    print('Test save & load')
    tw = 0
    print('Initialize model %s with TW=%s ...' % (str(cls), ['one', 'idf', 'pmi'][tw]))
    mdl = cls(tw=tw, min_cf=2, rm_top=2, **kargs)
    print('Adding docs...')
    for n, line in enumerate(open(inputFile, encoding='utf-8')):
        ch = line.strip().split()
        if len(ch) < mdFields + 1: continue
        if mdFields: mdl.add_doc(ch[mdFields:], f(ch[:mdFields]))
        else: mdl.add_doc(ch)
    mdl.train(20, parallel=ps)
    mdl.save('test.model.{}.bin'.format(cls.__name__))
    mdl = cls.load('test.model.{}.bin'.format(cls.__name__))
    mdl.train(20, parallel=ps)

def infer(cls, inputFile, mdFields, f, kargs, ps):
    print('Test infer')
    tw = 0
    print('Initialize model %s with TW=%s ...' % (str(cls), ['one', 'idf', 'pmi'][tw]))
    mdl = cls(tw=tw, min_cf=2, rm_top=2, **kargs)
    print('Adding docs...')
    unseen_docs = []
    for n, line in enumerate(open(inputFile, encoding='utf-8')):
        ch = line.strip().split()
        if len(ch) < mdFields + 1: continue
        if n < 20: unseen_docs.append(line)
        else:
            if mdFields:
                mdl.add_doc(ch[mdFields:], f(ch[:mdFields]))
            else:
                mdl.add_doc(ch)
    mdl.train(20, parallel=ps)
    for n, line in enumerate(unseen_docs):
        if mdFields:
            unseen_docs[n] = mdl.make_doc(ch[mdFields:], f(ch[:mdFields]))
        else:
            unseen_docs[n] = mdl.make_doc(ch)

    mdl.infer(unseen_docs, parallel=ps)


for model_case in model_cases:
    pss = model_case[5]
    if not pss: pss = [tp.ParallelScheme.COPY_MERGE, tp.ParallelScheme.PARTITION]
    for ps in pss:
        for func in [train1, train4, train0, save_and_load, infer]:
            locals()['test_{}_{}_{}'.format(model_case[0].__name__, func.__name__, ps.name)] = (lambda f, mc, ps: lambda: f(*mc, ps))(func, model_case[:-1], ps)
