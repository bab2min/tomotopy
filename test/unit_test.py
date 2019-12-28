import tomotopy as tp

model_cases = [
    (tp.LDAModel, 'test/sample.txt', 0, None, {'k':10}),
    (tp.LLDAModel, 'test/sample_with_md.txt', 0, None, {'k':5}),
    (tp.PLDAModel, 'test/sample_with_md.txt', 0, None, {'latent_topics':2, 'topics_per_label':2}),
	(tp.PLDAModel, 'test/sample_with_md.txt', 1, lambda x:x, {'latent_topics':2, 'topics_per_label':2}),
    (tp.HLDAModel, 'test/sample.txt', 0, None, {'depth':3}),
    (tp.CTModel, 'test/sample.txt', 0, None, {'k':10}),
    (tp.HDPModel, 'test/sample.txt', 0, None, {'initial_k':10}),
    (tp.MGLDAModel, 'test/sample.txt', 0, None, {'k_g':5, 'k_l':5}),
    (tp.PAModel, 'test/sample.txt', 0, None, {'k1':5, 'k2':10}),
    (tp.HPAModel, 'test/sample.txt', 0, None, {'k1':5, 'k2':10}),
    (tp.DMRModel, 'test/sample_with_md.txt', 1, lambda x:'_'.join(x), {'k':10}),
    (tp.SLDAModel, 'test/sample_with_md.txt', 1, lambda x:list(map(float, x)), {'k':10, 'vars':'b'}),
]

def train(cls, inputFile, mdFields, f, kargs):
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
    mdl.train(200)

def save_and_load(cls, inputFile, mdFields, f, kargs):
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
    mdl.train(20)
    mdl.save('test.model.{}.bin'.format(cls.__name__))
    mdl = cls.load('test.model.{}.bin'.format(cls.__name__))
    mdl.train(20)

def infer(cls, inputFile, mdFields, f, kargs):
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
    mdl.train(20)
    for n, line in enumerate(unseen_docs):
        if mdFields:
            unseen_docs[n] = mdl.make_doc(ch[mdFields:], f(ch[:mdFields]))
        else:
            unseen_docs[n] = mdl.make_doc(ch)

    mdl.infer(unseen_docs)


for model_case in model_cases:
    for func in [train, save_and_load, infer]:
        locals()['test_{}_{}'.format(model_case[0].__name__, func.__name__)] = (lambda f, mc: lambda: f(*mc))(func, model_case)
