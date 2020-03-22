import tomotopy as tp

model_cases = [
    (tp.LDAModel, 'test/sample.txt', 0, None, {'k':10}, None),
    (tp.LLDAModel, 'test/sample_with_md.txt', 1, lambda x:x, {'k':5}, None),
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

model_cases_raw = [
    (tp.LDAModel, 'test/sample_raw.txt', 0, None, {'k':10}, None),
    (tp.HLDAModel, 'test/sample_raw.txt', 0, None, {'depth':3}, [tp.ParallelScheme.NONE]),
    (tp.CTModel, 'test/sample_raw.txt', 0, None, {'k':10}, None),
    (tp.HDPModel, 'test/sample_raw.txt', 0, None, {'initial_k':10}, [tp.ParallelScheme.COPY_MERGE]),
    (tp.MGLDAModel, 'test/sample_raw.txt', 0, None, {'k_g':5, 'k_l':5}, None),
    (tp.PAModel, 'test/sample_raw.txt', 0, None, {'k1':5, 'k2':10}, [tp.ParallelScheme.COPY_MERGE]),
    (tp.HPAModel, 'test/sample_raw.txt', 0, None, {'k1':5, 'k2':10}, [tp.ParallelScheme.COPY_MERGE]),
]

model_cases_corpus = [
    (tp.LDAModel, 'test/sample.txt', 0, None, {'k':10}, None),
    (tp.LLDAModel, 'test/sample_with_md.txt', 1, lambda x:{'labels':x}, {'k':5}, None),
    (tp.PLDAModel, 'test/sample_with_md.txt', 0, None, {'latent_topics':2, 'topics_per_label':2}, None),
    (tp.PLDAModel, 'test/sample_with_md.txt', 1, lambda x:{'labels':x}, {'latent_topics':2, 'topics_per_label':2}, None),
    (tp.HLDAModel, 'test/sample.txt', 0, None, {'depth':3}, [tp.ParallelScheme.NONE]),
    (tp.CTModel, 'test/sample.txt', 0, None, {'k':10}, None),
    (tp.HDPModel, 'test/sample.txt', 0, None, {'initial_k':10}, [tp.ParallelScheme.COPY_MERGE]),
    (tp.MGLDAModel, 'test/sample.txt', 0, None, {'k_g':5, 'k_l':5}, None),
    (tp.PAModel, 'test/sample.txt', 0, None, {'k1':5, 'k2':10}, [tp.ParallelScheme.COPY_MERGE]),
    (tp.HPAModel, 'test/sample.txt', 0, None, {'k1':5, 'k2':10}, [tp.ParallelScheme.COPY_MERGE]),
    (tp.DMRModel, 'test/sample_with_md.txt', 1, lambda x:{'metadata':'_'.join(x)}, {'k':10}, None),
    (tp.SLDAModel, 'test/sample_with_md.txt', 1, lambda x:{'y':list(map(float, x))}, {'k':10, 'vars':'b'}, None),
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

def infer_together(cls, inputFile, mdFields, f, kargs, ps):
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

    mdl.infer(unseen_docs, parallel=ps, together=True)

def train_raw_corpus(cls, inputFile, mdFields, f, kargs, ps):
    print('Test train with raw corpus')
    from nltk.stem.porter import PorterStemmer
    from nltk.corpus import stopwords
    stemmer = PorterStemmer()
    stopwords = set(stopwords.words('english'))
    corpus = tp.utils.Corpus(tokenizer=tp.utils.SimpleTokenizer(stemmer=stemmer.stem), 
        stopwords=lambda x: len(x) <= 2 or x in stopwords)
    corpus.process(open(inputFile, encoding='utf-8'))
    mdl = cls(min_cf=2, rm_top=2, corpus=corpus, **kargs)
    mdl.train(100, parallel=ps)

def train_corpus(cls, inputFile, mdFields, f, kargs, ps):
    print('Test train with corpus')
    def feeder(file):
        for line in open(file, encoding='utf-8'):
            chs = line.split(None, maxsplit=mdFields)
            yield chs[-1], None, (f(chs[:mdFields]) if f else {})
    corpus = tp.utils.Corpus(tokenizer=tp.utils.SimpleTokenizer())
    corpus.process(feeder(inputFile))
    mdl = cls(min_cf=2, rm_top=2, corpus=corpus, **kargs)
    mdl.train(100, parallel=ps)

def test_estimate_SLDA_PARTITION(cls=tp.SLDAModel, inputFile='test/sample_with_md.txt', mdFields=1, f=lambda x:list(map(float, x)), kargs={'k':10, 'vars':'b'}, ps=tp.ParallelScheme.PARTITION):
    print('Test estimate')
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
        unseen_docs[n] = mdl.make_doc(ch)

    mdl.infer(unseen_docs, parallel=ps)
    mdl.estimate(unseen_docs)

def test_auto_labeling():
    from nltk.stem.porter import PorterStemmer
    from nltk.corpus import stopwords
    stemmer = PorterStemmer()
    stopwords = set(stopwords.words('english'))
    corpus = tp.utils.Corpus(tokenizer=tp.utils.SimpleTokenizer(stemmer=stemmer.stem), 
        stopwords=lambda x: len(x) <= 2 or x in stopwords)
    # data_feeder yields a tuple of (raw string, user data) or a str (raw string)
    corpus.process(open('test/sample_raw.txt', encoding='utf-8'))

    # make LDA model and train
    mdl = tp.LDAModel(k=10, min_cf=5, min_df=3, corpus=corpus)
    mdl.train(0)
    print('Num docs:', len(mdl.docs), ', Vocab size:', mdl.num_vocabs, ', Num words:', mdl.num_words)
    print('Removed top words:', mdl.removed_top_words)
    for i in range(0, 1000, 10):
        mdl.train(10)
        print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))
    
    # extract candidates for auto topic labeling
    extractor = tp.label.PMIExtractor(min_cf=5, min_df=3, max_len=5, max_cand=10000)
    cands = extractor.extract(mdl)

    labeler = tp.label.FoRelevance(mdl, cands, min_df=3, smoothing=1e-2, mu=0.25)
    for k in range(mdl.k):
        print("== Topic #{} ==".format(k))
        print("Labels:", ', '.join(label for label, score in labeler.get_topic_labels(k, top_n=5)))
        for word, prob in mdl.get_topic_words(k, top_n=10):
            print(word, prob, sep='\t')


for model_case in model_cases:
    pss = model_case[5]
    if not pss: pss = [tp.ParallelScheme.COPY_MERGE, tp.ParallelScheme.PARTITION]
    for ps in pss:
        for func in [train1, train4, train0, save_and_load, infer, infer_together]:
            locals()['test_{}_{}_{}'.format(model_case[0].__name__, func.__name__, ps.name)] = (lambda f, mc, ps: lambda: f(*(mc + (ps,))))(func, model_case[:-1], ps)

for model_case in model_cases_corpus:
    pss = model_case[5]
    if not pss: pss = [tp.ParallelScheme.COPY_MERGE, tp.ParallelScheme.PARTITION]
    for ps in pss:
        for func in [train_corpus]:
            locals()['test_{}_{}_{}'.format(model_case[0].__name__, func.__name__, ps.name)] = (lambda f, mc, ps: lambda: f(*(mc + (ps,))))(func, model_case[:-1], ps)

for model_case in model_cases_raw:
    pss = model_case[5]
    if not pss: pss = [tp.ParallelScheme.COPY_MERGE, tp.ParallelScheme.PARTITION]
    for ps in pss:
        for func in [train_raw_corpus]:
            locals()['test_{}_{}_{}'.format(model_case[0].__name__, func.__name__, ps.name)] = (lambda f, mc, ps: lambda: f(*(mc + (ps,))))(func, model_case[:-1], ps)