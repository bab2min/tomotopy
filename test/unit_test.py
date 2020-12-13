import tomotopy as tp
import sys

model_cases = [
    (tp.LDAModel, 'test/sample.txt', 0, None, {'k':40}, None),
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
    (tp.DTModel, 'test/sample_tp.txt', 1, lambda x:int(x[0]), {'k':10, 't':13}, None),
    (tp.GDMRModel, 'test/sample_tp.txt', 1, lambda x:list(map(float, x)), {'k':10, 'degrees':[3]}, None),
]

model_raw_cases = [
    (tp.LDAModel, 'test/sample_raw.txt', 0, None, {'k':10}, None),
    (tp.HLDAModel, 'test/sample_raw.txt', 0, None, {'depth':3}, [tp.ParallelScheme.NONE]),
    (tp.CTModel, 'test/sample_raw.txt', 0, None, {'k':10}, None),
    (tp.HDPModel, 'test/sample_raw.txt', 0, None, {'initial_k':10}, [tp.ParallelScheme.COPY_MERGE]),
    (tp.MGLDAModel, 'test/sample_raw.txt', 0, None, {'k_g':5, 'k_l':5}, None),
    (tp.PAModel, 'test/sample_raw.txt', 0, None, {'k1':5, 'k2':10}, [tp.ParallelScheme.COPY_MERGE]),
    (tp.HPAModel, 'test/sample_raw.txt', 0, None, {'k1':5, 'k2':10}, [tp.ParallelScheme.COPY_MERGE]),
]

model_corpus_cases = [
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
    (tp.DTModel, 'test/sample_tp.txt', 1, lambda x:{'timepoint':int(x[0])}, {'k':10, 't':13}, None),
    (tp.GDMRModel, 'test/sample_tp.txt', 1, lambda x:{'metadata':list(map(float, x))}, {'k':10, 'degrees':[3]}, None),
]

def train1(cls, inputFile, mdFields, f, kargs, ps):
    print('Test train')
    tw = 0
    print('Initialize model %s with TW=%s ...' % (str(cls), ['one', 'idf', 'pmi'][tw]))
    mdl = cls(tw=tw, min_df=2, rm_top=2, **kargs)
    print('Adding docs...')
    for n, line in enumerate(open(inputFile, encoding='utf-8')):
        ch = line.strip().split()
        if len(ch) < mdFields + 1: continue
        if mdFields: mdl.add_doc(ch[mdFields:], f(ch[:mdFields]))
        else: mdl.add_doc(ch)
    mdl.train(2000, workers=1, parallel=ps)

def train4(cls, inputFile, mdFields, f, kargs, ps):
    print('Test train')
    tw = 0
    print('Initialize model %s with TW=%s ...' % (str(cls), ['one', 'idf', 'pmi'][tw]))
    mdl = cls(tw=tw, min_df=2, rm_top=2, **kargs)
    print('Adding docs...')
    for n, line in enumerate(open(inputFile, encoding='utf-8')):
        ch = line.strip().split()
        if len(ch) < mdFields + 1: continue
        if mdFields: mdl.add_doc(ch[mdFields:], f(ch[:mdFields]))
        else: mdl.add_doc(ch)
    mdl.train(2000, workers=4, parallel=ps)

def train0(cls, inputFile, mdFields, f, kargs, ps):
    print('Test train')
    tw = 0
    print('Initialize model %s with TW=%s ...' % (str(cls), ['one', 'idf', 'pmi'][tw]))
    mdl = cls(tw=tw, min_df=2, rm_top=2, **kargs)
    print('Adding docs...')
    for n, line in enumerate(open(inputFile, encoding='utf-8')):
        ch = line.strip().split()
        if len(ch) < mdFields + 1: continue
        if mdFields: mdl.add_doc(ch[mdFields:], f(ch[:mdFields]))
        else: mdl.add_doc(ch)
    mdl.train(2000, parallel=ps)
    mdl.summary(file=sys.stderr)

def save_and_load(cls, inputFile, mdFields, f, kargs, ps):
    print('Test save & load')
    tw = 0
    print('Initialize model %s with TW=%s ...' % (str(cls), ['one', 'idf', 'pmi'][tw]))
    mdl = cls(tw=tw, min_df=2, rm_top=2, **kargs)
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
    mdl = cls(tw=tw, min_df=2, rm_top=2, **kargs)
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
        ch = line.strip().split()
        if mdFields:
            unseen_docs[n] = mdl.make_doc(ch[mdFields:], f(ch[:mdFields]))
        else:
            unseen_docs[n] = mdl.make_doc(ch)

    mdl.infer(unseen_docs, parallel=ps)

def infer_together(cls, inputFile, mdFields, f, kargs, ps):
    print('Test infer')
    tw = 0
    print('Initialize model %s with TW=%s ...' % (str(cls), ['one', 'idf', 'pmi'][tw]))
    mdl = cls(tw=tw, min_df=2, rm_top=2, **kargs)
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
        ch = line.strip().split()
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

def train_infer_raw_corpus(cls, inputFile, mdFields, f, kargs, ps):
    print('Test train with raw corpus')
    from nltk.stem.porter import PorterStemmer
    from nltk.corpus import stopwords
    stemmer = PorterStemmer()
    stopwords = set(stopwords.words('english'))
    corpus = tp.utils.Corpus(tokenizer=tp.utils.SimpleTokenizer(stemmer=stemmer.stem), 
        stopwords=lambda x: len(x) <= 2 or x in stopwords)
    corpus.process(open(inputFile, encoding='utf-8'))
    test_size = len(corpus) // 5
    test_set, train_set = corpus[:test_size], corpus[test_size:]
    mdl = cls(min_cf=2, rm_top=2, corpus=train_set, **kargs)
    mdl.train(100, parallel=ps)
    mdl.summary()

    result, ll = mdl.infer(test_set)
    print('infer ll:', ll)
    for r in result: print(r.get_ll(), *(r[i] for i in range(10)))

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

def train_multi_corpus(cls, inputFile, mdFields, f, kargs, ps):
    print('Test train with corpus')
    corpus1 = tp.utils.Corpus(tokenizer=tp.utils.SimpleTokenizer())
    corpus2 = tp.utils.Corpus(tokenizer=tp.utils.SimpleTokenizer())
    for i, line in enumerate(open(inputFile, encoding='utf-8')):
        chs = line.split(None, maxsplit=mdFields)
        (corpus1 if i < 10 else corpus2).add_doc(raw=chs[-1], **f(chs[:mdFields]) if f else {})
    mdl = cls(min_cf=2, rm_top=2, **kargs)
    tcorpus1 = mdl.add_corpus(corpus1)
    tcorpus2 = mdl.add_corpus(corpus2)
    mdl.train(100, parallel=ps)
    
    print('Corpus1')
    for d in tcorpus1: print(d)
    print()
    print('Corpus2')
    for d in tcorpus2: print(d)

def test_estimate_SLDA_PARTITION(cls=tp.SLDAModel, inputFile='test/sample_with_md.txt', mdFields=1, f=lambda x:list(map(float, x)), kargs={'k':10, 'vars':'b'}, ps=tp.ParallelScheme.PARTITION):
    print('Test estimate')
    tw = 0
    print('Initialize model %s with TW=%s ...' % (str(cls), ['one', 'idf', 'pmi'][tw]))
    mdl = cls(tw=tw, min_df=2, rm_top=2, **kargs)
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
    
    ngrams = corpus.extract_ngrams(min_cf=5, min_df=3)
    for c in ngrams:
        print(c)

    corpus.concat_ngrams(ngrams)

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

    labeler = tp.label.FoRelevance(mdl, cands, min_df=3, smoothing=1e-2, mu=0.25, workers=1)
    for k in range(mdl.k):
        print("== Topic #{} ==".format(k))
        print("Labels:", ', '.join(label for label, score in labeler.get_topic_labels(k, top_n=5)))
        for word, prob in mdl.get_topic_words(k, top_n=10):
            print(word, prob, sep='\t')

def test_docs():
    from nltk.stem.porter import PorterStemmer
    from nltk.corpus import stopwords
    stemmer = PorterStemmer()
    stopwords = set(stopwords.words('english'))
    corpus = tp.utils.Corpus(tokenizer=tp.utils.SimpleTokenizer(stemmer=stemmer.stem), 
        stopwords=lambda x: len(x) <= 2 or x in stopwords)
    # data_feeder yields a tuple of (raw string, user data) or a str (raw string)
    for i, line in enumerate(open('test/sample_raw.txt', encoding='utf-8')):
        corpus.add_doc(raw=line, uid='doc{:05}'.format(i), etc=len(line))

    def _test_doc(doc, etc=False):
        print("doc", doc)
    
        print("len(doc)", len(doc))
        print("doc.__getitem__", doc[0], doc[1], doc[2], doc[3])

        if etc: print("doc.etc", doc.etc)
        print("doc.words", doc.words)
        print("doc.span", doc.span)
        print("doc.raw", doc.raw)
    
    print("len(corpus)", len(corpus))
    print("len(corpus[:10])", len(corpus[:10]))

    _test_doc(corpus[0], etc=True)

    mdl = tp.LDAModel(k=10, corpus=corpus)
    mdl.train(100)
    print("len(mdl.docs)", len(mdl.docs))
    print("len(mdl.docs[:10])", len(mdl.docs[:10]))

    ch = tp.coherence.Coherence(corpus=mdl, coherence='u_mass')
    for k in range(mdl.k):
        print('Coherence of #{} : {}'.format(k, ch.get_score(topic_id=k)))

    _test_doc(mdl.docs[0])

def test_corpus_transform():
    from nltk.stem.porter import PorterStemmer
    from nltk.corpus import stopwords
    stemmer = PorterStemmer()
    stopwords = set(stopwords.words('english'))
    corpus = tp.utils.Corpus(tokenizer=tp.utils.SimpleTokenizer(stemmer=stemmer.stem), 
        stopwords=lambda x: len(x) <= 2 or x in stopwords)
    # data_feeder yields a tuple of (raw string, user data) or a str (raw string)
    for i, line in enumerate(open('test/sample_raw.txt', encoding='utf-8')):
        corpus.add_doc(raw=line, uid='doc{:05}'.format(i), metadata=line[0])
    
    def xform(misc):
        misc['metadata'] += '0'
        return misc
    mdl = tp.DMRModel(k=10, corpus=corpus, transform=xform)
    mdl.train(100)
    mdl.summary()

def test_hdp_to_lda():
    mdl = tp.HDPModel(tw=tp.TermWeight.ONE, min_df=5, rm_top=5, alpha=0.5, gamma=0.5, initial_k=5)
    for n, line in enumerate(open('test/sample.txt', encoding='utf-8')):
        ch = line.strip().split()
        mdl.add_doc(ch)
    mdl.burn_in = 100
    mdl.train(0)
    print('Num docs:', len(mdl.docs), ', Vocab size:', mdl.num_vocabs, ', Num words:', mdl.num_words)
    print('Removed top words:', mdl.removed_top_words)
    for i in range(0, 1000, 10):
        mdl.train(10)
        print('Iteration: {}\tLog-likelihood: {}\tNum. of topics: {}\tNum. of tables: {}'.format(i, mdl.ll_per_word, mdl.live_k, mdl.num_tables))

    lda, topic_mapping = mdl.convert_to_lda(topic_threshold=1e-3)
    print(topic_mapping)
    for i in range(0, 100, 10):
        lda.train(10)
        print('Iteration: {}\tLog-likelihood: {}'.format(i, lda.ll_per_word))

    for k in range(lda.k):
        print('Topic #{} ({})'.format(k, lda.get_count_by_topics()[k]))
        for word, prob in lda.get_topic_words(k):
            print('\t', word, prob, sep='\t')

def test_coherence():
    mdl = tp.LDAModel(tw=tp.TermWeight.ONE, k=20, min_df=5, rm_top=5)
    for n, line in enumerate(open('test/sample.txt', encoding='utf-8')):
        ch = line.strip().split()
        mdl.add_doc(ch)
    mdl.train(1000)

    for coh in ('u_mass', 'c_uci', 'c_npmi', 'c_v'):
        coherence = tp.coherence.Coherence(corpus=mdl, coherence=coh)
        print(coherence.get_score())


for model_case in model_cases:
    pss = model_case[5]
    if not pss: pss = [tp.ParallelScheme.COPY_MERGE, tp.ParallelScheme.PARTITION]
    for ps in pss:
        for func in [train1, train4, train0, 
            save_and_load, infer, infer_together
            ]:
            locals()['test_{}_{}_{}'.format(model_case[0].__name__, func.__name__, ps.name)] = (lambda f, mc, ps: lambda: f(*(mc + (ps,))))(func, model_case[:-1], ps)


for model_case in model_corpus_cases:
    pss = model_case[5]
    if not pss: pss = [tp.ParallelScheme.COPY_MERGE, tp.ParallelScheme.PARTITION]
    for ps in pss:
        for func in [train_corpus, train_multi_corpus]:
            locals()['test_{}_{}_{}'.format(model_case[0].__name__, func.__name__, ps.name)] = (lambda f, mc, ps: lambda: f(*(mc + (ps,))))(func, model_case[:-1], ps)


for model_case in model_raw_cases:
    pss = model_case[5]
    if not pss: pss = [tp.ParallelScheme.COPY_MERGE, tp.ParallelScheme.PARTITION]
    for ps in pss:
        for func in [train_raw_corpus, train_infer_raw_corpus]:
            locals()['test_{}_{}_{}'.format(model_case[0].__name__, func.__name__, ps.name)] = (lambda f, mc, ps: lambda: f(*(mc + (ps,))))(func, model_case[:-1], ps)
