import time
import tomotopy as tp
filename = 'enwiki-stemmed-1000.txt'

def bench_gensim(k):
    from gensim import corpora, models
    dictionary = corpora.Dictionary(filter(lambda x:x!='.', text.strip().split()) for text in open(filename, encoding='utf-8'))
    corpus = [dictionary.doc2bow(filter(lambda x:x!='.', text.strip().split())) for text in open(filename, encoding='utf-8')]
    #print('Number of vocabs:', len(dictionary))

    start_time = time.time()
    model = models.ldamodel.LdaModel(corpus, num_topics=k, id2word=dictionary, passes=10)
    #model = models.ldamulticore.LdaMulticore(corpus, num_topics=k, id2word=dictionary, passes=10, workers=8) # not work at Windows
    #for i in range(k): print(model.show_topic(i))
    print('K=%d\tTime: %.5g' % (k, time.time() - start_time), end='\t')
    print('LL: %g' % model.log_perplexity(corpus), flush=True)

def bench_tomotopy(k, ps, w=0):
    model = tp.LDAModel(k=k)
    for text in open(filename, encoding='utf-8'): model.add_doc(filter(lambda x:x!='.', text.strip().split()))
    #print('Number of vocabs:', len(model.vocabs))

    start_time = time.time()
    model.train(200, workers=w, parallel=ps)
    #for i in range(k): print(model.get_topic_words(i))
    print('K=%d\tW=%d\tTime: %.5g' % (k, w, time.time() - start_time), end='\t')
    print('LL: %g' % model.ll_per_word, flush=True)


print('== tomotopy (K x ParallelScheme) ==')
for ps in [tp.ParallelScheme.COPY_MERGE, tp.ParallelScheme.PARTITION]:
    print('= {} ='.format(ps.name))
    for k in range(10, 101, 10):
        bench_tomotopy(k, ps)
        time.sleep(2)

print('== tomotopy (Workers x ParallelScheme) ==')
for ps in [tp.ParallelScheme.COPY_MERGE, tp.ParallelScheme.PARTITION]:
    print('= {} ='.format(ps.name))
    for w in [1, 2, 3, 4, 5, 6, 7, 8]:
        bench_tomotopy(50, ps, w)
        time.sleep(2)

print('== gensim (K) ==')
for k in range(10, 101, 10):
    bench_gensim(k)
    time.sleep(2)
