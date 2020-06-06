::

    import tomotopy as tp

    # This code requires nltk package for stemming.
    from nltk.stem.porter import PorterStemmer
    from nltk.corpus import stopwords

    stemmer = PorterStemmer()
    stopwords = set(stopwords.words('english'))
    corpus = tp.utils.Corpus(tokenizer=tp.utils.SimpleTokenizer(stemmer=stemmer.stem), 
        stopwords=lambda x: len(x) <= 2 or x in stopwords)
    # data_feeder yields a tuple of (raw string, user data) or a str (raw string)
    corpus.process(open(input_file, encoding='utf-8'))

    # make LDA model and train
    mdl = tp.LDAModel(k=20, min_cf=10, min_df=5, corpus=corpus)
    mdl.train(0)
    print('Num docs:', len(mdl.docs), ', Vocab size:', len(mdl.used_vocabs), ', Num words:', mdl.num_words)
    print('Removed top words:', mdl.removed_top_words)
    for i in range(0, 1000, 10):
        mdl.train(10)
        print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))
    
    # extract candidates for auto topic labeling
    extractor = tp.label.PMIExtractor(min_cf=10, min_df=5, max_len=5, max_cand=10000)
    cands = extractor.extract(mdl)

    labeler = tp.label.FoRelevance(mdl, cands, min_df=5, smoothing=1e-2, mu=0.25)
    for k in range(mdl.k):
        print("== Topic #{} ==".format(k))
        print("Labels:", ', '.join(label for label, score in labeler.get_topic_labels(k, top_n=5)))
        for word, prob in mdl.get_topic_words(k, top_n=10):
            print(word, prob, sep='\t')
        print()
    
    # Example of Results
    # -----------------
    # == Topic #13 ==
    # Labels: weapon systems, weaponry, anti-aircraft, towed, long-range
    # aircraft        0.020458335056900978
    # use     0.019993379712104797
    # airlin  0.012523100711405277
    # car     0.012058146297931671
    # vehicl  0.01165518444031477
    # carrier 0.011531196534633636
    # tank    0.011221226304769516
    # design  0.010694277472794056
    # audi    0.010322313755750656
    # martin  0.009981346316635609
    # 
    # == Topic #17 ==
    # Labels: American baseball player, American baseball, American actress, singer-songwriter and guitarist, American actor, director, producer, and screenwriter
    # american        0.04471408948302269
    # english 0.01746685802936554
    # player  0.01714528724551201
    # politician      0.014698212035000324
    # footbal 0.012313882820308208
    # author  0.010909952223300934
    # actor   0.008949155919253826
    # french  0.007647186517715454
    # academ  0.0073020863346755505
    # produc  0.006815808825194836
    # 
