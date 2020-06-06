::

    import tomotopy as tp

    corpus = tp.utils.Corpus(tokenizer=tp.utils.SimpleTokenizer(), stopwords=['.'])
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

    # ranking the candidates of labels for a specific topic
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
    # Labels: american basebal, american actress, lawyer politician, race car driver, brown american
    # american        0.061747949570417404
    # english 0.02476435713469982
    # player  0.02357063814997673
    # politician      0.020087148994207382
    # footbal 0.016364915296435356
    # author  0.014303036034107208
    # actor   0.01202411763370037
    # french  0.009745198301970959
    # academ  0.009701790288090706
    # produc  0.008822779171168804
    # 
    # == Topic #16 ==
    # Labels: lunar, saturn, orbit moon, nasa report, orbit around
    # apollo  0.03052366152405739
    # star    0.017564402893185616
    # mission 0.015656694769859314
    # earth   0.01532777864485979
    # lunar   0.015130429528653622
    # moon    0.013683202676475048
    # orbit   0.011315013282001019
    # crew    0.01092031504958868
    # space   0.010821640491485596
    # nasa    0.009999352507293224
