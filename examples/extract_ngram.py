import sys
import tomotopy as tp

def extract_ngrams_example(input_file):
    from nltk.corpus import stopwords
    stops = set(stopwords.words('english'))
    stops.update(['many', 'also', 'would', 'often', 'could'])
    corpus = tp.utils.Corpus(tokenizer=tp.utils.SimpleTokenizer(), 
        stopwords=lambda x: len(x) <= 2 or x in stops)
    # data_feeder yields a tuple of (raw string, user data) or a str (raw string)
    corpus.process(open(input_file, encoding='utf-8'))

    # extract the n-gram candidates first
    cands = corpus.extract_ngrams(min_cf=20, min_df=10, max_len=5, max_cand=1000, normalized=False)
    print('==== extracted n-gram collocations (using PMI) ====')
    for cand in cands:
        print(cand)
    
    # it prints like:
    # tomotopy.label.Candidate(words=["academic","nobel","prize","laureate"], name="", score=23.376673)
    # tomotopy.label.Candidate(words=["canadian","ice","hockey","player"], name="", score=21.658447)
    # tomotopy.label.Candidate(words=["english","race","car","driver"], name="", score=20.356688)
    # tomotopy.label.Candidate(words=["australian","rugby","league","player"], name="", score=20.124966)
    # tomotopy.label.Candidate(words=["american","race","car","driver"], name="", score=19.717760)
    # tomotopy.label.Candidate(words=["new","zealand","rugby","player"], name="", score=18.866398)
    # tomotopy.label.Candidate(words=["american","ice","hockey","player"], name="", score=17.599983)
    # tomotopy.label.Candidate(words=["american","actor","director","producer"], name="", score=16.722300)
    # tomotopy.label.Candidate(words=["nobel","prize","laureate"], name="", score=16.635370)
    # tomotopy.label.Candidate(words=["eastern","orthodox","liturgics"], name="", score=16.540277)
    # ...

    cands = corpus.extract_ngrams(min_cf=20, min_df=10, max_len=5, max_cand=1000, normalized=True)
    print('==== extracted n-gram collocations (using Normalized PMI) ====')
    for cand in cands:
        print(cand)
    
    # it prints like:
    # tomotopy.label.Candidate(words=["buenos","aires"], name="", score=0.996445)
    # tomotopy.label.Candidate(words=["los","angeles"], name="", score=0.988719)
    # tomotopy.label.Candidate(words=["las","vegas"], name="", score=0.982273)
    # tomotopy.label.Candidate(words=["hong","kong"], name="", score=0.978606)
    # tomotopy.label.Candidate(words=["hip","hop"], name="", score=0.965971)
    # tomotopy.label.Candidate(words=["nova","scotia"], name="", score=0.957440)
    # tomotopy.label.Candidate(words=["ice","hockey"], name="", score=0.932300)
    # tomotopy.label.Candidate(words=["nobel","prize","laureate"], name="", score=0.927281)
    # tomotopy.label.Candidate(words=["sri","lankan"], name="", score=0.925504)
    # tomotopy.label.Candidate(words=["ann","arbor"], name="", score=0.921129)
    # ...

    # before concat
    print(corpus[3])

    # concat n-grams in the corpus
    corpus.concat_ngrams(cands, delimiter='_')

    # after concat
    print(corpus[3])

# You can get the sample data file 'enwiki-1000.txt'
# at https://drive.google.com/file/d/18OpNijd4iwPyYZ2O7pQoPyeTAKEXa71J/view?usp=sharing

extract_ngrams_example('enwiki-1000.txt')
