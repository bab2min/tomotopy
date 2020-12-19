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
    cands = corpus.extract_ngrams(min_cf=20, min_df=10, max_len=5, max_cand=1000)
    print('==== extracted n-gram collocations ====')
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

    # before concat
    print(corpus[3])

    # concat n-grams in the corpus
    corpus.concat_ngrams(cands, delimiter='_')

    # after concat
    print(corpus[3])

# You can get the sample data file 'enwiki-1000.txt'
# at https://drive.google.com/file/d/18OpNijd4iwPyYZ2O7pQoPyeTAKEXa71J/view?usp=sharing

extract_ngrams_example('enwiki-1000.txt')
