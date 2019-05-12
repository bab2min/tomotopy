"""
Python package `tomotopy` provides types and functions for various Topic Model 
including LDA, DMR, HDP, MG-LDA, PA and HPA. It is written in C++ for speed and provides Python extension.

.. include:: ./documentation.md
"""
from enum import IntEnum

class TermWeight(IntEnum):
    """
    This enumeration is for Term Weighting Scheme and it is based on following paper:
    
    > * Wilson, A. T., & Chew, P. A. (2010, June). Term weighting schemes for latent dirichlet allocation. In human language technologies: The 2010 annual conference of the North American Chapter of the Association for Computational Linguistics (pp. 465-473). Association for Computational Linguistics.
    
    There are three options for term weighting and the basic one is ONE. The others also can be applied for every topic model in `tomotopy`. 
    """

    ONE = 0
    """ Consider every term equal (default)"""

    IDF = 1
    """ 
    Use Inverse Document Frequency term weighting.
    
    Thus, a term occurring at almost every document has very low weighting
    and a term occurring at a few document has high weighting. 
    """

    PMI = 2
    """
    Use Pointwise Mutual Information term weighting.
    """

isa = ''
"""
Indicate which SIMD instruction set is used for acceleration.
It can be one of `'avx2'`, `'avx'`, `'sse2'` and `'none'`.
"""
def _load():
    import importlib
    from cpuinfo import get_cpu_info
    flags = get_cpu_info()['flags']
    for isa in ['avx2', 'avx', 'sse2', '']:
        if isa in flags or not isa:
            try:
                mod_name = '_tomotopy' + ('_' + isa if isa else '')
                globals().update({k:v for k, v in vars(importlib.import_module(mod_name)).items() if not k.startswith('_')})
                return
            except:
                pass
_load()
if not isa: isa = 'none'
del _load, IntEnum


__pdoc__ = {}
__pdoc__['LDA.__init__'] = '''
This type provides Latent Dirichlet Allocation(LDA) topic model and its implementation is based on following papers:

> * Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent dirichlet allocation. Journal of machine Learning research, 3(Jan), 993-1022.
> * Newman, D., Asuncion, A., Smyth, P., & Welling, M. (2009). Distributed algorithms for topic models. Journal of Machine Learning Research, 10(Aug), 1801-1828.

`LDA(tw=TermWeight.ONE, k=1, alpha=0.1, eta=0.01, seed=?)`

* `tw` : term weighting scheme in `tomotopy.TermWeight`. default is TermWeight.ONE
* `k` : the number of topics between 1 ~ 65535. 
* `alpha` : hyperparameter of Dirichlet distribution for document-topic 
* `eta` : hyperparameter of Dirichlet distribution for topic-word
* `seed` : random seed. default value is a random number from `random_device{}` in C++

'''

__pdoc__['DMR.__init__'] = '''
This type provides Dirichlet Multinomial Regression(DMR) topic model and its implementation is based on following papers:

> * Mimno, D., & McCallum, A. (2012). Topic models conditioned on arbitrary features with dirichlet-multinomial regression. arXiv preprint arXiv:1206.3278.

`DMR(tw=TermWeight.ONE, k=1, alpha=0.1, eta=0.01, sigma=1.0, alpha_epsilon=1e-10, seed=?)`

* `tw` : term weighting scheme in `tomotopy.TermWeight`. default is TermWeight.ONE
* `k` : the number of topics between 1 ~ 65535. 
* `alpha` : exponential of mean of normal distribution for `lambdas`
* `eta` : hyperparameter of Dirichlet distribution for topic-word
* `sigma` : standard deviation of normal distribution for `lambdas`
* `alpha_epsilon` : small value for preventing `exp(lambdas)` to be zero
* `seed` : random seed. default value is a random number from `random_device{}` in C++

'''

__pdoc__['HDP.__init__'] = '''
This type provides Hierarchical Dirichlet Process(HDP) topic model and its implementation is based on following papers:

> * Teh, Y. W., Jordan, M. I., Beal, M. J., & Blei, D. M. (2005). Sharing clusters among related groups: Hierarchical Dirichlet processes. In Advances in neural information processing systems (pp. 1385-1392).
> * Newman, D., Asuncion, A., Smyth, P., & Welling, M. (2009). Distributed algorithms for topic models. Journal of Machine Learning Research, 10(Aug), 1801-1828.

`HDP(tw=TermWeight.ONE, initial_k=1, alpha=0.1, eta=0.01, gamma=0.1, seed=?)`

* `tw` : term weighting scheme in `tomotopy.TermWeight`. default is TermWeight.ONE
* `initial_k` : the initial number of topics between 1 ~ 65535. The number of topics will be adjusted for data during training.
* `alpha` : concentration coeficient of Dirichlet Process for document-table 
* `eta` : hyperparameter of Dirichlet distribution for topic-word
* `gamma` : concentration coeficient of Dirichlet Process for table-topic
* `seed` : random seed. default value is a random number from `random_device{}` in C++

'''

__pdoc__['MGLDA.__init__'] = '''
This type provides Multi Grain Latent Dirichlet Allocation(MG-LDA) topic model and its implementation is based on following papers:

> * Titov, I., & McDonald, R. (2008, April). Modeling online reviews with multi-grain topic models. In Proceedings of the 17th international conference on World Wide Web (pp. 111-120). ACM.

`MGLDA(tw=TermWeight.ONE, ..., seed=?)`

* `tw` : term weighting scheme in `tomotopy.TermWeight`. default is TermWeight.ONE
* `seed` : random seed. default value is a random number from `random_device{}` in C++

'''

__pdoc__['PA.__init__'] = '''
This type provides Pachinko Allocation(PA) topic model and its implementation is based on following papers:

> * Li, W., & McCallum, A. (2006, June). Pachinko allocation: DAG-structured mixture models of topic correlations. In Proceedings of the 23rd international conference on Machine learning (pp. 577-584). ACM.

`PA(tw=TermWeight.ONE, k1=1, k2=1, alpha=0.1, eta=0.01, seed=?)`

* `tw` : term weighting scheme in `tomotopy.TermWeight`. default is TermWeight.ONE
* `k1` : the number of super topics between 1 ~ 65535.
* `k2` : the number of sub topics between 1 ~ 65535.
* `alpha` : initial hyperparameter of Dirichlet distribution for document-topic 
* `eta` : hyperparameter of Dirichlet distribution for topic-word
* `seed` : random seed. default value is a random number from `random_device{}` in C++

'''

__pdoc__['HPA.__init__'] = '''
This type provides Hierarchical Pachinko Allocation(HPA) topic model and its implementation is based on following papers:

> * Mimno, D., Li, W., & McCallum, A. (2007, June). Mixtures of hierarchical topics with pachinko allocation. In Proceedings of the 24th international conference on Machine learning (pp. 633-640). ACM.

`HPA(tw=TermWeight.ONE, k1=1, k2=1, alpha=0.1, eta=0.01, seed=?)`

* `tw` : term weighting scheme in `tomotopy.TermWeight`. default is TermWeight.ONE
* `k1` : the number of super topics between 1 ~ 65535.
* `k2` : the number of sub topics between 1 ~ 65535.
* `alpha` : initial hyperparameter of Dirichlet distribution for document-topic 
* `eta` : hyperparameter of Dirichlet distribution for topic-word
* `seed` : random seed. default value is a random number from `random_device{}` in C++

'''

