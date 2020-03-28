tomotopy
========

.. image:: https://badge.fury.io/py/tomotopy.svg
    :target: https://pypi.python.org/pypi/tomotopy

🎌
**English**,
`한국어`_.

.. _한국어: README.kr.rst

What is tomotopy?
------------------

`tomotopy` is a Python extension of `tomoto` (Topic Modeling Tool) which is a Gibbs-sampling based topic model library written in C++.
It utilizes a vectorization of modern CPUs for maximizing speed. 
The current version of `tomoto` supports several major topic models including 

* Latent Dirichlet Allocation (`tomotopy.LDAModel`)
* Labeled LDA (`tomotopy.LLDAModel`)
* Partially Labeled LDA (`tomotopy.PLDAModel`)
* Supervised LDA (`tomotopy.SLDAModel`)
* Dirichlet Multinomial Regression (`tomotopy.DMRModel`)
* Hierarchical Dirichlet Process (`tomotopy.HDPModel`)
* Hierarchical LDA (`tomotopy.HLDAModel`)
* Multi Grain LDA (`tomotopy.MGLDAModel`) 
* Pachinko Allocation (`tomotopy.PAModel`)
* Hierarchical PA (`tomotopy.HPAModel`)
* Correlated Topic Model (`tomotopy.CTModel`).

Please visit https://bab2min.github.io/tomotopy to see more information.

The most recent version of tomotopy is 0.6.2.

Getting Started
---------------
You can install tomotopy easily using pip. (https://pypi.org/project/tomotopy/)
::

    $ pip install tomotopy

The supported OS and Python versions are:

* Linux (x86-64) with Python >= 3.5 
* macOS >= 10.13 with Python >= 3.5
* Windows 7 or later (x86, x86-64) with Python >= 3.5
* Other OS with Python >= 3.5: Compilation from source code required (with c++11 compatible compiler)

After installing, you can start tomotopy by just importing.
::

    import tomotopy as tp
    print(tp.isa) # prints 'avx2', 'avx', 'sse2' or 'none'

Currently, tomotopy can exploits AVX2, AVX or SSE2 SIMD instruction set for maximizing performance.
When the package is imported, it will check available instruction sets and select the best option.
If `tp.isa` tells `none`, iterations of training may take a long time. 
But, since most of modern Intel or AMD CPUs provide SIMD instruction set, the SIMD acceleration could show a big improvement.

Here is a sample code for simple LDA training of texts from 'sample.txt' file.
::

    import tomotopy as tp
    mdl = tp.LDAModel(k=20)
    for line in open('sample.txt'):
        mdl.add_doc(line.strip().split())
    
    for i in range(0, 100, 10):
        mdl.train(10)
        print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))
    
    for k in range(mdl.k):
        print('Top 10 words of topic #{}'.format(k))
        print(mdl.get_topic_words(k, top_n=10))

Performance of tomotopy
-----------------------
`tomotopy` uses Collapsed Gibbs-Sampling(CGS) to infer the distribution of topics and the distribution of words.
Generally CGS converges more slowly than Variational Bayes(VB) that `gensim's LdaModel`_ uses, but its iteration can be computed much faster.
In addition, `tomotopy` can take advantage of multicore CPUs with a SIMD instruction set, which can result in faster iterations.

.. _gensim's LdaModel: https://radimrehurek.com/gensim/models/ldamodel.html 

Following chart shows the comparison of LDA model's running time between `tomotopy` and `gensim`. 
The input data consists of 1000 random documents from English Wikipedia with 1,506,966 words (about 10.1 MB).
`tomotopy` trains 200 iterations and `gensim` trains 10 iterations.

.. image:: https://bab2min.github.io/tomotopy/images/tmt_i5.png

Performance in Intel i5-6600, x86-64 (4 cores)

.. image:: https://bab2min.github.io/tomotopy/images/tmt_xeon.png

Performance in Intel Xeon E5-2620 v4, x86-64 (8 cores, 16 threads)

Although `tomotopy` iterated 20 times more, the overall running time was 5~10 times faster than `gensim`. And it yields a stable result.

It is difficult to compare CGS and VB directly because they are totaly different techniques.
But from a practical point of view, we can compare the speed and the result between them.
The following chart shows the log-likelihood per word of two models' result. 

.. image:: https://bab2min.github.io/tomotopy/images/LLComp.png

The SIMD instruction set has a great effect on performance. Following is a comparison between SIMD instruction sets.

.. image:: https://bab2min.github.io/tomotopy/images/SIMDComp.png

Fortunately, most of recent x86-64 CPUs provide AVX2 instruction set, so we can enjoy the performance of AVX2.

Model Save and Load
-------------------
`tomotopy` provides `save` and `load` method for each topic model class, 
so you can save the model into the file whenever you want, and re-load it from the file.
::

    import tomotopy as tp
    
    mdl = tp.HDPModel()
    for line in open('sample.txt'):
        mdl.add_doc(line.strip().split())
    
    for i in range(0, 100, 10):
        mdl.train(10)
        print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))
    
    # save into file
    mdl.save('sample_hdp_model.bin')
    
    # load from file
    mdl = tp.HDPModel.load('sample_hdp_model.bin')
    for k in range(mdl.k):
        if not mdl.is_live_topic(k): continue
        print('Top 10 words of topic #{}'.format(k))
        print(mdl.get_topic_words(k, top_n=10))
    
    # the saved model is HDP model, 
    # so when you load it by LDA model, it will raise an exception
    mdl = tp.LDAModel.load('sample_hdp_model.bin')

When you load the model from a file, a model type in the file should match the class of methods.

See more at `tomotopy.LDAModel.save` and `tomotopy.LDAModel.load` methods.

Documents in the Model and out of the Model
-------------------------------------------
We can use Topic Model for two major purposes. 
The basic one is to discover topics from a set of documents as a result of trained model,
and the more advanced one is to infer topic distributions for unseen documents by using trained model.

We named the document in the former purpose (used for model training) as **document in the model**,
and the document in the later purpose (unseen document during training) as **document out of the model**.

In `tomotopy`, these two different kinds of document are generated differently.
A **document in the model** can be created by `tomotopy.LDAModel.add_doc` method.
`add_doc` can be called before `tomotopy.LDAModel.train` starts. 
In other words, after `train` called, `add_doc` cannot add a document into the model because the set of document used for training has become fixed.

To acquire the instance of the created document, you should use `tomotopy.LDAModel.docs` like:

::

    mdl = tp.LDAModel(k=20)
    idx = mdl.add_doc(words)
    if idx < 0: raise RuntimeError("Failed to add doc")
    doc_inst = mdl.docs[idx]
    # doc_inst is an instance of the added document

A **document out of the model** is generated by `tomotopy.LDAModel.make_doc` method. `make_doc` can be called only after `train` starts.
If you use `make_doc` before the set of document used for training has become fixed, you may get wrong results.
Since `make_doc` returns the instance directly, you can use its return value for other manipulations.

::

    mdl = tp.LDAModel(k=20)
    # add_doc ...
    mdl.train(100)
    doc_inst = mdl.make_doc(unseen_words) # doc_inst is an instance of the unseen document

Inference for Unseen Documents
------------------------------
If a new document is created by `tomotopy.LDAModel.make_doc`, its topic distribution can be inferred by the model.
Inference for unseen document should be performed using `tomotopy.LDAModel.infer` method.

::

    mdl = tp.LDAModel(k=20)
    # add_doc ...
    mdl.train(100)
    doc_inst = mdl.make_doc(unseen_words)
    topic_dist, ll = mdl.infer(doc_inst)
    print("Topic Distribution for Unseen Docs: ", topic_dist)
    print("Log-likelihood of inference: ", ll)

The `infer` method can infer only one instance of `tomotopy.Document` or a `list` of instances of `tomotopy.Document`. 
See more at `tomotopy.LDAModel.infer`.

Parallel Sampling Algorithms
----------------------------
Since version 0.5.0, `tomotopy` allows you to choose a parallelism algorithm. 
The algorithm provided in versions prior to 0.4.2 is `COPY_MERGE`, which is provided for all topic models.
The new algorithm `PARTITION`, available since 0.5.0, makes training generally faster and more memory-efficient, but it is available at not all topic models.

The following chart shows the speed difference between the two algorithms based on the number of topics and the number of workers.

.. image:: https://bab2min.github.io/tomotopy/images/algo_comp.png

.. image:: https://bab2min.github.io/tomotopy/images/algo_comp2.png


Examples
--------
You can find an example python code of tomotopy at https://github.com/bab2min/tomotopy/blob/master/example.py .

You can also get the data file used in the example code at https://drive.google.com/file/d/18OpNijd4iwPyYZ2O7pQoPyeTAKEXa71J/view .

License
---------
`tomotopy` is licensed under the terms of MIT License, 
meaning you can use it for any reasonable purpose and remain in complete ownership of all the documentation you produce.

History
-------
* 0.6.2 (2020-03-28)
    * A critical bug related to `save` and `load` was fixed. Version 0.6.0 and 0.6.1 have been removed from releases.

* 0.6.1 (2020-03-22) (removed)
    * A bug related to module loading was fixed.

* 0.6.0 (2020-03-22) (removed)
    * `tomotopy.utils.Corpus` class that manages multiple documents easily was added.
    * `tomotopy.LDAModel.set_word_prior` method that controls word-topic priors of topic models was added.
    * A new argument `min_df` that filters words based on document frequency was added into every topic model's __init__.
    * `tomotopy.label`, the submodule about topic labeling was added. Currently, only `tomotopy.label.FoRelevance` is provided.

* 0.5.2 (2020-03-01)
    * A segmentation fault problem was fixed in `tomotopy.LLDAModel.add_doc`.
    * A bug was fixed that `infer` of `tomotopy.HDPModel` sometimes crashes the program.
    * A crash issue was fixed of `tomotopy.LDAModel.infer` with ps=tomotopy.ParallelScheme.PARTITION, together=True.

* 0.5.1 (2020-01-11)
    * A bug was fixed that `tomotopy.SLDAModel.make_doc` doesn't support missing values for `y`.
    * Now `tomotopy.SLDAModel` fully supports missing values for response variables `y`. Documents with missing values (NaN) are included in modeling topic, but excluded from regression of response variables.

* 0.5.0 (2019-12-30)
    * Now `tomotopy.PAModel.infer` returns both topic distribution nd sub-topic distribution.
    * New methods get_sub_topics and get_sub_topic_dist were added into `tomotopy.Document`. (for PAModel)
    * New parameter `parallel` was added for `tomotopy.LDAModel.train` and `tomotopy.LDAModel.infer` method. You can select parallelism algorithm by changing this parameter.
    * `tomotopy.ParallelScheme.PARTITION`, a new algorithm, was added. It works efficiently when the number of workers is large, the number of topics or the size of vocabulary is big.
    * A bug where `rm_top` didn't work at `min_cf` < 2 was fixed.

* 0.4.2 (2019-11-30)
    * Wrong topic assignments of `tomotopy.LLDAModel` and `tomotopy.PLDAModel` were fixed.
    * Readable __repr__ of `tomotopy.Document` and `tomotopy.Dictionary` was implemented.

* 0.4.1 (2019-11-27)
    * A bug at init function of `tomotopy.PLDAModel` was fixed.

* 0.4.0 (2019-11-18)
    * New models including `tomotopy.PLDAModel` and `tomotopy.HLDAModel` were added into the package.

* 0.3.1 (2019-11-05)
    * An issue where `get_topic_dist()` returns incorrect value when `min_cf` or `rm_top` is set was fixed.
    * The return value of `get_topic_dist()` of `tomotopy.MGLDAModel` document was fixed to include local topics.
    * The estimation speed with `tw=ONE` was improved.

* 0.3.0 (2019-10-06)
    * A new model, `tomotopy.LLDAModel` was added into the package.
    * A crashing issue of `HDPModel` was fixed.
    * Since hyperparameter estimation for `HDPModel` was implemented, the result of `HDPModel` may differ from previous versions.
        If you want to turn off hyperparameter estimation of HDPModel, set `optim_interval` to zero.

* 0.2.0 (2019-08-18)
    * New models including `tomotopy.CTModel` and `tomotopy.SLDAModel` were added into the package.
    * A new parameter option `rm_top` was added for all topic models.
    * The problems in `save` and `load` method for `PAModel` and `HPAModel` were fixed.
    * An occassional crash in loading `HDPModel` was fixed.
    * The problem that `ll_per_word` was calculated incorrectly when `min_cf` > 0 was fixed.

* 0.1.6 (2019-08-09)
    * Compiling errors at clang with macOS environment were fixed.

* 0.1.4 (2019-08-05)
    * The issue when `add_doc` receives an empty list as input was fixed.
    * The issue that `tomotopy.PAModel.get_topic_words` doesn't extract the word distribution of subtopic was fixed.

* 0.1.3 (2019-05-19)
    * The parameter `min_cf` and its stopword-removing function were added for all topic models.

* 0.1.0 (2019-05-12)
    * First version of **tomotopy**