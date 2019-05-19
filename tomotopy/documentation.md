What is tomotopy?
------------------
`tomotopy` is a Python extension of `tomoto` (Topic Modeling Tool) which is a Gibbs-sampling based topic model library written in C++.
It utilizes a vectorization of modern CPUs for maximizing speed. 
The current version of `tomoto` supports several major topic models including 
Latent Dirichlet Allocation(`tomotopy.LDAModel`), Dirichlet Multinomial Regression(`tomotopy.DMRModel`),
Hierarchical Dirichlet Process(`tomotopy.HDPModel`), Multi Grain LDA(`tomotopy.MGLDAModel`), 
Pachinko Allocation(`tomotopy.PAModel`) and Hierarchical PA(`tomotopy.HPAModel`).

Getting Started
---------------
You can install tomotopy easily using pip.
::

    $ pip install tomotopy

For Linux, it is neccesary to have gcc 5 or more for compiling C++14 codes.
After installing, you can start tomotopy by just importing.
::

    import tomotopy as tp
    print(tp.isa) # prints 'avx2', 'avx', 'sse2' or 'none'

Currently, tomotopy can exploits AVX2, AVX or SSE2 SIMD instruction set
for maximizing performance. When the package is imported, it will check available instruction sets and select the best option.
If `tp.isa` tells `none`, iterations of training may take a long time. 
But, since most of modern Intel or AMD CPUs provide SIMD instruction set, the SIMD acceleration could show a big improvement.

Here is a sample code for simple LDA training of texts from 'sample.txt' file.
::

    import tomotopy as tp
    mdl = tp.LDAModel(k=20)
    for line in open('sample.txt'):
        mdl.add_doc(line.strip().split())
    
    for i in range(100):
        mdl.train()
        print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))
    
    for k in range(mdl.k):
        print('Top 10 words of topic #{}'.format(k))
        print(mdl.get_topic_words(k, top_n=10))

Performance of tomotopy
-----------------------
`tomotopy` uses Collapsed Gibbs-Sampling(CGS) to infer the distribution of topics and the distribution of words.
Generally CGS converges more slowly than Variational Bayes(VB) that [gensim's LdaModel] uses, but its iteration can be computed much faster.
In addition, `tomotopy` can take advantage of multicore CPUs with a SIMD instruction set, which can result in faster iterations.

[gensim's LdaModel]: https://radimrehurek.com/gensim/models/ldamodel.html 

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

The  SIMD instruction set has a great effect on performance. Following is a comparison between SIMD instruction sets.

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
    
    for i in range(100):
        mdl.train()
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

Inference for Unseen Documents
------------------------------

License
---------
`tomotopy` is licensed under the terms of MIT License, 
meaning you can use it for any reasonable purpose and remain in complete ownership of all the documentation you produce.