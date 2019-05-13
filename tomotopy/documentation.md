What is tomotopy?
------------------
`tomotopy` is a Python extension of `tomoto` (Topic Modeling Tool) which is a Gibbs-sampling based topic model library written in C++.
It utilizes a vectorization of modern CPUs for maximizing speed. 
The current version of `tomoto` supports several major topic models including 
Latent Dirichlet Allocation(`tomotopy.LDA`), Dirichlet Multinomial Regression(`tomotopy.DMR`),
Hierarchical Dirichlet Process(`tomotopy.HDP`), Multi Grain LDA(`tomotopy.MGLDA`), 
Pachinko Allocation(`tomotopy.PA`) and Hierarchical PA(`tomotopy.HPA`).

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
    mdl = tp.LDA(k=20)
    for line in open('sample.txt'):
        mdl.add_doc(line.strip().split())
    
    for i in range(100):
        mdl.train()
        print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))
    
    for k in range(mdl.k):
        print('Top 10 words of topic #{}'.format(k))
        print(mdl.get_topic_words(k, top_n=10))

Model Save and Load
-------------------
`tomotopy` provides `save` and `load` method for each topic model class, 
so you can save the model into the file whenever you want, and re-load it from the file.
::

    import tomotopy as tp
    
    mdl = tp.HDP()
    for line in open('sample.txt'):
        mdl.add_doc(line.strip().split())
    
    for i in range(100):
        mdl.train()
        print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))
    
    # save into file
    mdl.save('sample_hdp_model.bin')
    
    # load from file
    mdl = tp.HDP.load('sample_hdp_model.bin')
    for k in range(mdl.k):
        if not mdl.is_live_topic(k): continue
        print('Top 10 words of topic #{}'.format(k))
        print(mdl.get_topic_words(k, top_n=10))
    
    # the saved model is HDP model, 
    # so when you load it by LDA model, it will raise an exception
    mdl = tp.LDA.load('sample_hdp_model.bin')

When you load the model from a file, a model type in the file should match the class of methods.

See more at `tomotopy.LDA.save` and `tomotopy.LDA.load` methods.

Documents in the Model and out of the Model
-------------------------------------------

Inference for Unseen Documents
------------------------------

License
---------
`tomotopy` is licensed under the terms of MIT License, 
meaning you can use it for any reasonable purpose and remain in complete ownership of all the documentation you produce.