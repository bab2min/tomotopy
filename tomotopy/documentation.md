What is `tomotopy`?
---------------------

Getting Started
---------------
You can install tomotopy easily using pip.

    $ pip install tomotopy

For Linux, it is neccesary to have gcc 5 or more for compiling C++14 codes.
After installing, you can start tomotopy by just importing.

    import tomotopy as tp
    print(tp.isa) # prints 'avx2', 'avx', 'sse2' or 'none'

Currently, tomotopy can exploits AVX2, AVX or SSE2 SIMD instruction set
for maximizing performance. When the package is imported, it will check available instruction sets and select the best option.
If `tp.isa` tells `none`, iterations of training may take a long time. 
But, since most of modern Intel or AMD CPUs provide SIMD instruction set, the SIMD acceleration could show a big improvement.

License
---------
`tomotopy` is licensed under the terms of MIT License, 
meaning you can use it for any reasonable purpose and remain in complete ownership of all the documentation you produce.