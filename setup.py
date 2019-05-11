from setuptools import setup, Extension
from codecs import open
import os, os.path
from setuptools.command.install import install

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README'), encoding='utf-8') as f:
    long_description = f.read()

sources = []
for f in os.listdir(os.path.join(here, 'src')):
    if f.endswith('.cpp'): sources.append('src/' + f)

if os.name == 'nt': 
    cargs = ['/O2', '/MT', '/Gy']
    arch_levels = {'':'', 'sse2':'/arch:SSE2', 'avx':'/arch:AVX', 'avx2':'/arch:AVX2'}
else: 
    cargs = ['-std=c++1y', '-O3', '-fpermissive']
    arch_levels = {'':'-march=native'}

modules = []
for arch, aopt in arch_levels.items():
    module_name = '_tomotopy' + ('_' + arch if arch else '')
    modules.append(Extension(module_name,
                    libraries = [],
                    include_dirs=['include'],
                    sources = sources,
                    define_macros=[('MODULE_NAME', 'PyInit_' + module_name)],
                    extra_compile_args=cargs + [aopt]))


setup(
    name='tomotopy',

    version='0.1.0',

    description='Tomoto, The Topic Modeling Tool for Python',
    long_description=long_description,

    url='https://github.com/bab2min/tomotopy',

    author='bab2min',
    author_email='bab2min@gmail.com',

    license='MIT License',

    classifiers=[
        'Development Status :: 3 - Alpha',

        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries",
        "Topic :: Text Processing :: Linguistic",
		"Scientific/Engineering :: Information Analysis",
		"Text Processing :: Linguistic",

        "License :: OSI Approved :: MIT License",

        'Programming Language :: Python :: 3',
        'Programming Language :: C++'
    ],
    install_requires=['py-cpuinfo'],
    keywords='NLP, Topic Model',

    packages = ['tomotopy'],
    include_package_data=True,
    ext_modules = modules
)
