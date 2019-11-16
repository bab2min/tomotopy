from setuptools import setup, Extension
from codecs import open
import os, os.path, struct, re, platform
from setuptools.command.install import install

here = os.path.abspath(os.path.dirname(__file__))

long_description = ''
for line in open(os.path.join(here, 'tomotopy/documentation.rst'), encoding='utf-8'):
    long_description += re.sub(r'^<.+>\s*$', '', line)

sources = []
for f in os.listdir(os.path.join(here, 'src/python')):
    if f.endswith('.cpp'): sources.append('src/python/' + f)
for f in os.listdir(os.path.join(here, 'src/TopicModel')):
    if f.endswith('.cpp'): sources.append('src/TopicModel/' + f)

largs = []
if platform.system() == 'Windows': 
    cargs = ['/O2', '/MT', '/Gy']
    arch_levels = {'':'', 'sse2':'/arch:SSE2', 'avx':'/arch:AVX', 'avx2':'/arch:AVX2'}
elif platform.system() == 'Darwin': 
    cargs = ['-std=c++0x', '-O3', '-fpermissive', '-stdlib=libc++']
    largs += ['-stdlib=libc++']
    arch_levels = {'':'-march=native'}
elif 'manylinux' in os.environ.get('AUDITWHEEL_PLAT', ''):
    cargs = ['-std=c++0x', '-O3', '-fpermissive', '-g0']
    arch_levels = {'':'', 'sse2':'-msse2', 'avx':'-mavx', 'avx2':'-mavx2'}
else:
    cargs = ['-std=c++0x', '-O3', '-fpermissive']
    arch_levels = {'':'-march=native'}

if struct.calcsize('P') < 8: arch_levels = {k:v for k, v in arch_levels.items() if k in ('', 'sse2')}
else: arch_levels = {k:v for k, v in arch_levels.items() if k not in ('sse2',)}

modules = []
for arch, aopt in arch_levels.items():
    module_name = '_tomotopy' + ('_' + arch if arch else '')
    modules.append(Extension(module_name,
                    libraries=[],
                    include_dirs=['include'],
                    sources=sources,
                    define_macros=[('MODULE_NAME', 'PyInit_' + module_name)],
                    extra_compile_args=cargs + ([aopt] if aopt else []), extra_link_args=largs))


setup(
    name='tomotopy',

    version='0.3.1',

    description='Tomoto, The Topic Modeling Tool for Python',
    long_description=long_description,

    url='https://github.com/bab2min/tomotopy',

    author='bab2min',
    author_email='bab2min@gmail.com',

    license='MIT License',

    classifiers=[
        'Development Status :: 3 - Alpha',

        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries",
        "Topic :: Text Processing :: Linguistic",
		"Topic :: Scientific/Engineering :: Information Analysis",

        "License :: OSI Approved :: MIT License",

        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
		"Operating System :: Microsoft :: Windows :: Windows Vista",
		"Operating System :: Microsoft :: Windows :: Windows 7",
		"Operating System :: Microsoft :: Windows :: Windows 8",
		"Operating System :: Microsoft :: Windows :: Windows 8.1",
		"Operating System :: Microsoft :: Windows :: Windows 10",
		"Operating System :: POSIX"
    ],
    install_requires=['py-cpuinfo'],
    keywords='NLP,Topic Model',

    packages = ['tomotopy'],
    include_package_data=True,
    ext_modules = modules
)
