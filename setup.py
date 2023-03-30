from setuptools import setup, Extension
from setuptools.command.install import install
from codecs import open
import os, os.path, struct, re, platform

import numpy

try:
    from setuptools._distutils import _msvccompiler
    _msvccompiler.PLAT_TO_VCVARS['win-amd64'] = 'amd64'
except:
    pass

if os.environ.get('TOMOTOPY_CPU_ARCH'):
    tomotopy_cpu_arch = os.environ['TOMOTOPY_CPU_ARCH']
else:
    try:
        import platform
        tomotopy_cpu_arch = platform.machine()
    except:
        tomotopy_cpu_arch = 'x86_64'

from sysconfig import get_platform
fd = get_platform().split('-')
if fd[0] == 'macosx':
    if os.environ.get('MACOSX_DEPLOYMENT_TARGET'): 
        from distutils import sysconfig
        cfg_target = sysconfig.get_config_var('MACOSX_DEPLOYMENT_TARGET') or ''
        if cfg_target.startswith('10.'):
            cfg_target = list(map(int, cfg_target.split('.')))
            cur_target = list(map(int, os.environ['MACOSX_DEPLOYMENT_TARGET'].split('.')))
            target = max(cfg_target, cur_target)
            if target != cur_target:
                print(f"MACOSX_DEPLOYMENT_TARGET={'.'.join(map(str, cur_target))} is not supported. MACOSX_DEPLOYMENT_TARGET={'.'.join(map(str, target))} is used instead.")
            os.environ['MACOSX_DEPLOYMENT_TARGET'] = '.'.join(map(str, target))
        fd[1] = os.environ['MACOSX_DEPLOYMENT_TARGET']
    os.environ['_PYTHON_HOST_PLATFORM'] = '-'.join(fd[:-1] + [tomotopy_cpu_arch])
    
exec(open('tomotopy/_version.py').read())

here = os.path.abspath(os.path.dirname(__file__))

long_description = ''
for line in open(os.path.join(here, 'tomotopy/documentation.rst'), encoding='utf-8'):
    long_description += re.sub(r'^<.+>\s*$', '', line)

sources = []
for f in os.listdir(os.path.join(here, 'src/python')):
    if f.endswith('.cpp') and not f.endswith('py_rt.cpp'): sources.append('src/python/' + f)
for f in os.listdir(os.path.join(here, 'src/TopicModel')):
    if f.endswith('.cpp'): sources.append('src/TopicModel/' + f)
for f in os.listdir(os.path.join(here, 'src/Labeling')):
    if f.endswith('.cpp'): sources.append('src/Labeling/' + f)

largs = []
arch_levels = {'':'', 'sse2':'-msse2', 'avx':'-mavx', 'avx2':'-mavx2 -mfma'}
if platform.system() == 'Windows': 
    cargs = ['/O2', '/MT', '/Gy', '/D__SSE2__']
    arch_levels = {'':'', 'sse2':'/arch:SSE2', 'avx':'/arch:AVX', 'avx2':'/arch:AVX2'}
elif platform.system() == 'Darwin': 
    cargs = ['-std=c++1y', '-O3', '-fpermissive', '-stdlib=libc++', '-Wno-unused-variable', '-Wno-switch']
    largs += ['-stdlib=libc++']
    if 'many' in os.environ.get('AUDITWHEEL_PLAT', ''): cargs.append('-g0')
    if tomotopy_cpu_arch == 'arm64':
        arch_levels = {'': ''}
        cargs += ['-DTOMOTOPY_ISA=arm64', '-arch', 'arm64']
        largs += ['-arch', 'arm64']
    else:
        if 'many' not in os.environ.get('AUDITWHEEL_PLAT', ''): arch_levels = {'':'-msse2'}
        cargs += ['-arch', 'x86_64']
        largs += ['-arch', 'x86_64']
elif 'many' in os.environ.get('AUDITWHEEL_PLAT', ''):
    cargs = ['-std=c++1y', '-O3', '-fpermissive', '-g0', '-Wno-unused-variable', '-Wno-switch']
    if tomotopy_cpu_arch in ('arm64', 'aarch64'):
        arch_levels = {'': ''}
        cargs += ['-DTOMOTOPY_ISA=arm64']
else:
    cargs = ['-std=c++1y', '-O3', '-fpermissive', '-Wno-unused-variable', '-Wno-switch']
    arch_levels = {'': '-march=native'}

if struct.calcsize('P') < 8: arch_levels = {k:v for k, v in arch_levels.items() if k in ('', 'sse2')}
else: arch_levels = {k:v for k, v in arch_levels.items() if k not in ('sse2',)}

lang_macro = []
if os.environ.get('TOMOTOPY_LANG') == 'kr': lang_macro = [('DOC_KO', '1')]

modules = []
if len(arch_levels) > 1:
    modules.append(Extension('_tomotopy',
        libraries=[],
        include_dirs=['include', numpy.get_include()],
        sources=['src/python/py_rt.cpp'],
        extra_compile_args=cargs, extra_link_args=largs))

for arch, aopt in arch_levels.items():
    if len(arch_levels) > 1:
        module_name = '_tomotopy_' + (arch or 'none')
    else:
        module_name = '_tomotopy'
    modules.append(Extension(module_name,
                    libraries=[],
                    include_dirs=['include', numpy.get_include()],
                    sources=sources,
                    define_macros=[('MODULE_NAME', 'PyInit_' + module_name)] + lang_macro,
                    extra_compile_args=cargs + (aopt.split(' ') if aopt else []), extra_link_args=largs))

setup(
    name='tomotopy',

    version=__version__,

    description='Tomoto, Topic Modeling Tool for Python',
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
        "Operating System :: POSIX",
        "Operating System :: MacOS"
    ],
    install_requires=['numpy>=1.11.0'],
    keywords='NLP,Topic Model',

    packages=['tomotopy'],
    include_package_data=True,
    ext_modules=modules
)
