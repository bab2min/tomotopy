import os, os.path
from glob import glob
import platform
import sys
import shutil
import subprocess
import re

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion
from sysconfig import get_platform
import struct

import numpy as np


def get_extra_cmake_options():
    """read --clean, --no, --set, --compiler-flags, and -G options from the command line and add them as cmake switches.
    """
    _cmake_extra_options = ["-DCMAKE_POSITION_INDEPENDENT_CODE=1"]
    if os.environ.get('MACOSX_DEPLOYMENT_TARGET'):
        _cmake_extra_options.append("-DCMAKE_OSX_DEPLOYMENT_TARGET=" + os.environ['MACOSX_DEPLOYMENT_TARGET'])
    _clean_build_folder = False

    opt_key = None

    argv = [arg for arg in sys.argv]  # take a copy
    # parse command line options and consume those we care about
    for arg in argv:
        if opt_key == 'compiler-flags':
            _cmake_extra_options.append('-DCMAKE_CXX_FLAGS={arg}'.format(arg=arg.strip()))
        elif opt_key == 'G':
            _cmake_extra_options += ['-G', arg.strip()]
        elif opt_key == 'no':
            _cmake_extra_options.append('-D{arg}=no'.format(arg=arg.strip()))
        elif opt_key == 'set':
            _cmake_extra_options.append('-D{arg}'.format(arg=arg.strip()))

        if opt_key:
            sys.argv.remove(arg)
            opt_key = None
            continue

        if arg == '--clean':
            _clean_build_folder = True
            sys.argv.remove(arg)
            continue

        if arg == '--yes':
            print("The --yes options to kiwipiepy's setup.py don't do anything since all these options ")
            print("are on by default.  So --yes has been removed.  Do not give it to setup.py.")
            sys.exit(1)
        if arg in ['--no', '--set', '--compiler-flags']:
            opt_key = arg[2:].lower()
            sys.argv.remove(arg)
            continue
        if arg in ['-G']:
            opt_key = arg[1:]
            sys.argv.remove(arg)
            continue

    return _cmake_extra_options, _clean_build_folder

def num_available_cpu_cores(ram_per_build_process_in_gb):
    import multiprocessing
    try:
        mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')  
        mem_gib = mem_bytes/(1024.**3)
        num_cores = multiprocessing.cpu_count() 
        mem_cores = int(mem_gib/float(ram_per_build_process_in_gb)+0.5)
        return max(min(num_cores, mem_cores), 1)
    except ValueError:
        return 2 # just assume 2 if we can't get the os to tell us the right answer.


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir='', cmake_options=[], *args, **kwargs):
        Extension.__init__(self, name, sources=[], *args, **kwargs)
        self.sourcedir = os.path.abspath(sourcedir)
        self.additional_options = cmake_options

class CMakeBuild(build_ext):

    def get_cmake_version(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except:
            sys.stderr.write("\nERROR: CMake must be installed to build tomotopy\n\n") 
            sys.exit(1)
        return re.search(r'version\s*([\d.]+)', out.decode()).group(1)

    def run(self):
        cmake_version = self.get_cmake_version()
        if platform.system() == "Windows":
            if LooseVersion(cmake_version) < '3.1.0':
                sys.stderr.write("\nERROR: CMake >= 3.1.0 is required on Windows\n\n")
                sys.exit(1)

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        libs = self.get_libraries(ext)

        cmake_args = [
            '-DINCLUDE_DIRS={}'.format(';'.join(self.include_dirs + ext.include_dirs)),
            '-DLIBRARY_DIRS={}'.format(';'.join(self.library_dirs)),
            '-DLIBRARIES={}'.format(';'.join(libs)),
            '-DPYTHON_EXECUTABLE=' + sys.executable,
        ]
        cmake_args += ext.additional_options
        print(cmake_args)

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            else:
                cmake_args += ['-A', 'Win32']
            build_args += ['--', '/m'] 
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j'+str(num_available_cpu_cores(2))]

        build_folder = os.path.abspath(self.build_temp)

        if not os.path.exists(build_folder):
            os.makedirs(build_folder)

        cmake_setup = ['cmake', ext.sourcedir] + cmake_args
        cmake_build = ['cmake', '--build', '.'] + build_args

        print("Building extension for Python {}".format(sys.version.split('\n',1)[0]))
        print("Invoking CMake setup: '{}'".format(' '.join(cmake_setup)))
        sys.stdout.flush()
        subprocess.check_call(cmake_setup, cwd=build_folder)
        print("Invoking CMake build: '{}'".format(' '.join(cmake_build)))
        sys.stdout.flush()
        subprocess.check_call(cmake_build, cwd=build_folder)
        
        fullname = self.get_ext_fullname(ext.name)
        filename = self.get_ext_filename(fullname)

        if platform.system() == "Windows":
            shutil.move(os.sep.join([build_folder, cfg, f'_tomotopy_target.dll']), os.sep.join([extdir, filename]))
        else:
            shutil.move(glob(os.sep.join([build_folder, 'lib_tomotopy_target.*']))[0], os.sep.join([extdir, filename]))


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


exec(open('tomotopy/_version.py').read())

here = os.path.abspath(os.path.dirname(__file__))

long_description = ''
for line in open(os.path.join(here, 'tomotopy/documentation.rst'), encoding='utf-8'):
    long_description += re.sub(r'^<.+>\s*$', '', line)

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
    
cmake_extra_options, _ = get_extra_cmake_options()

build_for_manyplatform = ('many' in os.environ.get('AUDITWHEEL_PLAT', ''))

arch_levels = ['none', 'sse2', 'avx2']
if platform.system() == 'Windows': 
    arch_levels = ['none', 'sse2', 'avx2']
elif platform.system() == 'Darwin': 
    if tomotopy_cpu_arch == 'arm64':
        arch_levels = ['arm64']
    elif build_for_manyplatform: 
        arch_levels = ['sse2', 'avx', 'avx2']
    else:
        arch_levels = ['sse2']
elif build_for_manyplatform:
    if tomotopy_cpu_arch in ('arm64', 'aarch64'):
        arch_levels = ['arm64']
else:
    arch_levels = ['native']

# if target is in 64bit, remove 'none'
if struct.calcsize("P") == 8:
    try:
        arch_levels.remove('none')
    except ValueError:
        pass
else:
    arch_levels = [a for a in arch_levels if a in ('none', 'sse2')]

modules = []
if len(arch_levels) > 1:
    modules.append(CMakeExtension('_tomotopy',
        libraries=[],
        include_dirs=['include', np.get_include()],
        cmake_options=cmake_extra_options + ['-DEXT_TYPE=dispatcher'],
    ))

if build_for_manyplatform:
    cmake_extra_options += ['-DNO_DEBUG_INFO=1']
else:
    cmake_extra_options += ['-DNO_DEBUG_INFO=0']

if os.environ.get('TOMOTOPY_LANG'):
    cmake_extra_options += ['-DTOMOTOPY_LANG=' + os.environ['TOMOTOPY_LANG']]

for arch in arch_levels:
    if len(arch_levels) > 1:
        if arch in ('none', 'native', 'arm64'):
            module_name = '_tomotopy_none'
        else:
            module_name = '_tomotopy_' + arch
    else:
        module_name = '_tomotopy'
    modules.append(CMakeExtension(module_name,
        libraries=[],
        include_dirs=['include', np.get_include()],
        cmake_options=cmake_extra_options + [
            '-DEXT_TYPE=handler', 
            '-DTARGET_ARCH=' + arch,
            '-DTOMOTOPY_ISA=' + arch, 
            '-DMODULE_NAME=PyInit_' + module_name],
    ))

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
    install_requires=['numpy>=1.11.0,<2'],
    keywords='NLP,Topic Model',

    packages=['tomotopy'],
    include_package_data=True,
    ext_modules=modules,
    cmdclass=dict(build_ext=CMakeBuild),
)
