
def get_old_numpy_version():
    import sys
    try:
        import platform
        cpu_arch = platform.machine()
    except:
        cpu_arch = 'x86_64'

    py_version = sys.version_info
    if py_version >= (3, 11): return '1.24.0'
    if py_version >= (3, 10): return '1.22.0'
    if py_version >= (3, 9): return '1.20.0'
    if py_version >= (3, 8) and cpu_arch in ('aarch64', 'arm64'): return '1.19.0'
    if py_version >= (3, 8): return '1.18.0'
    if py_version >= (3, 7): return '1.15.0'
    if py_version >= (3, 6): return '1.12.0'
    if py_version >= (3, 5): return '1.11.0'
    return '1.10.0'

if __name__ == '__main__':
    print(get_old_numpy_version())
