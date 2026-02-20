#include <stdlib.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>

#ifdef _DEBUG
#undef _DEBUG
#include <Python.h>
#define _DEBUG
#else
#include <Python.h>
#endif

#if defined(_WIN32) || defined(_WIN64)
#include <Windows.h>
#include <immintrin.h>

#define cpuid(info, x)    __cpuidex(info, x, 0)

#elif defined(__unix__) || defined(__APPLE__) || defined(__MACOSX)
#include <cpuid.h>

//#if __GNUC__ > 4 || __GNUC__ == 4 && __GNUC_MINOR__ >= 4
static inline unsigned long long _xgetbv(unsigned int index) {
	unsigned int eax, edx;
	__asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
	return ((unsigned long long)edx << 32) | eax;
}
/*#else
#define _xgetbv(x) 0
#endif*/

void cpuid(int info[4], int InfoType) {
	__cpuid_count(InfoType, 0, info[0], info[1], info[2], info[3]);
}
#endif

PyMODINIT_FUNC PyInit__tomotopy()
{
	using namespace std;

	bool sse2 = false, avx2 = false, avx512 = false;
	bool env_sse2 = false, env_avx2 = false, env_avx512 = false;

	string isaEnv;
	const char* p = getenv("TOMOTOPY_ISA");
	if (p) isaEnv = p;
	transform(isaEnv.begin(), isaEnv.end(), isaEnv.begin(), ::tolower);

	istringstream iss{ isaEnv };
	string item;

	while (getline(iss, item, ','))
	{
		if (item == "avx512") env_avx512 = true;
		else if (item == "avx2") env_avx2 = true;
		else if (item == "sse2") env_sse2 = true;
		else if (item == "none");
		else fprintf(stderr, "Unknown ISA option '%s' ignored.\n", item.c_str());
	}

	if (!env_sse2 && !env_avx2 && !env_avx512)
	{
		env_sse2 = true;
		env_avx2 = true;
		env_avx512 = true;
	}

	int info[4];
	cpuid(info, 0);
	int nIds = info[0];

	cpuid(info, 0x80000000);
	unsigned nExIds = info[0];

	unsigned long long xcrFeatureMask = 0;
    if (nIds >= 1) {
        cpuid(info, 1);
        sse2 = (info[3] & ((int)1 << 26)) != 0;
		if ((info[2] & (1 << 27)) && ((info[2] & ((int)1 << 28)) != 0))
		{
			xcrFeatureMask = _xgetbv(0);
		}
    }
    if (nIds >= 7) {
        cpuid(info, 7);
        bool avx = (xcrFeatureMask & 0x6) == 0x6;
        avx2 = avx && (info[1] & ((int)1 << 5)) != 0;
		// AVX-512F: bit 16 of EBX
		// Also check XCR0 for opmask (bit 5), ZMM_Hi256 (bit 6), Hi16_ZMM (bit 7)
		bool hasAVX512F = (info[1] & ((int)1 << 16)) != 0;
		bool osSupportsAVX512 = (xcrFeatureMask & 0xE6) == 0xE6;
		avx512 = avx && hasAVX512F && osSupportsAVX512;
    }

	PyObject* module = nullptr;
	vector<string> triedModules;
	if (!module && avx512 && env_avx512)
	{
		module = PyImport_ImportModule("_tomotopy_avx512");
		if (!module)
		{
			PyErr_Clear();
			triedModules.emplace_back("avx512");
		}
	}
	if (!module && avx2 && env_avx2)
	{
		module = PyImport_ImportModule("_tomotopy_avx2");
		if (!module)
		{
			PyErr_Clear();
			triedModules.emplace_back("avx2");
		}
	}
	if (!module && sse2 && env_sse2)
	{
		module = PyImport_ImportModule("_tomotopy_sse2");
		if (!module)
		{
			PyErr_Clear();
			triedModules.emplace_back("sse2");
		}
	}
	if (!module)
	{
		module = PyImport_ImportModule("_tomotopy_none");
		if (!module)
		{
			PyErr_Clear();
			triedModules.emplace_back("none");
		}
	}

	if (!module)
	{
		string err = "No module named any of ";
		for (auto& s : triedModules) err += "'_tomotopy_" + s + "', ";
		err.pop_back(); err.pop_back();
		PyErr_SetString(PyExc_ModuleNotFoundError, err.c_str());
	}
	return module;
}
