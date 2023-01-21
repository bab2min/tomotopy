#define MAIN_MODULE
#include "module.h"
#include "label.h"
#include "utils.h"
#include "coherence.h"

using namespace std;

PyObject* gModule;

#ifdef TOMOTOPY_ISA
#define TO_STR(name) #name
#define TO_STR_WRAP(name) TO_STR(name)
#define TOMOTOPY_ISA_STR TO_STR_WRAP(TOMOTOPY_ISA)
static const char* isa_str = TOMOTOPY_ISA_STR;
#else
static const char* isa_str = "none";
#endif

void char2Byte(const char* strBegin, const char* strEnd, vector<uint32_t>& startPos, vector<uint16_t>& length)
{
	if (strBegin == strEnd) return;
	vector<size_t> charPos;
	auto it = strBegin;
	for (; it != strEnd; )
	{
		charPos.emplace_back(it - strBegin);
		uint8_t c = *it;
		if ((c & 0xF8) == 0xF0)
		{
			it += 4;
		}
		else if ((c & 0xF0) == 0xE0)
		{
			it += 3;
		}
		else if ((c & 0xE0) == 0xC0)
		{
			it += 2;
		}
		else if ((c & 0x80))
		{
			throw std::runtime_error{ "utf-8 decoding error" };
		}
		else it += 1;
	}
	charPos.emplace_back(strEnd - strBegin);

	for (size_t i = 0; i < startPos.size(); ++i)
	{
		size_t s = startPos[i], e = (size_t)startPos[i] + length[i];
		startPos[i] = charPos[s];
		length[i] = charPos[e] - charPos[s];
	}
}

void char2Byte(const string& str, vector<uint32_t>& startPos, vector<uint16_t>& length)
{
	return char2Byte(&str[0], &str[0] + str.size(), startPos, length);
}

void char2Byte(const tomoto::SharedString& str, vector<uint32_t>& startPos, vector<uint16_t>& length)
{
	return char2Byte(str.begin(), str.end(), startPos, length);
}

void TopicModelObject::dealloc(TopicModelObject* self)
{
	DEBUG_LOG("TopicModelObject Dealloc " << self);
	if (self->inst)
	{
		delete self->inst;
	}
	Py_XDECREF(self->initParams);
	Py_TYPE(self)->tp_free((PyObject*)self);
}

PyMODINIT_FUNC MODULE_NAME()
{
	import_array();

	static PyModuleDef mod =
	{
		PyModuleDef_HEAD_INIT,
		"tomotopy",
		"Tomoto Module for Python",
		-1,
		nullptr,
	};

	gModule = PyModule_Create(&mod);
	if (!gModule) return nullptr;

	if (PyType_Ready(&LDA_type) < 0) return nullptr;
	Py_INCREF(&LDA_type);
	PyModule_AddObject(gModule, "LDAModel", (PyObject*)&LDA_type);

#ifdef TM_DMR
	if (PyType_Ready(&DMR_type) < 0) return nullptr;
	Py_INCREF(&DMR_type);
	PyModule_AddObject(gModule, "DMRModel", (PyObject*)&DMR_type);
#endif
#ifdef TM_HDP
	if (PyType_Ready(&HDP_type) < 0) return nullptr;
	Py_INCREF(&HDP_type);
	PyModule_AddObject(gModule, "HDPModel", (PyObject*)&HDP_type);
#endif
#ifdef TM_MGLDA
	if (PyType_Ready(&MGLDA_type) < 0) return nullptr;
	Py_INCREF(&MGLDA_type);
	PyModule_AddObject(gModule, "MGLDAModel", (PyObject*)&MGLDA_type);
#endif
#ifdef TM_PA
	if (PyType_Ready(&PA_type) < 0) return nullptr;
	Py_INCREF(&PA_type);
	PyModule_AddObject(gModule, "PAModel", (PyObject*)&PA_type);
#endif
#ifdef TM_HPA
	if (PyType_Ready(&HPA_type) < 0) return nullptr;
	Py_INCREF(&HPA_type);
	PyModule_AddObject(gModule, "HPAModel", (PyObject*)&HPA_type);
#endif
#ifdef TM_HLDA
	if (PyType_Ready(&HLDA_type) < 0) return nullptr;
	Py_INCREF(&HLDA_type);
	PyModule_AddObject(gModule, "HLDAModel", (PyObject*)&HLDA_type);
#endif
#ifdef TM_CT
	if (PyType_Ready(&CT_type) < 0) return nullptr;
	Py_INCREF(&CT_type);
	PyModule_AddObject(gModule, "CTModel", (PyObject*)&CT_type);
#endif
#ifdef TM_SLDA
	if (PyType_Ready(&SLDA_type) < 0) return nullptr;
	Py_INCREF(&SLDA_type);
	PyModule_AddObject(gModule, "SLDAModel", (PyObject*)&SLDA_type);
#endif
#ifdef TM_LLDA
	if (PyType_Ready(&LLDA_type) < 0) return nullptr;
	Py_INCREF(&LLDA_type);
	PyModule_AddObject(gModule, "LLDAModel", (PyObject*)&LLDA_type);
#endif
#ifdef TM_PLDA
	if (PyType_Ready(&PLDA_type) < 0) return nullptr;
	Py_INCREF(&PLDA_type);
	PyModule_AddObject(gModule, "PLDAModel", (PyObject*)&PLDA_type);
#endif
#ifdef TM_DT
	if (PyType_Ready(&DT_type) < 0) return nullptr;
	Py_INCREF(&DT_type);
	PyModule_AddObject(gModule, "DTModel", (PyObject*)&DT_type);
#endif
#ifdef TM_GDMR
	if (PyType_Ready(&GDMR_type) < 0) return nullptr;
	Py_INCREF(&GDMR_type);
	PyModule_AddObject(gModule, "GDMRModel", (PyObject*)&GDMR_type);
#endif
#ifdef TM_PT
	if (PyType_Ready(&PT_type) < 0) return nullptr;
	Py_INCREF(&PT_type);
	PyModule_AddObject(gModule, "PTModel", (PyObject*)&PT_type);
#endif

#ifdef __AVX2__
	PyModule_AddStringConstant(gModule, "isa", "avx2");
#elif defined(__AVX__)
	PyModule_AddStringConstant(gModule, "isa", "avx");
#elif defined(__SSE2__) || defined(__x86_64__) || defined(_WIN64)
	PyModule_AddStringConstant(gModule, "isa", "sse2");
#else
	PyModule_AddStringConstant(gModule, "isa", isa_str);
#endif
	addLabelTypes(gModule);
	addUtilsTypes(gModule);
	addCoherenceTypes(gModule);

	return gModule;
}
