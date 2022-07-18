#pragma once

#include <fstream>
#include <iostream>

#ifdef _DEBUG
//#undef _DEBUG
#define DEBUG_LOG(t) do{ cerr << t << endl; }while(0)
#include "PyUtils.h"
//#define _DEBUG
#else 
#define DEBUG_LOG(t)
#include "PyUtils.h"
#endif

#include "../TopicModel/TopicModel.hpp"
#include "../Utils/serializer.hpp"
#include "docs.h"

void char2Byte(const std::string& str, std::vector<uint32_t>& startPos, std::vector<uint16_t>& length);

void char2Byte(const char* begin, const char* end, std::vector<uint32_t> & startPos, std::vector<uint16_t> & length);

#define DEFINE_GETTER_PROTOTYPE(PREFIX, GETTER)\
PyObject* PREFIX##_##GETTER(TopicModelObject *self, void *closure);

#define DEFINE_GETTER(BASE, PREFIX, GETTER)\
PyObject* PREFIX##_##GETTER(TopicModelObject *self, void *closure)\
{\
	try\
	{\
		if (!self->inst) throw runtime_error{ "inst is null" };\
		auto* inst = static_cast<BASE*>(self->inst);\
		return py::buildPyValue(inst->GETTER());\
	}\
	catch (const bad_exception&)\
	{\
		return nullptr;\
	}\
	catch (const exception& e)\
	{\
		PyErr_SetString(PyExc_Exception, e.what());\
		return nullptr;\
	}\
}\

#define DEFINE_SETTER_PROTOTYPE(PREFIX, SETTER)\
PyObject* PREFIX##_##SETTER(TopicModelObject *self, PyObject* val, void *closure);

#define DEFINE_SETTER_CHECKED_FLOAT(BASE, PREFIX, SETTER, PRED)\
int PREFIX##_##SETTER(TopicModelObject* self, PyObject* val, void* closure)\
{\
	try\
	{\
		if (!self->inst) throw runtime_error{ "inst is null" };\
		auto* inst = static_cast<BASE*>(self->inst);\
		auto value = PyFloat_AsDouble(val);\
		if (value == -1 && PyErr_Occurred()) throw bad_exception{};\
		if (!(PRED)) throw runtime_error{ #SETTER " must satify " #PRED };\
		inst->SETTER(value);\
	}\
	catch (const bad_exception&)\
	{\
		return -1;\
	}\
	catch (const exception& e)\
	{\
		PyErr_SetString(PyExc_Exception, e.what());\
		return -1;\
	}\
	return 0;\
}

#define DEFINE_SETTER_NON_NEGATIVE_INT(BASE, PREFIX, SETTER)\
int PREFIX##_##SETTER(TopicModelObject* self, PyObject* val, void* closure)\
{\
	try\
	{\
		if (!self->inst) throw runtime_error{ "inst is null" };\
		auto* inst = static_cast<BASE*>(self->inst);\
		auto v = PyLong_AsLong(val);\
		if (v == -1 && PyErr_Occurred()) throw bad_exception{};\
		if (v < 0) throw runtime_error{ #SETTER " must >= 0" };\
		inst->SETTER(v);\
	}\
	catch (const bad_exception&)\
	{\
		return -1;\
	}\
	catch (const exception& e)\
	{\
		PyErr_SetString(PyExc_Exception, e.what());\
		return -1;\
	}\
	return 0;\
}


#define DEFINE_LOADER(PREFIX, TYPE) \
PyObject* PREFIX##_load(PyObject*, PyObject* args, PyObject *kwargs)\
{\
	const char* filename;\
	static const char* kwlist[] = { "filename", nullptr };\
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s", (char**)kwlist, &filename)) return nullptr;\
	try\
	{\
		ifstream str{ filename, ios_base::binary };\
		if (!str) throw ios_base::failure{ std::string("cannot open file '") + filename + std::string("'") };\
		for (size_t i = 0; i < (size_t)tomoto::TermWeight::size; ++i)\
		{\
			str.seekg(0);\
			py::UniqueObj args{ Py_BuildValue("(n)", i) };\
			auto* p = PyObject_CallObject((PyObject*)&TYPE, args);\
			try\
			{\
				vector<uint8_t> extra_data;\
				((TopicModelObject*)p)->inst->loadModel(str, &extra_data);\
				if (!extra_data.empty())\
				{\
					py::UniqueObj pickle{ PyImport_ImportModule("pickle") };\
					PyObject* pickle_dict{ PyModule_GetDict(pickle) };\
					py::UniqueObj bytes{ PyBytes_FromStringAndSize((const char*)extra_data.data(), extra_data.size()) };\
					py::UniqueObj args{ Py_BuildValue("(O)", bytes.get()) };\
					Py_XDECREF(((TopicModelObject*)p)->initParams);\
					((TopicModelObject*)p)->initParams = PyObject_CallObject(\
						PyDict_GetItemString(pickle_dict, "loads"),\
						args\
					);\
				}\
			}\
			catch (const tomoto::serializer::UnfitException&)\
			{\
				Py_XDECREF(p);\
				continue;\
			}\
			((TopicModelObject*)p)->isPrepared = true;\
			return p;\
		}\
		throw runtime_error{ std::string("'") + filename + std::string("' is not valid model file") };\
	}\
	catch (const bad_exception&)\
	{\
	}\
	catch (const ios_base::failure& e)\
	{\
		PyErr_SetString(PyExc_OSError, e.what());\
	}\
	catch (const exception& e)\
	{\
		PyErr_SetString(PyExc_Exception, e.what());\
	}\
	return nullptr;\
}\
\
PyObject* PREFIX##_loads(PyObject*, PyObject* args, PyObject *kwargs)\
{\
	Py_buffer data;\
	static const char* kwlist[] = { "data", nullptr };\
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "y*", (char**)kwlist, &data)) return nullptr;\
	try\
	{\
		tomoto::serializer::imstream str{ (const char*)data.buf, data.len };\
		for (size_t i = 0; i < (size_t)tomoto::TermWeight::size; ++i)\
		{\
			str.seekg(0);\
			py::UniqueObj args{ Py_BuildValue("(n)", i) };\
			auto* p = PyObject_CallObject((PyObject*)&TYPE, args);\
			try\
			{\
				vector<uint8_t> extra_data;\
				((TopicModelObject*)p)->inst->loadModel(str, &extra_data);\
				if (!extra_data.empty())\
				{\
					py::UniqueObj pickle{ PyImport_ImportModule("pickle") };\
					PyObject* pickle_dict{ PyModule_GetDict(pickle) };\
					py::UniqueObj bytes{ PyBytes_FromStringAndSize((const char*)extra_data.data(), extra_data.size()) };\
					py::UniqueObj args{ Py_BuildValue("(O)", bytes.get()) };\
					Py_XDECREF(((TopicModelObject*)p)->initParams);\
					((TopicModelObject*)p)->initParams = PyObject_CallObject(\
						PyDict_GetItemString(pickle_dict, "loads"),\
						args\
					);\
				}\
			}\
			catch (const tomoto::serializer::UnfitException&)\
			{\
				Py_XDECREF(p);\
				continue;\
			}\
			((TopicModelObject*)p)->isPrepared = true;\
			return p;\
		}\
		throw runtime_error{ "`data` is not valid model file" };\
	}\
	catch (const bad_exception&)\
	{\
	}\
	catch (const ios_base::failure& e)\
	{\
		PyErr_SetString(PyExc_OSError, e.what());\
	}\
	catch (const exception& e)\
	{\
		PyErr_SetString(PyExc_Exception, e.what());\
	}\
	return nullptr;\
}\


extern PyObject* gModule;
struct TopicModelObject;

struct TopicModelTypeObject : public PyTypeObject
{
	using MiscConverter = tomoto::RawDoc::MiscType(TopicModelObject*, const tomoto::RawDoc::MiscType&);
	MiscConverter* miscConverter = nullptr;
	TopicModelTypeObject(const PyTypeObject& _tp = {}, MiscConverter* _miscConverter = nullptr)
		: PyTypeObject{ _tp }, miscConverter{ _miscConverter }
	{
	}
};

template<typename Ty, typename FailMsg>
inline std::vector<Ty> broadcastObj(PyObject* obj, size_t k, FailMsg&& msg)
{
	try
	{
		std::vector<Ty> ret;
		try
		{
			ret = py::toCpp<std::vector<Ty>>(obj);
			if (ret.size() != k) throw py::ConversionFail{ "" };
		}
		catch (const py::ConversionFail&)
		{
			PyErr_Clear();
			ret.emplace_back(py::toCpp<Ty>(obj));
		}
		return ret;
	}
	catch (const py::ConversionFail&)
	{
		throw py::ConversionFail{ std::forward<FailMsg>(msg) };
	}
}

namespace py
{
	struct RawDocVarToPy
	{
		PyObject* ret = nullptr;

		template<typename _Ty>
		void operator()(const _Ty& s)
		{
			ret = buildPyValue(s);
		}

		void operator()(const std::shared_ptr<void>& s)
		{
			if (s)
			{
				ret = (PyObject*)s.get();
				Py_INCREF(ret);
			}
		}
	};

	template<>
	struct ValueBuilder<tomoto::RawDoc::Var>
	{
		PyObject* operator()(const tomoto::RawDoc::Var& v)
		{
			RawDocVarToPy visitor;
			mapbox::util::apply_visitor(visitor, v);
			return visitor.ret;
		}

		template<typename _FailMsg>
		tomoto::RawDoc::Var _toCpp(PyObject* obj, _FailMsg&& failMsg)
		{
			tomoto::RawDoc::Var ret;
			Py_INCREF(obj);
			ret = std::shared_ptr<void>{ obj, [](void* p)
			{
				Py_XDECREF(p);
			} };
			return ret;
		}
	};
}

template<typename _Ty>
_Ty getValueFromMisc(const char* key, const tomoto::RawDoc::MiscType& misc, const char* failMsg)
{
	auto it = misc.find(key);
	if (it == misc.end()) throw std::runtime_error{ failMsg + std::string{ " (the required value was not given)" } };
	auto obj = (PyObject*)it->second.template get<std::shared_ptr<void>>().get();
	return py::toCpp<_Ty>(obj, 
		[=](){ return failMsg + (" (given " + py::repr(obj) + ")"); }
	);
}

template<typename _Ty>
_Ty getValueFromMiscDefault(const char* key, const tomoto::RawDoc::MiscType& misc, const char* failMsg, const _Ty& def = {})
{
	auto it = misc.find(key);
	if (it == misc.end()) return def;
	auto obj = (PyObject*)it->second.template get<std::shared_ptr<void>>().get();
	return py::toCpp<_Ty>(obj,
		[=]() { return failMsg + (" (given " + py::repr(obj) + ")"); }
	);
}

extern TopicModelTypeObject LDA_type;
extern TopicModelTypeObject DMR_type;
extern TopicModelTypeObject HDP_type;
extern TopicModelTypeObject MGLDA_type;
extern TopicModelTypeObject PA_type;
extern TopicModelTypeObject HPA_type;
extern TopicModelTypeObject CT_type;
extern TopicModelTypeObject SLDA_type;
extern TopicModelTypeObject HLDA_type;
extern TopicModelTypeObject LLDA_type;
extern TopicModelTypeObject PLDA_type;
extern TopicModelTypeObject DT_type;
extern TopicModelTypeObject GDMR_type;
extern TopicModelTypeObject PT_type;

struct TopicModelObject
{
	PyObject_HEAD;
	tomoto::ITopicModel* inst;
	bool isPrepared, seedGiven;
	size_t minWordCnt, minWordDf;
	size_t removeTopWord;
	PyObject* initParams;
	static void dealloc(TopicModelObject* self);
};

DEFINE_GETTER_PROTOTYPE(LDA, getK);
DEFINE_GETTER_PROTOTYPE(LDA, getAlpha);
DEFINE_GETTER_PROTOTYPE(LDA, getEta);

inline std::string getVersion()
{
	py::UniqueObj mod{ PyImport_ImportModule("tomotopy") };
	if (!mod) throw std::bad_exception{};
	PyObject* mod_dict = PyModule_GetDict(mod);
	if (!mod_dict) throw std::bad_exception{};
	PyObject* version = PyDict_GetItemString(mod_dict, "__version__");
	return PyUnicode_AsUTF8(version);
}

inline tomoto::RawDoc buildRawDoc(PyObject* words)
{
	tomoto::RawDoc raw;
	raw.rawWords = py::toCpp<std::vector<std::string>>(words, "`words` must be an iterable of str.");
	return raw;
}


#define TM_DMR
#define TM_GDMR
#define TM_HDP
#define TM_MGLDA
#define TM_PA
#define TM_HPA
#define TM_CT
#define TM_SLDA
#define TM_HLDA
#define TM_LLDA
#define TM_PLDA
#define TM_DT
#define TM_PT
