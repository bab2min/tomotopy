#pragma once
#include <Python.h>
#include <frameobject.h>
#include <type_traits>
#include <vector>
#include <tuple>
#include <set>
#include <limits>
#include <exception>
#include <string>
#include <iostream>

namespace py
{
	using namespace std;

	struct UniqueObj
	{
		PyObject* obj;
		UniqueObj(PyObject* _obj = nullptr) : obj(_obj) {}
		~UniqueObj()
		{
			Py_XDECREF(obj);
		}

		UniqueObj(const UniqueObj&) = delete;
		UniqueObj& operator=(const UniqueObj&) = delete;

		UniqueObj(UniqueObj&& o)
		{
			std::swap(obj, o.obj);
		}

		UniqueObj& operator=(UniqueObj&& o)
		{
			std::swap(obj, o.obj);
			return *this;
		}

		PyObject* get() const
		{
			return obj;
		}

		operator bool() const
		{
			return !!obj;
		}

		operator PyObject*() const
		{
			return obj;
		}
	};

	template<typename T, typename = void>
	struct is_iterator
	{
		static constexpr bool value = false;
	};

	template<typename T>
	struct is_iterator<T, typename std::enable_if<!std::is_same<typename std::iterator_traits<T>::value_type, void>::value>::type>
	{
		static constexpr bool value = true;
	};

	template<typename T,
		typename std::enable_if<!std::is_integral<T>::value, int>::type = 0>
	inline T makeObjectToCType(PyObject *obj)
	{
	}

	template<>
	inline string makeObjectToCType<string>(PyObject *obj)
	{
		const char* str = PyUnicode_AsUTF8(obj);
		if (!str) throw bad_exception{};
		return str;
	}

	template<>
	inline float makeObjectToCType<float>(PyObject *obj)
	{
		float d = PyFloat_AsDouble(obj);
		if (d == -1 && PyErr_Occurred()) throw bad_exception{};
		return d;
	}

	template<>
	inline double makeObjectToCType<double>(PyObject *obj)
	{
		double d = PyFloat_AsDouble(obj);
		if (d == -1 && PyErr_Occurred()) throw bad_exception{};
		return d;
	}

	template<typename T,
		typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
	inline T makeObjectToCType(PyObject *obj)
	{
		long long v = PyLong_AsLongLong(obj);
		if (v == -1 && PyErr_Occurred()) throw bad_exception{};
		return (T)v;
	}

	template<typename T>
	inline vector<T> makeIterToVector(PyObject *iter)
	{
		UniqueObj item;
		vector<T> v;
		while ((item = PyIter_Next(iter)))
		{
			v.emplace_back(makeObjectToCType<T>(item));
		}
		if (PyErr_Occurred())
		{
			throw bad_exception{};
		}
		return v;
	}

	template<typename _Ty>
	inline typename enable_if<numeric_limits<_Ty>::is_integer, PyObject*>::type buildPyValue(_Ty v)
	{
		return Py_BuildValue("n", v);
	}

	template<typename _Ty>
	inline typename enable_if<is_enum<_Ty>::value, PyObject*>::type buildPyValue(_Ty v)
	{
		return Py_BuildValue("n", (size_t)v);
	}

	inline PyObject* buildPyValue(float v)
	{
		return Py_BuildValue("f", v);
	}

	inline PyObject* buildPyValue(double v)
	{
		return Py_BuildValue("d", v);
	}

	inline PyObject* buildPyValue(const string& v)
	{
		return Py_BuildValue("s", v.c_str());
	}

	inline PyObject* buildPyValue(nullptr_t)
	{
		Py_INCREF(Py_None);
		return Py_None;
	}

	template<typename _Ty>
	inline PyObject* buildPyValue(const vector<_Ty>& v);

	template<typename _Ty>
	inline PyObject* buildPyValue(vector<_Ty>&& v);

	template<typename _Ty1, typename _Ty2>
	inline PyObject* buildPyValue(const pair<_Ty1, _Ty2>& v)
	{
		auto ret = PyTuple_New(2);
		size_t id = 0;
		PyTuple_SetItem(ret, id++, buildPyValue(get<0>(v)));
		PyTuple_SetItem(ret, id++, buildPyValue(get<1>(v)));
		return ret;
	}

	template<typename _Ty1, typename _Ty2>
	inline PyObject* buildPyValue(pair<_Ty1, _Ty2>&& v)
	{
		auto ret = PyTuple_New(2);
		size_t id = 0;
		PyTuple_SetItem(ret, id++, buildPyValue(get<0>(v)));
		PyTuple_SetItem(ret, id++, buildPyValue(get<1>(v)));
		return ret;
	}

	template<typename _Ty>
	inline PyObject* buildPyValue(const vector<_Ty>& v)
	{
		auto ret = PyList_New(v.size());
		size_t id = 0;
		for (auto& e : v)
		{
			PyList_SetItem(ret, id++, buildPyValue(e));
		}
		return ret;
	}

	template<typename _Ty>
	inline PyObject* buildPyValue(vector<_Ty>&& v)
	{
		auto ret = PyList_New(v.size());
		size_t id = 0;
		for (auto& e : v)
		{
			PyList_SetItem(ret, id++, buildPyValue(e));
		}
		return ret;
	}

	template<typename _Ty>
	inline typename enable_if<is_iterator<_Ty>::value, PyObject*>::type buildPyValue(_Ty first, _Ty last)
	{
		auto ret = PyList_New(distance(first, last));
		size_t id = 0;
		for (; first != last; ++first)
		{
			PyList_SetItem(ret, id++, buildPyValue(*first));
		}
		return ret;
	}

	template<typename _Ty, typename _Tx>
	inline typename enable_if<is_iterator<_Ty>::value, PyObject*>::type buildPyValueTransform(_Ty first, _Ty last, _Tx tx)
	{
		auto ret = PyList_New(distance(first, last));
		size_t id = 0;
		for (; first != last; ++first)
		{
			PyList_SetItem(ret, id++, buildPyValue(tx(*first)));
		}
		return ret;
	}

	class WarningLog
	{
		std::set<std::tuple<std::string, int, std::string>> printed;

		WarningLog()
		{
		}
	public:
		static WarningLog& get()
		{
			thread_local WarningLog inst;
			return inst;
		}

		void printOnce(std::ostream& ostr, const std::string& msg)
		{
			auto frame = PyEval_GetFrame();
			auto key = std::make_tuple(
				std::string{ PyUnicode_AsUTF8(frame->f_code->co_filename) }, 
				PyFrame_GetLineNumber(frame),
				msg);

			if (!printed.count(key))
			{
				ostr << std::get<0>(key) << "(" << std::get<1>(key) << "): " << std::get<2>(key) << std::endl;
				printed.insert(key);
			}
		}
	};
}

#define PRINT_WARN(msg) do{ py::WarningLog::get().printOnce(std::cerr, msg); } while(0)
