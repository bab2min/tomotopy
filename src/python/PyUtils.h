#pragma once
#include <Python.h>
#include <type_traits>
#include <vector>
#include <tuple>
#include <limits>
#include <exception>

namespace py
{
	using namespace std;

	struct AutoReleaser
	{
		PyObject*& obj;
		AutoReleaser(PyObject*& _obj) : obj(_obj) {}
		~AutoReleaser()
		{
			Py_XDECREF(obj);
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

	template<typename T>
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

	template<typename T>
	inline vector<T> makeIterToVector(PyObject *iter)
	{
		PyObject* item;
		vector<T> v;
		while ((item = PyIter_Next(iter)))
		{
			AutoReleaser ar{ item };
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
}