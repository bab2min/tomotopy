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

	vector<string> makeIterToVector(PyObject *iter)
	{
		PyObject* item;
		vector<string> v;
		while (item = PyIter_Next(iter))
		{
			AutoReleaser ar{ item };
			const char* str = PyUnicode_AsUTF8(item);
			if (!str) throw bad_exception{};
			v.emplace_back(str);
		}
		if (PyErr_Occurred())
		{
			throw bad_exception{};
		}
		return v;
	}

	template<typename _Ty>
	typename enable_if<numeric_limits<_Ty>::is_integer, PyObject*>::type buildPyValue(_Ty v)
	{
		return Py_BuildValue("n", v);
	}

	template<typename _Ty>
	typename enable_if<is_enum<_Ty>::value, PyObject*>::type buildPyValue(_Ty v)
	{
		return Py_BuildValue("n", (size_t)v);
	}

	PyObject* buildPyValue(float v)
	{
		return Py_BuildValue("f", v);
	}

	PyObject* buildPyValue(double v)
	{
		return Py_BuildValue("d", v);
	}

	PyObject* buildPyValue(const string& v)
	{
		return Py_BuildValue("s", v.c_str());
	}

	PyObject* buildPyValue(nullptr_t)
	{
		Py_INCREF(Py_None);
		return Py_None;
	}

	template<typename _Ty1, typename _Ty2>
	PyObject* buildPyValue(const pair<_Ty1, _Ty2>& v)
	{
		auto ret = PyTuple_New(2);
		size_t id = 0;
		PyTuple_SetItem(ret, id++, buildPyValue(get<0>(v)));
		PyTuple_SetItem(ret, id++, buildPyValue(get<1>(v)));
		return ret;
	}

	template<typename _Ty1, typename _Ty2>
	PyObject* buildPyValue(pair<_Ty1, _Ty2>&& v)
	{
		auto ret = PyTuple_New(2);
		size_t id = 0;
		PyTuple_SetItem(ret, id++, buildPyValue(get<0>(v)));
		PyTuple_SetItem(ret, id++, buildPyValue(get<1>(v)));
		return ret;
	}

	template<typename _Ty>
	PyObject* buildPyValue(const vector<_Ty>& v)
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
	PyObject* buildPyValue(vector<_Ty>&& v)
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
	typename enable_if<is_iterator<_Ty>::value, PyObject*>::type buildPyValue(_Ty first, _Ty last)
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
	typename enable_if<is_iterator<_Ty>::value, PyObject*>::type buildPyValueTransform(_Ty first, _Ty last, _Tx tx)
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