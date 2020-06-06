#pragma once
#include <type_traits>
#include <vector>
#include <tuple>
#include <set>
#include <limits>
#include <exception>
#include <string>
#include <iostream>
#include <cstring>

#include <Python.h>
#include <frameobject.h>
#ifdef MAIN_MODULE
#else
#define NO_IMPORT_ARRAY
#endif
#define PY_ARRAY_UNIQUE_SYMBOL TOMOTOPY_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

namespace py
{
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

		PyObject* release()
		{
			auto o = obj;
			obj = nullptr;
			return o;
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

	template<typename T,
		typename std::enable_if<!std::is_integral<T>::value, int>::type = 0>
	inline T makeObjectToCType(PyObject *obj)
	{
	}

	template<>
	inline std::string makeObjectToCType<std::string>(PyObject *obj)
	{
		const char* str = PyUnicode_AsUTF8(obj);
		if (!str) throw std::bad_exception{};
		return str;
	}

	template<>
	inline float makeObjectToCType<float>(PyObject *obj)
	{
		float d = PyFloat_AsDouble(obj);
		if (d == -1 && PyErr_Occurred()) throw std::bad_exception{};
		return d;
	}

	template<>
	inline double makeObjectToCType<double>(PyObject *obj)
	{
		double d = PyFloat_AsDouble(obj);
		if (d == -1 && PyErr_Occurred()) throw std::bad_exception{};
		return d;
	}

	template<typename T,
		typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
	inline T makeObjectToCType(PyObject *obj)
	{
		long long v = PyLong_AsLongLong(obj);
		if (v == -1 && PyErr_Occurred()) throw std::bad_exception{};
		return (T)v;
	}

	template<typename T>
	inline std::vector<T> makeIterToVector(PyObject *iter)
	{
		UniqueObj item;
		std::vector<T> v;
		while ((item = PyIter_Next(iter)))
		{
			v.emplace_back(makeObjectToCType<T>(item));
		}
		if (PyErr_Occurred())
		{
			throw std::bad_exception{};
		}
		return v;
	}

	template<typename T>
	inline std::vector<std::string> makeIterToStringVector(PyObject *iter)
	{
		UniqueObj item;
		std::vector<std::string> v;
		while ((item = PyIter_Next(iter)))
		{
			v.emplace_back(std::to_string(makeObjectToCType<T>(item)));
		}
		if (PyErr_Occurred())
		{
			throw std::bad_exception{};
		}
		return v;
	}

	template<typename _Ty>
	inline typename std::enable_if<std::numeric_limits<_Ty>::is_integer, PyObject*>::type buildPyValue(_Ty v)
	{
		return Py_BuildValue("n", v);
	}

	template<typename _Ty>
	inline typename std::enable_if<std::is_enum<_Ty>::value, PyObject*>::type buildPyValue(_Ty v)
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

	inline PyObject* buildPyValue(const std::string& v)
	{
		return Py_BuildValue("s", v.c_str());
	}

	inline PyObject* buildPyValue(std::nullptr_t)
	{
		Py_INCREF(Py_None);
		return Py_None;
	}

	template<typename _Ty>
	inline typename std::enable_if<std::is_arithmetic<_Ty>::value, PyObject*>::type
		buildPyValue(const std::vector<_Ty>& v);

	template<typename _Ty>
	inline typename std::enable_if<!std::is_arithmetic<_Ty>::value, PyObject*>::type
		buildPyValue(const std::vector<_Ty>& v);

	template<typename _Ty>
	inline typename std::enable_if<!std::is_arithmetic<_Ty>::value, PyObject*>::type
		buildPyValue(std::vector<_Ty>&& v);

	template<typename _Ty1, typename _Ty2>
	inline PyObject* buildPyValue(const std::pair<_Ty1, _Ty2>& v)
	{
		auto ret = PyTuple_New(2);
		size_t id = 0;
		PyTuple_SetItem(ret, id++, buildPyValue(std::get<0>(v)));
		PyTuple_SetItem(ret, id++, buildPyValue(std::get<1>(v)));
		return ret;
	}

	template<typename _Ty1, typename _Ty2>
	inline PyObject* buildPyValue(std::pair<_Ty1, _Ty2>&& v)
	{
		auto ret = PyTuple_New(2);
		size_t id = 0;
		PyTuple_SetItem(ret, id++, buildPyValue(std::get<0>(v)));
		PyTuple_SetItem(ret, id++, buildPyValue(std::get<1>(v)));
		return ret;
	}

	namespace detail
	{
		template<typename _Ty>
		struct NpyType
		{
		};

		template<>
		struct NpyType<int8_t>
		{
			enum { 
				type = NPY_INT8,
				signed_type = type
			};
		};

		template<>
		struct NpyType<uint8_t>
		{
			enum { 
				type = NPY_UINT8,
				signed_type = NPY_INT8
			};
		};

		template<>
		struct NpyType<int16_t>
		{
			enum { 
				type = NPY_INT16,
				signed_type = type
			};
		};

		template<>
		struct NpyType<uint16_t>
		{
			enum { 
				type = NPY_UINT16,
				signed_type = NPY_INT16
			};
		};

		template<>
		struct NpyType<int32_t>
		{
			enum { 
				type = NPY_INT32,
				signed_type = type
			};
		};

		template<>
		struct NpyType<uint32_t>
		{
			enum { 
				type = NPY_UINT32,
				signed_type = NPY_INT32
			};
		};

		template<>
		struct NpyType<int64_t>
		{
			enum { 
				type = NPY_INT64,
				signed_type = type
			};
		};

		template<>
		struct NpyType<uint64_t>
		{
			enum { 
				type = NPY_UINT64, 
				signed_type = NPY_INT64
			};
		};

		template<>
		struct NpyType<float>
		{
			enum { 
				type = NPY_FLOAT, 
				signed_type = type 
			};
		};

		template<>
		struct NpyType<double>
		{
			enum { 
				type = NPY_DOUBLE, 
				signed_type = type 
			};
		};
	}

	struct cast_to_signed_t
	{
	};

	static constexpr cast_to_signed_t cast_to_signed{};

	template<typename _Ty>
	inline typename std::enable_if<std::is_arithmetic<_Ty>::value, PyObject*>::type
		buildPyValue(const std::vector<_Ty>& v)
	{
		npy_intp size = v.size();
		PyObject* obj = PyArray_EMPTY(1, &size, detail::NpyType<_Ty>::type, 0);
		std::memcpy(PyArray_DATA((PyArrayObject*)obj), v.data(), sizeof(_Ty) * size);
		return obj;
	}

	template<typename _Ty>
	inline typename std::enable_if<std::is_arithmetic<_Ty>::value, PyObject*>::type
		buildPyValue(const std::vector<_Ty>& v, cast_to_signed_t)
	{
		npy_intp size = v.size();
		PyObject* obj = PyArray_EMPTY(1, &size, detail::NpyType<_Ty>::signed_type, 0);
		std::memcpy(PyArray_DATA((PyArrayObject*)obj), v.data(), sizeof(_Ty) * size);
		return obj;
	}

	template<typename _Ty>
	inline typename std::enable_if<!std::is_arithmetic<_Ty>::value, PyObject*>::type 
		buildPyValue(const std::vector<_Ty>& v)
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
	inline typename std::enable_if<!std::is_arithmetic<_Ty>::value, PyObject*>::type 
		buildPyValue(std::vector<_Ty>&& v)
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
	inline typename std::enable_if<
		!std::is_arithmetic<typename std::iterator_traits<_Ty>::value_type>::value,
		PyObject*
	>::type buildPyValue(_Ty first, _Ty last)
	{
		auto ret = PyList_New(std::distance(first, last));
		size_t id = 0;
		for (; first != last; ++first)
		{
			PyList_SetItem(ret, id++, buildPyValue(*first));
		}
		return ret;
	}

	template<typename _Ty, typename _Tx>
	inline typename std::enable_if<
		!std::is_arithmetic<
			typename std::result_of<_Tx(typename std::iterator_traits<_Ty>::value_type)>::type
		>::value,
		PyObject*
	>::type buildPyValueTransform(_Ty first, _Ty last, _Tx tx)
	{
		auto ret = PyList_New(std::distance(first, last));
		size_t id = 0;
		for (; first != last; ++first)
		{
			PyList_SetItem(ret, id++, buildPyValue(tx(*first)));
		}
		return ret;
	}


	template<typename _Ty>
	inline typename std::enable_if<
		std::is_arithmetic<typename std::iterator_traits<_Ty>::value_type>::value,
		PyObject*
	>::type buildPyValue(_Ty first, _Ty last)
	{
		using value_type = typename std::iterator_traits<_Ty>::value_type;
		npy_intp size = std::distance(first, last);
		PyObject* ret = PyArray_EMPTY(1, &size, detail::NpyType<value_type>::type, 0);
		size_t id = 0;
		for (; first != last; ++first, ++id)
		{
			*(value_type*)PyArray_GETPTR1((PyArrayObject*)ret, id) = *first;
		}
		return ret;
	}

	template<typename _Ty, typename _Tx>
	inline typename std::enable_if<
		std::is_arithmetic<
			typename std::result_of<_Tx(typename std::iterator_traits<_Ty>::value_type)>::type
		>::value,
		PyObject*
	>::type buildPyValueTransform(_Ty first, _Ty last, _Tx tx)
	{
		using value_type = typename std::iterator_traits<_Ty>::value_type;
		npy_intp size = std::distance(first, last);
		PyObject* ret = PyArray_EMPTY(1, &size, detail::NpyType<value_type>::type, 0);
		size_t id = 0;
		for (; first != last; ++first, ++id)
		{
			*(value_type*)PyArray_GETPTR1((PyArrayObject*)ret, id) = tx(*first);
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
