#pragma once
#include <type_traits>
#include <vector>
#include <unordered_map>
#include <tuple>
#include <set>
#include <limits>
#include <exception>
#include <string>
#include <functional>
#include <iostream>
#include <cstring>

#ifdef _DEBUG
#undef _DEBUG
#include <Python.h>
#define _DEBUG
#else
#include <Python.h>
#endif

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
		PyObject* obj = nullptr;
		explicit UniqueObj(PyObject* _obj = nullptr) : obj(_obj) {}
		~UniqueObj()
		{
			Py_XDECREF(obj);
		}

		UniqueObj(const UniqueObj&) = delete;
		UniqueObj& operator=(const UniqueObj&) = delete;

		UniqueObj(UniqueObj&& o) noexcept
		{
			std::swap(obj, o.obj);
		}

		UniqueObj& operator=(UniqueObj&& o) noexcept
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

	class ExcPropagation : public std::runtime_error
	{
	public:
		ExcPropagation() : std::runtime_error{ "" }
		{
		}
	};

	class BaseException : public std::runtime_error
	{
	public:
		using std::runtime_error::runtime_error;

		virtual PyObject* pytype() const
		{
			return PyExc_BaseException;
		}
	};

	class Exception : public BaseException
	{
	public:
		using BaseException::BaseException;

		virtual PyObject* pytype() const
		{
			return PyExc_Exception;
		}
	};

	class StopIteration : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_StopIteration;
		}
	};

	class StopAsyncIteration : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_StopAsyncIteration;
		}
	};

	class ArithmeticError : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_ArithmeticError;
		}
	};

	class AssertionError : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_AssertionError;
		}
	};

	class AttributeError : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_AttributeError;
		}
	};

	class BufferError : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_BufferError;
		}
	};

	class EOFError : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_EOFError;
		}
	};

	class ImportError : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_ImportError;
		}
	};

	class LookupError : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_LookupError;
		}
	};

	class IndexError : public LookupError
	{
	public:
		using LookupError::LookupError;

		virtual PyObject* pytype() const
		{
			return PyExc_IndexError;
		}
	};

	class KeyError : public LookupError
	{
	public:
		using LookupError::LookupError;

		virtual PyObject* pytype() const
		{
			return PyExc_KeyError;
		}
	};

	class MemoryError : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_MemoryError;
		}
	};

	class NameError : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_NameError;
		}
	};

	class OSError : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_OSError;
		}
	};

	class ReferenceError : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_ReferenceError;
		}
	};

	class RuntimeError : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_RuntimeError;
		}
	};

	class SyntaxError : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_SyntaxError;
		}
	};

	class SystemError : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_SystemError;
		}
	};

	class TypeError : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_TypeError;
		}
	};

	class ValueError : public Exception
	{
	public:
		using Exception::Exception;

		virtual PyObject* pytype() const
		{
			return PyExc_ValueError;
		}
	};

	template<typename _Fn>
	auto handleExc(_Fn&& fn) 
		-> typename std::enable_if<std::is_pointer<decltype(fn())>::value, decltype(fn())>::type
	{
		try
		{
			return fn();
		}
		catch (const ExcPropagation&)
		{
		}
		catch (const BaseException& e)
		{
			PyErr_SetString(e.pytype(), e.what());
		}
		catch (const std::exception& e)
		{
			std::cerr << "Uncaughted c++ exception: " << e.what() << std::endl;
			PyErr_SetString(PyExc_RuntimeError, e.what());
		}
		return nullptr;
	}

	template<typename _Fn>
	auto handleExc(_Fn&& fn)
		-> typename std::enable_if<std::is_integral<decltype(fn())>::value, decltype(fn())>::type
	{
		try
		{
			return fn();
		}
		catch (const ExcPropagation&)
		{
		}
		catch (const BaseException& e)
		{
			PyErr_SetString(e.pytype(), e.what());
		}
		catch (const std::exception& e)
		{
			std::cerr << "Uncaughted c++ exception: " << e.what() << std::endl;
			PyErr_SetString(PyExc_RuntimeError, e.what());
		}
		return -1;
	}

	class ConversionFail : public ValueError
	{
	public:
		using ValueError::ValueError;

		template<typename _Ty, 
			typename = typename std::enable_if<std::is_constructible<std::function<std::string()>, _Ty>::value>::type
		>
		ConversionFail(_Ty&& callable) : ValueError{ callable() }
		{
		}
	};

	template<typename _Ty, typename = void>
	struct ValueBuilder;

	template<typename _Ty>
	inline PyObject* buildPyValue(_Ty&& v)
	{
		return ValueBuilder<
			typename std::remove_const<typename std::remove_reference<_Ty>::type>::type
		>{}(std::forward<_Ty>(v));
	}

	template<typename _Ty, typename _FailMsg>
	inline _Ty toCpp(PyObject* obj, _FailMsg&& fail)
	{
		if (!obj) throw ConversionFail{ std::forward<_FailMsg>(fail) };
		return ValueBuilder<_Ty>{}._toCpp(obj, std::forward<_FailMsg>(fail));
	}

	inline std::string repr(PyObject* o)
	{
		UniqueObj r{ PyObject_Repr(o) };
		return toCpp<std::string>(r, "");
	}

	template<typename _Ty>
	inline std::string reprFromCpp(_Ty&& o)
	{
		UniqueObj p{ py::buildPyValue(std::forward<_Ty>(o)) };
		UniqueObj r{ PyObject_Repr(p) };
		return toCpp<std::string>(r, "");
	}

	template<typename _Ty>
	inline _Ty toCpp(PyObject* obj)
	{
		if (!obj) throw ConversionFail{ "cannot convert null pointer into appropriate C++ type" };
		return ValueBuilder<_Ty>{}._toCpp(obj, [=](){ return "cannot convert " + repr(obj) + " into appropriate C++ type"; });
	}

	template<typename _Ty>
	struct ValueBuilder<_Ty, 
		typename std::enable_if<std::is_integral<_Ty>::value>::type>
	{
		PyObject* operator()(_Ty v)
		{
			return PyLong_FromLongLong(v);
		}

		template<typename _FailMsg>
		_Ty _toCpp(PyObject* obj, _FailMsg&& msg)
		{
			long long v = PyLong_AsLongLong(obj);
			if (v == -1 && PyErr_Occurred()) throw ConversionFail{ std::forward<_FailMsg>(msg) };
			return (_Ty)v;
		}
	};

	template<typename _Ty>
	struct ValueBuilder<_Ty,
		typename std::enable_if<std::is_enum<_Ty>::value>::type>
	{
		PyObject* operator()(_Ty v)
		{
			return PyLong_FromLongLong((long long)v);
		}

		template<typename _FailMsg>
		_Ty _toCpp(PyObject* obj, _FailMsg&& msg)
		{
			long long v = PyLong_AsLongLong(obj);
			if (v == -1 && PyErr_Occurred()) throw ConversionFail{ std::forward<_FailMsg>(msg) };
			return (_Ty)v;
		}
	};

	template<typename _Ty>
	struct ValueBuilder<_Ty,
		typename std::enable_if<std::is_floating_point<_Ty>::value>::type>
	{
		PyObject* operator()(_Ty v)
		{
			return PyFloat_FromDouble(v);
		}

		template<typename _FailMsg>
		_Ty _toCpp(PyObject* obj, _FailMsg&& msg)
		{
			double v = PyFloat_AsDouble(obj);
			if (v == -1 && PyErr_Occurred()) throw ConversionFail{ std::forward<_FailMsg>(msg) };
			return (_Ty)v;
		}
	};

	template<>
	struct ValueBuilder<std::string>
	{
		PyObject* operator()(const std::string& v)
		{
			return PyUnicode_FromStringAndSize(v.data(), v.size());
		}

		template<typename _FailMsg>
		std::string _toCpp(PyObject* obj, _FailMsg&& msg)
		{
			const char* str = PyUnicode_AsUTF8(obj);
			if (!str) throw ConversionFail{ std::forward<_FailMsg>(msg) };
			return str;
		}
	};

	template<>
	struct ValueBuilder<const char*>
	{
		PyObject* operator()(const char* v)
		{
			return PyUnicode_FromString(v);
		}

		template<typename _FailMsg>
		const char* _toCpp(PyObject* obj, _FailMsg&& msg)
		{
			const char* p = PyUnicode_AsUTF8(obj);
			if (!p) throw ConversionFail{ std::forward<_FailMsg>(msg) };
			return p;
		}
	};

	template<>
	struct ValueBuilder<bool>
	{
		PyObject* operator()(bool v)
		{
			return PyBool_FromLong(v);
		}

		template<typename _FailMsg>
		bool _toCpp(PyObject* obj, _FailMsg&&)
		{
			return !!PyObject_IsTrue(obj);
		}
	};

	template<>
	struct ValueBuilder<PyObject*>
	{
		PyObject* operator()(PyObject* v)
		{
			if (v)
			{
				Py_INCREF(v);
				return v;
			}
			else
			{
				Py_INCREF(Py_None);
				return Py_None;
			}
		}

		template<typename _FailMsg>
		PyObject* _toCpp(PyObject* obj, _FailMsg&&)
		{
			return obj;
		}
	};

	template<>
	struct ValueBuilder<UniqueObj>
	{
		PyObject* operator()(UniqueObj&& v)
		{
			if (v)
			{
				Py_INCREF(v);
				return v;
			}
			else
			{
				Py_INCREF(Py_None);
				return Py_None;
			}
		}
	};

	template<typename _Ty1, typename _Ty2>
	struct ValueBuilder<std::pair<_Ty1, _Ty2>>
	{
		PyObject* operator()(const std::pair<_Ty1, _Ty2>& v)
		{
			PyObject* ret = PyTuple_New(2);
			size_t id = 0;
			PyTuple_SetItem(ret, id++, buildPyValue(std::get<0>(v)));
			PyTuple_SetItem(ret, id++, buildPyValue(std::get<1>(v)));
			return ret;
		}

		template<typename _FailMsg>
		std::pair<_Ty1, _Ty2> _toCpp(PyObject* obj, _FailMsg&&)
		{
			if (PyTuple_Size(obj) != 2) throw ConversionFail{ "input is not tuple with len=2" };
			return std::make_tuple(
				toCpp<_Ty1>(PyTuple_GetItem(obj, 0)),
				toCpp<_Ty2>(PyTuple_GetItem(obj, 1))
			);
		}
	};

	template<typename _Ty1, typename _Ty2>
	struct ValueBuilder<std::unordered_map<_Ty1, _Ty2>>
	{
		PyObject* operator()(const std::unordered_map<_Ty1, _Ty2>& v)
		{
			PyObject* ret = PyDict_New();
			for (auto& p : v)
			{
				py::UniqueObj key{ buildPyValue(p.first) }, val{ buildPyValue(p.second) };
				if(PyDict_SetItem(ret, key, val)) return nullptr;
			}
			return ret;
		}

		template<typename _FailMsg>
		std::unordered_map<_Ty1, _Ty2> _toCpp(PyObject* obj, _FailMsg&& failMsg)
		{
			std::unordered_map<_Ty1, _Ty2> ret;
			PyObject* key, * value;
			Py_ssize_t pos = 0;
			while (PyDict_Next(obj, &pos, &key, &value)) {
				ret.emplace(toCpp<_Ty1>(key), toCpp<_Ty2>(value));
			}
			if (PyErr_Occurred()) throw ConversionFail{ failMsg };
			return ret;
		}
	};

	namespace detail
	{
		template<typename _Ty>
		struct NpyType
		{
			enum {
				npy_type = -1,
			};
		};

		template<>
		struct NpyType<int8_t>
		{
			enum { 
				type = NPY_INT8,
				signed_type = type,
				npy_type = type,
			};
		};

		template<>
		struct NpyType<uint8_t>
		{
			enum { 
				type = NPY_UINT8,
				signed_type = NPY_INT8,
				npy_type = type,
			};
		};

		template<>
		struct NpyType<int16_t>
		{
			enum { 
				type = NPY_INT16,
				signed_type = type,
				npy_type = type,
			};
		};

		template<>
		struct NpyType<uint16_t>
		{
			enum { 
				type = NPY_UINT16,
				signed_type = NPY_INT16,
				npy_type = type,
			};
		};

		template<>
		struct NpyType<int32_t>
		{
			enum { 
				type = NPY_INT32,
				signed_type = type,
				npy_type = type,
			};
		};

		template<>
		struct NpyType<uint32_t>
		{
			enum { 
				type = NPY_UINT32,
				signed_type = NPY_INT32,
				npy_type = type,
			};
		};

		template<>
		struct NpyType<int64_t>
		{
			enum { 
				type = NPY_INT64,
				signed_type = type,
				npy_type = type,
			};
		};

		template<>
		struct NpyType<uint64_t>
		{
			enum { 
				type = NPY_UINT64, 
				signed_type = NPY_INT64,
				npy_type = type,
			};
		};

		template<>
		struct NpyType<float>
		{
			enum { 
				type = NPY_FLOAT, 
				signed_type = type,
				npy_type = type,
			};
		};

		template<>
		struct NpyType<double>
		{
			enum { 
				type = NPY_DOUBLE, 
				signed_type = type,
				npy_type = type,
			};
		};
	}

	struct cast_to_signed_t {};
	static constexpr cast_to_signed_t cast_to_signed{};

	struct force_list_t {};
	static constexpr force_list_t force_list{};

	template<typename _Ty>
	struct ValueBuilder<std::vector<_Ty>, 
		typename std::enable_if<std::is_arithmetic<_Ty>::value>::type>
	{
		PyObject* operator()(const std::vector<_Ty>& v)
		{
			npy_intp size = v.size();
			PyObject* obj = PyArray_EMPTY(1, &size, detail::NpyType<_Ty>::type, 0);
			std::memcpy(PyArray_DATA((PyArrayObject*)obj), v.data(), sizeof(_Ty) * size);
			return obj;
		}

		template<typename _FailMsg>
		std::vector<_Ty> _toCpp(PyObject* obj, _FailMsg&& failMsg)
		{
			if (detail::NpyType<_Ty>::npy_type >= 0 && PyArray_Check(obj) && PyArray_TYPE((PyArrayObject*)obj) == detail::NpyType<_Ty>::npy_type)
			{
				_Ty* ptr = (_Ty*)PyArray_GETPTR1((PyArrayObject*)obj, 0);
				return std::vector<_Ty>{ ptr, ptr + PyArray_Size(obj) };
			}
			else
			{
				UniqueObj iter{ PyObject_GetIter(obj) }, item;
				if (!iter) throw ConversionFail{ std::forward<_FailMsg>(failMsg) };
				std::vector<_Ty> v;
				while ((item = UniqueObj{ PyIter_Next(iter) }))
				{
					v.emplace_back(toCpp<_Ty>(item));
				}
				if (PyErr_Occurred())
				{
					throw ConversionFail{ std::forward<_FailMsg>(failMsg) };
				}
				return v;
			}
		}
	};

	template<typename _Ty>
	struct ValueBuilder<std::vector<_Ty>,
		typename std::enable_if<!std::is_arithmetic<_Ty>::value>::type>
	{
		PyObject* operator()(const std::vector<_Ty>& v)
		{
			PyObject* ret = PyList_New(v.size());
			size_t id = 0;
			for (auto& e : v)
			{
				PyList_SetItem(ret, id++, buildPyValue(e));
			}
			return ret;
		}

		template<typename _FailMsg>
		std::vector<_Ty> _toCpp(PyObject* obj, _FailMsg&& failMsg)
		{
			UniqueObj iter{ PyObject_GetIter(obj) }, item;
			if (!iter) throw ConversionFail{ std::forward<_FailMsg>(failMsg) };
			std::vector<_Ty> v;
			while ((item = UniqueObj{ PyIter_Next(iter) }))
			{
				v.emplace_back(toCpp<_Ty>(item));
			}
			if (PyErr_Occurred())
			{
				throw ConversionFail{ std::forward<_FailMsg>(failMsg) };
			}
			return v;
		}
	};

	template<typename T, typename Out, typename Msg>
	inline void transform(PyObject* iterable, Out out, Msg&& failMsg)
	{
		if (!iterable) throw ConversionFail{ std::forward<Msg>(failMsg) };
		UniqueObj iter{ PyObject_GetIter(iterable) }, item;
		if (!iter) throw ConversionFail{ std::forward<Msg>(failMsg) };
		while ((item = UniqueObj{ PyIter_Next(iter) }))
		{
			*out++ = toCpp<T>(item);
		}
		if (PyErr_Occurred())
		{
			throw ConversionFail{ std::forward<Msg>(failMsg) };
		}
	}

	template<typename T, typename Fn, typename Msg>
	inline void foreach(PyObject* iterable, Fn&& fn, Msg&& failMsg)
	{
		if (!iterable) throw ConversionFail{ std::forward<Msg>(failMsg) };
		UniqueObj iter{ PyObject_GetIter(iterable) }, item;
		if (!iter) throw ConversionFail{ std::forward<Msg>(failMsg) };
		while ((item = UniqueObj{ PyIter_Next(iter) }))
		{
			fn(toCpp<T>(item));
		}
		if (PyErr_Occurred())
		{
			throw ConversionFail{ std::forward<Msg>(failMsg) };
		}
	}

	template<typename T, typename Fn, typename Msg>
	inline void foreachWithPy(PyObject* iterable, Fn&& fn, Msg&& failMsg)
	{
		if (!iterable) throw ConversionFail{ std::forward<Msg>(failMsg) };
		UniqueObj iter{ PyObject_GetIter(iterable) }, item;
		if (!iter) throw ConversionFail{ std::forward<Msg>(failMsg) };
		while ((item = UniqueObj{ PyIter_Next(iter) }))
		{
			fn(toCpp<T>(item), item.get());
		}
		if (PyErr_Occurred())
		{
			throw ConversionFail{ std::forward<Msg>(failMsg) };
		}
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
	inline typename std::enable_if<
		!std::is_arithmetic<typename std::iterator_traits<_Ty>::value_type>::value,
		PyObject*
	>::type buildPyValue(_Ty first, _Ty last)
	{
		PyObject* ret = PyList_New(std::distance(first, last));
		size_t id = 0;
		for (; first != last; ++first)
		{
			PyList_SetItem(ret, id++, buildPyValue(*first));
		}
		return ret;
	}

	template<typename _Ty>
	inline PyObject* buildPyValue(const std::vector<_Ty>& v, force_list_t)
	{
		PyObject* ret = PyList_New(v.size());
		for (size_t i = 0; i < v.size(); ++i)
		{
			PyList_SetItem(ret, i, buildPyValue(v[i]));
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
		PyObject* ret = PyList_New(std::distance(first, last));
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
		using value_type = decltype(tx(*first));
		npy_intp size = std::distance(first, last);
		PyObject* ret = PyArray_EMPTY(1, &size, detail::NpyType<value_type>::type, 0);
		size_t id = 0;
		for (; first != last; ++first, ++id)
		{
			*(value_type*)PyArray_GETPTR1((PyArrayObject*)ret, id) = tx(*first);
		}
		return ret;
	}

	namespace detail
	{
		inline void setDictItem(PyObject* dict, const char** keys)
		{

		}

		template<typename _Ty, typename... _Rest>
		inline void setDictItem(PyObject* dict, const char** keys, _Ty&& value, _Rest&& ... rest)
		{
			{
				UniqueObj v{ buildPyValue(std::forward<_Ty>(value)) };
				PyDict_SetItemString(dict, keys[0], v);
			}
			return setDictItem(dict, keys + 1, std::forward<_Rest>(rest)...);
		}

		template<typename _Ty>
		inline bool isNull(_Ty v)
		{
			return false;
		}

		template<>
		inline bool isNull<PyObject*>(PyObject* v)
		{
			return !v;
		}

		inline void setDictItemSkipNull(PyObject* dict, const char** keys)
		{

		}

		template<typename _Ty, typename... _Rest>
		inline void setDictItemSkipNull(PyObject* dict, const char** keys, _Ty&& value, _Rest&& ... rest)
		{
			if(!isNull(value))
			{
				UniqueObj v{ buildPyValue(value) };
				PyDict_SetItemString(dict, keys[0], v);
			}
			return setDictItemSkipNull(dict, keys + 1, std::forward<_Rest>(rest)...);
		}

		template<size_t _n>
		inline void setTupleItem(PyObject* tuple)
		{
		}

		template<size_t _n, typename _Ty, typename... _Rest>
		inline void setTupleItem(PyObject* tuple, _Ty&& first, _Rest&&... rest)
		{
			PyTuple_SET_ITEM(tuple, _n, buildPyValue(std::forward<_Ty>(first)));
			return setTupleItem<_n + 1>(tuple, std::forward<_Rest>(rest)...);
		}
	}

	template<typename... _Rest>
	inline PyObject* buildPyDict(const char** keys, _Rest&&... rest)
	{
		PyObject* dict = PyDict_New();
		detail::setDictItem(dict, keys, std::forward<_Rest>(rest)...);
		return dict;
	}

	template<typename... _Rest>
	inline PyObject* buildPyDictSkipNull(const char** keys, _Rest&&... rest)
	{
		PyObject* dict = PyDict_New();
		detail::setDictItemSkipNull(dict, keys, std::forward<_Rest>(rest)...);
		return dict;
	}

	template<typename _Ty>
	inline void setPyDictItem(PyObject* dict, const char* key, _Ty&& value)
	{
		UniqueObj v{ buildPyValue(value) };
		PyDict_SetItemString(dict, key, v);
	}

	template<typename... _Rest>
	inline PyObject* buildPyTuple(_Rest&&... rest)
	{
		PyObject* tuple = PyTuple_New(sizeof...(_Rest));
		detail::setTupleItem<0>(tuple, std::forward<_Rest>(rest)...);
		return tuple;
	}
}
