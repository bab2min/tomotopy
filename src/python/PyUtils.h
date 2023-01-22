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
#include <deque>
#include <future>

#ifdef _DEBUG
	#undef _DEBUG
	#include <Python.h>
	#define _DEBUG
#else
	#include <Python.h>
#endif

#include <frameobject.h>

#define USE_NUMPY

#ifdef MAIN_MODULE
#else
	#define NO_IMPORT_ARRAY
#endif

#ifdef USE_NUMPY
	#define PY_ARRAY_UNIQUE_SYMBOL TOMOTOPY_ARRAY_API
	#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
	#include <numpy/arrayobject.h>
#endif

namespace py
{
	template<class Ty = PyObject>
	struct UniqueCObj
	{
		Ty* obj = nullptr;
		explicit UniqueCObj(Ty* _obj = nullptr) : obj(_obj) {}
		~UniqueCObj()
		{
			Py_XDECREF(obj);
		}

		UniqueCObj(const UniqueCObj&) = delete;
		UniqueCObj& operator=(const UniqueCObj&) = delete;

		UniqueCObj(UniqueCObj&& o) noexcept
		{
			std::swap(obj, o.obj);
		}

		UniqueCObj& operator=(UniqueCObj&& o) noexcept
		{
			std::swap(obj, o.obj);
			return *this;
		}

		Ty* get() const
		{
			return obj;
		}

		Ty* release()
		{
			auto o = obj;
			obj = nullptr;
			return o;
		}

		operator bool() const
		{
			return !!obj;
		}

		operator Ty* () const
		{
			return obj;
		}

		Ty* operator->()
		{
			return obj;
		}

		const Ty* operator->() const
		{
			return obj;
		}
	};

	template<class Ty = PyObject>
	struct SharedCObj
	{
		Ty* obj = nullptr;
		explicit SharedCObj(Ty* _obj = nullptr) : obj(_obj) {}
		~SharedCObj()
		{
			Py_XDECREF(obj);
		}

		SharedCObj(const SharedCObj& o)
			: obj(o.obj)
		{
			Py_INCREF(obj);
		}
		SharedCObj& operator=(const SharedCObj& o)
		{
			Py_XDECREF(obj);
			obj = o.obj;
			Py_INCREF(obj);
			return *this;
		}

		SharedCObj(SharedCObj&& o) noexcept
		{
			std::swap(obj, o.obj);
		}

		SharedCObj& operator=(SharedCObj&& o) noexcept
		{
			std::swap(obj, o.obj);
			return *this;
		}

		Ty* get() const
		{
			return obj;
		}

		operator bool() const
		{
			return !!obj;
		}

		operator Ty* () const
		{
			return obj;
		}

		Ty* operator->()
		{
			return obj;
		}

		const Ty* operator->() const
		{
			return obj;
		}
	};

	using UniqueObj = UniqueCObj<>;
	using SharedObj = SharedCObj<>;

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
		-> typename std::enable_if<std::is_pointer<decltype(fn())>::value, decltype(fn())>::type;

	template<typename _Fn>
	auto handleExc(_Fn&& fn)
		-> typename std::enable_if<std::is_integral<decltype(fn())>::value, decltype(fn())>::type;

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
		return ValueBuilder<_Ty>{}._toCpp(obj, [=]() { return "cannot convert " + repr(obj) + " into appropriate C++ type"; });
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
	struct ValueBuilder<std::u16string>
	{
		PyObject* operator()(const std::u16string& v)
		{
			return PyUnicode_DecodeUTF16((const char*)v.data(), v.size() * 2, nullptr, nullptr);
		}

		template<typename _FailMsg>
		std::u16string _toCpp(PyObject* obj, _FailMsg&& msg)
		{
			UniqueObj uobj{ PyUnicode_FromObject(obj) };
			if (!uobj) throw ConversionFail{ std::forward<_FailMsg>(msg) };
			size_t len = PyUnicode_GET_LENGTH(uobj.get());
			std::u16string ret;
			switch (PyUnicode_KIND(uobj.get()))
			{
			case PyUnicode_1BYTE_KIND:
			{
				auto* p = PyUnicode_1BYTE_DATA(uobj.get());
				ret.resize(len);
				std::copy(p, p + len, &ret[0]);
				break;
			}
			case PyUnicode_2BYTE_KIND:
			{
				auto* p = PyUnicode_2BYTE_DATA(uobj.get());
				ret.resize(len);
				std::copy(p, p + len, &ret[0]);
				break;
			}
			case PyUnicode_4BYTE_KIND:
			{
				auto* p = PyUnicode_4BYTE_DATA(uobj.get());
				for (size_t i = 0; i < len; ++i)
				{
					auto c = p[i];
					ret.reserve(len);
					if (c < 0x10000)
					{
						ret.push_back(c);
					}
					else
					{
						ret.push_back(0xD800 - (0x10000 >> 10) + (c >> 10));
						ret.push_back(0xDC00 + (c & 0x3FF));
					}
				}
				break;
			}
			default:
				throw ConversionFail{ std::forward<_FailMsg>(msg) };
			}
			return ret;
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

	template<size_t len>
	struct ValueBuilder<char[len]>
	{
		PyObject* operator()(const char(&v)[len])
		{
			return PyUnicode_FromStringAndSize(v, len - 1);
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

	template<typename Ty>
	struct ValueBuilder<UniqueCObj<Ty>>
	{
		PyObject* operator()(UniqueCObj<Ty>&& v)
		{
			if (v)
			{
				Py_INCREF(v);
				return (PyObject*)v.get();
			}
			else
			{
				Py_INCREF(Py_None);
				return Py_None;
			}
		}

		PyObject* operator()(const UniqueCObj<Ty>& v)
		{
			if (v)
			{
				Py_INCREF(v);
				return (PyObject*)v.get();
			}
			else
			{
				Py_INCREF(Py_None);
				return Py_None;
			}
		}
	};

	template<typename Ty>
	struct ValueBuilder<SharedCObj<Ty>>
	{
		PyObject* operator()(SharedCObj<Ty>&& v)
		{
			if (v)
			{
				Py_INCREF(v);
				return (PyObject*)v.get();
			}
			else
			{
				Py_INCREF(Py_None);
				return Py_None;
			}
		}

		PyObject* operator()(const SharedCObj<Ty>& v)
		{
			if (v)
			{
				Py_INCREF(v);
				return (PyObject*)v.get();
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
				if (PyDict_SetItem(ret, key, val)) return nullptr;
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

#ifdef USE_NUMPY
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

	template<typename _Ty>
	struct numpy_able : std::integral_constant<bool, std::is_arithmetic<_Ty>::value> {};

#else
	template<typename _Ty>
	struct numpy_able : std::false_type {};
#endif
	struct force_list_t {};
	static constexpr force_list_t force_list{};

#ifdef USE_NUMPY
	template<typename _Ty>
	struct ValueBuilder<std::vector<_Ty>,
		typename std::enable_if<numpy_able<_Ty>::value>::type>
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
#endif

	template<typename _Ty>
	struct ValueBuilder<std::vector<_Ty>,
		typename std::enable_if<!numpy_able<_Ty>::value>::type>
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

#ifdef USE_NUMPY
	template<typename _Ty>
	inline typename std::enable_if<numpy_able<_Ty>::value, PyObject*>::type
		buildPyValue(const std::vector<_Ty>& v, cast_to_signed_t)
	{
		npy_intp size = v.size();
		PyObject* obj = PyArray_EMPTY(1, &size, detail::NpyType<_Ty>::signed_type, 0);
		std::memcpy(PyArray_DATA((PyArrayObject*)obj), v.data(), sizeof(_Ty) * size);
		return obj;
	}
#endif

	template<typename _Ty>
	inline typename std::enable_if<
		!numpy_able<typename std::iterator_traits<_Ty>::value_type>::value,
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
		!numpy_able<
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

#ifdef USE_NUMPY
	template<typename _Ty>
	inline typename std::enable_if<
		numpy_able<typename std::iterator_traits<_Ty>::value_type>::value,
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
		numpy_able<
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
#endif

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
			if (!isNull(value))
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

	template<class Derived>
	struct CObject
	{
		PyObject_HEAD;

		static PyObject* _new(PyTypeObject* subtype, PyObject* args, PyObject* kwargs)
		{
			return handleExc([&]()
			{
				py::UniqueObj ret{ subtype->tp_alloc(subtype, 0) };
				new ((Derived*)ret.get()) Derived;
				return ret.release();
			});
		}

		static void dealloc(Derived* self)
		{
			self->~Derived();
			Py_TYPE(self)->tp_free((PyObject*)self);
		}
	};

	template<class Derived, class RetTy>
	struct ResultIter : public CObject<Derived>
	{
		UniqueObj inputIter;
		std::deque<std::future<RetTy>> futures;
		std::deque<SharedObj> inputItems;
		bool echo = false;

		~ResultIter()
		{
			for (auto& p : futures)
			{
				p.get();
			}
		}

		static int init(Derived* self, PyObject*, PyObject*)
		{
			return 0;
		}

		static Derived* iter(Derived* self)
		{
			Py_INCREF(self);
			return self;
		}

		static PyObject* iternext(Derived* self)
		{
			return handleExc([&]() -> PyObject* {
				if (!self->feed() && self->futures.empty()) return nullptr;
				auto f = std::move(self->futures.front());
				self->futures.pop_front();
				if (self->echo)
				{
					auto input = std::move(self->inputItems.front());
					self->inputItems.pop_front();
					return buildPyTuple(UniqueObj{ self->buildPy(f.get()) }, input);
				}
				else
				{
					return self->buildPy(f.get());
				}
			});
		}

		bool feed()
		{
			SharedObj item{ PyIter_Next(inputIter) };
			if (!item)
			{
				if (PyErr_Occurred()) throw ExcPropagation{};
				return false;
			}
			if (echo) inputItems.emplace_back(item);
			futures.emplace_back(static_cast<Derived*>(this)->feedNext(std::move(item)));
			return true;
		}

		std::future<RetTy> feedNext(py::SharedObj&& next)
		{
			return {};
		}

		PyObject* buildPy(RetTy&& v)
		{
			return py::buildPyValue(std::move(v));
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
			if (PyErr_Occurred())
			{
				PyObject* exc, * val, * tb, * val2;
				PyErr_Fetch(&exc, &val, &tb);
				PyErr_NormalizeException(&exc, &val, &tb);
				if (tb)
				{
					PyException_SetTraceback(val, tb);
					Py_DECREF(tb);
				}
				Py_DECREF(exc);
				PyObject* et = e.pytype();
				val2 = PyObject_CallFunctionObjArgs(et, py::UniqueObj{ buildPyValue(e.what()) }.get(), nullptr);
				Py_INCREF(val);
				PyException_SetCause(val2, val);
				PyException_SetContext(val2, val);
				PyErr_SetObject(et, val2);
				Py_DECREF(val2);
			}
			else
			{
				PyErr_SetString(e.pytype(), e.what());
			}
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
			if (PyErr_Occurred())
			{
				PyObject* exc, * val, * tb, * val2;
				PyErr_Fetch(&exc, &val, &tb);
				PyErr_NormalizeException(&exc, &val, &tb);
				if (tb)
				{
					PyException_SetTraceback(val, tb);
					Py_DECREF(tb);
				}
				Py_DECREF(exc);
				PyObject* et = e.pytype();
				val2 = PyObject_CallFunctionObjArgs(et, py::UniqueObj{ buildPyValue(e.what()) }.get(), nullptr);
				Py_INCREF(val);
				PyException_SetCause(val2, val);
				PyException_SetContext(val2, val);
				PyErr_SetObject(et, val2);
				Py_DECREF(val2);
			}
			else
			{
				PyErr_SetString(e.pytype(), e.what());
			}
		}
		catch (const std::exception& e)
		{
			std::cerr << "Uncaughted c++ exception: " << e.what() << std::endl;
			PyErr_SetString(PyExc_RuntimeError, e.what());
		}
		return -1;
	}
}
