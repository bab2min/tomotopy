#pragma once
#include <array>
#include <type_traits>
#include <vector>
#include <map>
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
#include <optional>
#include <variant>
#include <numeric>
#include <typeinfo>
#include <typeindex>

#ifdef _DEBUG
#undef _DEBUG
#include <Python.h>
#define _DEBUG
#else
#include <Python.h>
#endif

#include <frameobject.h>
#include <structmember.h>

#if defined(__clang__)
#define PY_STRONG_INLINE inline
#elif defined(__GNUC__) || defined(__GNUG__)
#define PY_STRONG_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#define PY_STRONG_INLINE __forceinline 
#endif

namespace py
{
	template<class Ty>
	struct CObject;

	template<class Ty>
	struct PObject;

	template<class Ty = PyObject>
	struct UniqueCObj
	{
		Ty* obj = nullptr;

		UniqueCObj() {}

		explicit UniqueCObj(Ty* _obj) : obj(_obj) 
		{
			static_assert(std::is_same_v<PyObject, Ty> || std::is_base_of_v<CObject<Ty>, Ty>, "UniqueCObj can only be used with PyObject, CObject or PObject");
		}

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

		void incref() const
		{
			Py_INCREF(obj);
		}

		void copyFrom(Ty* o)
		{
			~UniqueCObj();
			obj = o;
			if (obj) incref();
		}

		UniqueCObj copy() const
		{
			if (obj) incref();
			return UniqueCObj{ obj };
		}

		operator UniqueCObj<PyObject>()
		{
			return UniqueCObj<PyObject>{ (PyObject*)release() };
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

		explicit operator Ty* () const
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

	template<class Ty>
	struct UniqueCObj<PObject<Ty>>
	{
		PObject<Ty>* obj = nullptr;

		UniqueCObj() {}

		explicit UniqueCObj(PObject<Ty>* _obj) : obj(_obj)
		{
		}

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

		void incref() const
		{
			Py_INCREF(obj);
		}

		operator UniqueCObj<PyObject>()
		{
			return UniqueCObj<PyObject>{ (PyObject*)release() };
		}

		PObject<Ty>* get() const
		{
			return obj;
		}

		PObject<Ty>* release()
		{
			auto o = obj;
			obj = nullptr;
			return o;
		}

		operator bool() const
		{
			return !!obj;
		}

		explicit operator PObject<Ty>* () const
		{
			return obj;
		}

		Ty* operator->()
		{
			return &obj->value;
		}

		const Ty* operator->() const
		{
			return &obj->value;
		}
	};

	template<class Ty>
	using UniquePObj = UniqueCObj<PObject<Ty>>;

	template<class Ty = PyObject>
	struct SharedCObj
	{
		Ty* obj = nullptr;

		SharedCObj() {}

		SharedCObj(Ty* _obj) : obj(_obj) 
		{
			static_assert(std::is_same_v<PyObject, Ty> || std::is_base_of_v<CObject<Ty>, Ty>, "UniqueCObj can only be used with PyObject or CObject");
		}

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

		void incref() const
		{
			Py_INCREF(obj);
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

	template<class Ty>
	struct StringWithOffset
	{
		Ty str;
		std::vector<size_t> offsets;
	};

	class ForeachFailed : public std::runtime_error
	{
	public:
		ForeachFailed() : std::runtime_error{ "" }
		{
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
	inline UniqueObj buildPyValue(_Ty&& v)
	{
		return ValueBuilder<
			typename std::remove_const<typename std::remove_reference<_Ty>::type>::type
		>{}(std::forward<_Ty>(v));
	}

	template<typename _Ty, typename _FailMsg>
	inline _Ty toCppWithException(PyObject* obj, _FailMsg&& fail)
	{
		_Ty ret;
		if (!obj || !ValueBuilder<_Ty>{}._toCpp(obj, ret)) throw ConversionFail{ std::forward<_FailMsg>(fail) };
		return ret;
	}

	template<typename _Ty>
	inline _Ty getAttr(PyObject* obj, const char* attr)
	{
		py::UniqueObj item{ PyObject_GetAttrString(obj, attr) };
		return toCppWithException<_Ty>(item.get(), [&]() { return std::string{ "Failed to get attribute " } + attr; });
	}

	inline std::string repr(PyObject* o)
	{
		UniqueObj r{ PyObject_Repr(o) };
		if (!r) throw ExcPropagation{};
		return toCppWithException<std::string>(r.get(), "");
	}

	inline std::string reprWithNestedError(PyObject* o)
	{
		PyObject* type, * value, * traceback;
		PyErr_Fetch(&type, &value, &traceback);
		PyErr_Clear();
		UniqueObj r{ PyObject_Repr(o) };
		if (!r) throw ExcPropagation{};
		PyErr_Restore(type, value, traceback);
		return toCppWithException<std::string>(r.get(), "");
	}

	template<typename _Ty>
	inline std::string reprFromCpp(_Ty&& o)
	{
		UniqueObj p{ py::buildPyValue(std::forward<_Ty>(o)) };
		UniqueObj r{ PyObject_Repr(p.get()) };
		if (!r) throw ExcPropagation{};
		return toCppWithException<std::string>(r.get(), "");
	}

	template<typename _Ty>
	inline _Ty toCpp(PyObject* obj)
	{
		if (!obj) throw ConversionFail{ "cannot convert null pointer into appropriate C++ type" };
		_Ty v;
		if (!ValueBuilder<_Ty>{}._toCpp(obj, v)) throw ConversionFail{ "cannot convert " + reprWithNestedError(obj) + " into appropriate C++ type" };
		return v;
	}

	template<typename _Ty>
	inline bool toCpp(PyObject* obj, _Ty& out)
	{
		if (!obj) return false;
		return ValueBuilder<_Ty>{}._toCpp(obj, out);
	}

	inline void clearError()
	{
		PyErr_Clear();
	}

	template<typename... _Rest>
	inline UniqueObj buildPyTuple(_Rest&&... rest);

	template<typename _Ty>
	struct ValueBuilder<_Ty,
		typename std::enable_if<std::is_integral<_Ty>::value || std::is_enum<_Ty>::value>::type>
	{
		UniqueObj operator()(_Ty v)
		{
			return UniqueObj{ PyLong_FromLongLong((long long)v) };
		}

		bool _toCpp(PyObject* obj, _Ty& out)
		{
			long long v = PyLong_AsLongLong(obj);
			if (v == -1 && PyErr_Occurred()) return false;
			out = (_Ty)v;
			return true;
		}
	};

	template<typename _Ty>
	struct ValueBuilder<_Ty,
		typename std::enable_if<std::is_floating_point<_Ty>::value>::type>
	{
		UniqueObj operator()(_Ty v)
		{
			return UniqueObj{ PyFloat_FromDouble(v) };
		}

		bool _toCpp(PyObject* obj, _Ty& out)
		{
			double v = PyFloat_AsDouble(obj);
			if (v == -1 && PyErr_Occurred()) return false;
			out = (_Ty)v;
			return true;
		}
	};

	template<>
	struct ValueBuilder<std::string>
	{
		UniqueObj operator()(const std::string& v)
		{
			return UniqueObj{ PyUnicode_FromStringAndSize(v.data(), v.size()) };
		}

		bool _toCpp(PyObject* obj, std::string& out)
		{
			UniqueObj encoded{ PyUnicode_AsEncodedString(obj, "utf-8", "strict") };
			if (!encoded) return false;
			char* buffer;
			Py_ssize_t size;
			if (PyBytes_AsStringAndSize(encoded.get(), &buffer, &size)) return false;
			out = { buffer, buffer + size };
			return true;
		}
	};

	template<>
	struct ValueBuilder<std::string_view>
	{
		UniqueObj operator()(const std::string_view& v)
		{
			return UniqueObj{ PyUnicode_FromStringAndSize(v.data(), v.size()) };
		}
	};

	template<>
	struct ValueBuilder<std::u16string>
	{
		UniqueObj operator()(const std::u16string& v)
		{
			return UniqueObj{ PyUnicode_DecodeUTF16((const char*)v.data(), v.size() * 2, nullptr, nullptr) };
		}

		bool _toCpp(PyObject* obj, std::u16string& out)
		{
			UniqueObj uobj{ PyUnicode_FromObject(obj) };
			if (!uobj) return false;
			const size_t len = PyUnicode_GetLength(uobj.get());
			out.reserve(len);
			auto buf = std::make_unique<Py_UCS4[]>(len);
			if (!PyUnicode_AsUCS4(uobj.get(), buf.get(), len, 0)) return false;

			for (size_t i = 0; i < len; ++i)
			{
				auto c = buf[i];
				if (c < 0x10000)
				{
					out.push_back(c);
				}
				else
				{
					out.push_back(0xD800 - (0x10000 >> 10) + (c >> 10));
					out.push_back(0xDC00 + (c & 0x3FF));
				}
			}
			return true;
		}
	};


	template<>
	struct ValueBuilder<std::u16string_view>
	{
		UniqueObj operator()(const std::u16string_view& v)
		{
			return UniqueObj{ PyUnicode_DecodeUTF16((const char*)v.data(), v.size() * 2, nullptr, nullptr) };
		}
	};

	template<>
	struct ValueBuilder<StringWithOffset<std::u16string>>
	{
		bool _toCpp(PyObject* obj, StringWithOffset<std::u16string>& out)
		{
			UniqueObj uobj{ PyUnicode_FromObject(obj) };
			if (!uobj) return false;
			const size_t len = PyUnicode_GetLength(uobj.get());
			auto buf = std::make_unique<Py_UCS4[]>(len);
			if (!PyUnicode_AsUCS4(uobj.get(), buf.get(), len, 0)) return false;

			out.str.reserve(len);
			out.offsets.reserve(len);
			for (size_t i = 0; i < len; ++i)
			{
				auto c = buf[i];
				if (c < 0x10000)
				{
					out.offsets.emplace_back(out.str.size());
					out.str.push_back(c);
				}
				else
				{
					out.offsets.emplace_back(out.str.size());
					out.str.push_back(0xD800 - (0x10000 >> 10) + (c >> 10));
					out.str.push_back(0xDC00 + (c & 0x3FF));
				}
			}
			out.offsets.emplace_back(out.str.size());
			return true;
		}
	};

	template<>
	struct ValueBuilder<const char*>
	{
		UniqueObj operator()(const char* v)
		{
			return UniqueObj{ PyUnicode_FromString(v) };
		}
	};

	template<size_t len>
	struct ValueBuilder<char[len]>
	{
		UniqueObj operator()(const char(&v)[len])
		{
			return UniqueObj{ PyUnicode_FromStringAndSize(v, len - 1) };
		}
	};

	template<>
	struct ValueBuilder<bool>
	{
		UniqueObj operator()(bool v)
		{
			return UniqueObj{ PyBool_FromLong(v) };
		}

		bool _toCpp(PyObject* obj, bool& out)
		{
			if (!obj) return false;
			out = !!PyObject_IsTrue(obj);
			return true;
		}
	};

	template<>
	struct ValueBuilder<std::nullptr_t>
	{
		UniqueObj operator()(std::nullptr_t)
		{
			Py_INCREF(Py_None);
			return UniqueObj{ Py_None };
		}
	};

	template<>
	struct ValueBuilder<PyObject*>
	{
		UniqueObj operator()(PyObject* v)
		{
			if (!v) v = Py_None;
			Py_INCREF(v);
			return UniqueObj{ v };
		}

		bool _toCpp(PyObject* obj, PyObject*& out)
		{
			out = obj;
			return true;
		}
	};

	template<typename Ty>
	struct ValueBuilder<UniqueCObj<Ty>>
	{
		UniqueObj operator()(UniqueCObj<Ty>&& v)
		{
			if (v)
			{
				Py_INCREF(v.get());
				return UniqueObj{ (PyObject*)v.get() };
			}
			else
			{
				Py_INCREF(Py_None);
				return UniqueObj{ Py_None };
			}
		}

		UniqueObj operator()(const UniqueCObj<Ty>& v)
		{
			if (v)
			{
				Py_INCREF(v.get());
				return UniqueObj{ (PyObject*)v.get() };
			}
			else
			{
				Py_INCREF(Py_None);
				return UniqueObj{ Py_None };
			}
		}

		bool _toCpp(PyObject* obj, UniqueCObj<Ty>& out);
	};

	template<typename Ty>
	struct ValueBuilder<SharedCObj<Ty>>
	{
		UniqueObj operator()(SharedCObj<Ty>&& v)
		{
			if (v)
			{
				Py_INCREF(v);
				return UniqueObj{ (PyObject*)v.get() };
			}
			else
			{
				Py_INCREF(Py_None);
				return UniqueObj{ Py_None };
			}
		}

		UniqueObj operator()(const SharedCObj<Ty>& v)
		{
			if (v)
			{
				Py_INCREF(v);
				return UniqueObj{ (PyObject*)v.get() };
			}
			else
			{
				Py_INCREF(Py_None);
				return UniqueObj{ Py_None };
			}
		}
	};

	template<class Ty>
	struct ValueBuilder<PObject<Ty>*>
	{
		UniqueObj operator()(PObject<Ty>* v)
		{
			if (!v) v = Py_None;
			Py_INCREF(v);
			return UniqueObj{ v };
		}

		bool _toCpp(PyObject* obj, PObject<Ty>*& out)
		{
			out = (PObject<Ty>*)obj;
			return true;
		}
	};

	template<typename _Ty1, typename _Ty2>
	struct ValueBuilder<std::pair<_Ty1, _Ty2>>
	{
		UniqueObj operator()(const std::pair<_Ty1, _Ty2>& v)
		{
			UniqueObj ret{ PyTuple_New(2) };
			size_t id = 0;
			PyTuple_SetItem(ret.get(), id++, buildPyValue(std::get<0>(v)).release());
			PyTuple_SetItem(ret.get(), id++, buildPyValue(std::get<1>(v)).release());
			return ret;
		}

		bool _toCpp(PyObject* obj, std::pair<_Ty1, _Ty2>& out)
		{
			if (Py_SIZE(obj) != 2) throw ConversionFail{ "input is not tuple with len=2: " + reprWithNestedError(obj) };
			if (!toCpp<_Ty1>(UniqueObj{ PySequence_GetItem(obj, 0) }.get(), out.first)) return false;
			if (!toCpp<_Ty2>(UniqueObj{ PySequence_GetItem(obj, 1) }.get(), out.second)) return false;
			return true;
		}
	};

	template<typename... _Tys>
	struct ValueBuilder<std::tuple<_Tys...>>
	{
	private:
		void setValue(PyObject* o, const std::tuple<_Tys...>& v, std::integer_sequence<size_t>)
		{
		}

		template<size_t i, size_t ...rest>
		void setValue(PyObject* o, const std::tuple<_Tys...>& v, std::integer_sequence<size_t, i, rest...>)
		{
			PyTuple_SetItem(o, i, buildPyValue(std::get<i>(v)).release());
			return setValue(o, v, std::integer_sequence<size_t, rest...>{});
		}

		template<size_t n, size_t ...idx>
		bool getValue(PyObject* o, std::tuple<_Tys...>& out, std::integer_sequence<size_t, n, idx...>)
		{
			if (!toCpp<typename std::tuple_element<n, std::tuple<_Tys...>>::type>(UniqueObj{ PySequence_GetItem(o, n) }.get(), std::get<n>(out))) return false;
			return getValue(o, out, std::integer_sequence<size_t, idx...>{});
		}

		bool getValue(PyObject* o, std::tuple<_Tys...>& out, std::integer_sequence<size_t>)
		{
			return true;
		}

	public:
		UniqueObj operator()(const std::tuple<_Tys...>& v)
		{
			UniqueObj ret{ PyTuple_New(sizeof...(_Tys)) };
			size_t id = 0;
			setValue(ret.get(), v, std::make_index_sequence<sizeof...(_Tys)>{});
			return ret;
		}

		bool _toCpp(PyObject* obj, std::tuple<_Tys...>& out)
		{
			if (Py_SIZE(obj) != sizeof...(_Tys)) return false;
			getValue(obj, out, std::make_index_sequence<sizeof...(_Tys)>{});
			return true;
		}
	};

	template<typename _Ty1, typename _Ty2>
	struct ValueBuilder<std::unordered_map<_Ty1, _Ty2>>
	{
		UniqueObj operator()(const std::unordered_map<_Ty1, _Ty2>& v)
		{
			UniqueObj ret{ PyDict_New() };
			for (auto& p : v)
			{
				if (PyDict_SetItem(ret.get(), buildPyValue(p.first).get(), buildPyValue(p.second).get())) return UniqueObj{ nullptr };
			}
			return ret;
		}

		bool _toCpp(PyObject* obj, std::unordered_map<_Ty1, _Ty2>& out)
		{
#ifdef Py_GIL_DISABLED
			Py_BEGIN_CRITICAL_SECTION(obj);
#endif
			PyObject* key, * value;
			Py_ssize_t pos = 0;
			while (PyDict_Next(obj, &pos, &key, &value))
			{
				_Ty1 k;
				_Ty2 v;
				if (!toCpp<_Ty1>(key, k)) return false;
				if (!toCpp<_Ty2>(value, v)) return false;
				out.emplace(std::move(k), std::move(v));
			}
#ifdef Py_GIL_DISABLED
			Py_END_CRITICAL_SECTION();
#endif
			if (PyErr_Occurred()) return false;
			return true;
		}
	};

	template<typename _Ty>
	struct ValueBuilder<std::optional<_Ty>>
	{
		UniqueObj operator()(const std::optional<_Ty>& v)
		{
			if (v) return buildPyValue(*v);
			return buildPyValue(nullptr);
		}

		bool _toCpp(PyObject* obj, std::optional<_Ty>& out)
		{
			if (obj != Py_None)
			{
				_Ty v;
				if (!toCpp<_Ty>(obj, v)) return false;
				out = std::move(v);
				return true;
			}
			out = {};
			return true;
		}
	};

	template<typename Ty, typename... Ts>
	struct ValueBuilder<std::variant<Ty, Ts...>>
	{
		UniqueObj operator()(const std::variant<Ty, Ts...>& v)
		{
			return std::visit([](auto&& t)
			{
				return py::buildPyValue(std::forward<decltype(t)>(t));
			}, v);
		}

		bool _toCpp(PyObject* obj, std::variant<Ty, Ts...>& out)
		{
			Ty v;
			if (toCpp<Ty>(obj, v))
			{
				out = std::move(v);
				return true;
			}

			if constexpr (sizeof...(Ts) > 0)
			{
				std::variant<Ts...> v2;
				if (toCpp<std::variant<Ts...>>(obj, v2))
				{
					out = std::visit([](auto&& t) -> std::variant<Ty, Ts...>
					{
						return std::forward<decltype(t)>(t);
					}, std::move(v2));
					return true;
				}
			}
			else
			{
			}
			return false;
		}
	};


#ifdef USE_NUMPY
	enum NPY_TYPES {
		NPY_BOOL = 0,
		NPY_INT8, NPY_UINT8,
		NPY_INT16, NPY_UINT16,
		NPY_INT32, NPY_UINT32,
		NPY_LONG, NPY_ULONG,
		NPY_INT64, NPY_UINT64,
		NPY_FLOAT, NPY_DOUBLE, NPY_LONGDOUBLE,
		NPY_CFLOAT, NPY_CDOUBLE, NPY_CLONGDOUBLE,
		NPY_OBJECT = 17,
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
			static constexpr const char* dtype = "int8";
		};

		template<>
		struct NpyType<uint8_t>
		{
			enum {
				type = NPY_UINT8,
				signed_type = NPY_INT8,
				npy_type = type,
			};
			static constexpr const char* dtype = "uint8";
		};

		template<>
		struct NpyType<int16_t>
		{
			enum {
				type = NPY_INT16,
				signed_type = type,
				npy_type = type,
			};
			static constexpr const char* dtype = "int16";
		};

		template<>
		struct NpyType<uint16_t>
		{
			enum {
				type = NPY_UINT16,
				signed_type = NPY_INT16,
				npy_type = type,
			};
			static constexpr const char* dtype = "uint16";
		};

		template<>
		struct NpyType<int32_t>
		{
			enum {
				type = NPY_INT32,
				signed_type = type,
				npy_type = type,
			};
			static constexpr const char* dtype = "int32";
		};

		template<>
		struct NpyType<uint32_t>
		{
			enum {
				type = NPY_UINT32,
				signed_type = NPY_INT32,
				npy_type = type,
			};
			static constexpr const char* dtype = "uint32";
		};

		template<>
		struct NpyType<int64_t>
		{
			enum {
				type = NPY_INT64,
				signed_type = type,
				npy_type = type,
			};
			static constexpr const char* dtype = "int64";
		};

		template<>
		struct NpyType<uint64_t>
		{
			enum {
				type = NPY_UINT64,
				signed_type = NPY_INT64,
				npy_type = type,
			};
			static constexpr const char* dtype = "uint64";
		};

#ifdef __APPLE__
		template<>
		struct NpyType<long> : public NpyType<int64_t>
		{
		};

		template<>
		struct NpyType<unsigned long> : public NpyType<uint64_t>
		{
		};
#endif

		template<>
		struct NpyType<float>
		{
			enum {
				type = NPY_FLOAT,
				signed_type = type,
				npy_type = type,
			};
			static constexpr const char* dtype = "float32";
		};

		template<>
		struct NpyType<double>
		{
			enum {
				type = NPY_DOUBLE,
				signed_type = type,
				npy_type = type,
			};
			static constexpr const char* dtype = "float64";
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

	template<class _Ty>
	struct numpy_pair_test : std::false_type {};

	template<class _Ty>
	struct numpy_pair_test<std::pair<_Ty, _Ty>> : numpy_able<_Ty> {};

	struct force_list_t {};
	static constexpr force_list_t force_list{};

	inline UniqueObj importFrom(const char* moduleName, const char* targetName)
	{
		UniqueObj module{ PyImport_ImportModule(moduleName) };
		if (!module) throw ExcPropagation{};
		UniqueObj target{ PyObject_GetAttrString(module.get(), targetName) };
		if (!target) throw ExcPropagation{};
		return target;
	}

#ifdef USE_NUMPY

	inline void* getArrayDataPtr(PyObject* array)
	{
		UniqueObj ctypes{ PyObject_GetAttrString(array, "ctypes") };
		if (!ctypes) throw ExcPropagation{};
		UniqueObj data{ PyObject_GetAttrString(ctypes.get(), "data") };
		if (!data) throw ExcPropagation{};
		void* ptr = PyLong_AsVoidPtr(data.get());
		if (ptr == nullptr && PyErr_Occurred()) throw ExcPropagation{};
		return ptr;
	}

	template<class DType, class... Int>
	inline UniqueObj newEmptyArray(DType*& dataPtrOut, Int... shape)
	{
		static_assert((std::is_integral_v<Int> && ...), "shape parameters must be integral types");
		UniqueObj npempty = importFrom("numpy", "empty");
		UniqueObj array{ PyObject_CallFunctionObjArgs(
			npempty.get(), buildPyTuple(shape...).get(), buildPyValue(detail::NpyType<DType>::dtype).get(), nullptr)
		};
		if (!array) throw ExcPropagation{};
		dataPtrOut = (DType*)getArrayDataPtr(array.get());
		return array;
	}


	template<class DType, class Int>
	inline UniqueObj newEmptyArrayFromDim(DType*& dataPtrOut, size_t dim, const Int* shape)
	{
		static_assert(std::is_integral_v<Int>, "shape parameters must be integral types");
		UniqueObj npempty = importFrom("numpy", "empty");
		py::UniqueObj shapeObj{ PyTuple_New(dim) };
		for (size_t i = 0; i < dim; ++i)
		{
			PyTuple_SetItem(shapeObj.get(), i, buildPyValue(shape[i]).release());
		}
		UniqueObj array{ PyObject_CallFunctionObjArgs(
			npempty.get(), shapeObj.get(), buildPyValue(detail::NpyType<DType>::dtype).get(), nullptr)
		};
		if (!array) throw ExcPropagation{};
		dataPtrOut = (DType*)getArrayDataPtr(array.get());
		return array;
	}

	inline int getArrayDtype(PyObject* obj)
	{
		UniqueObj ndarray = importFrom("numpy", "ndarray");

		if (!PyObject_IsInstance(obj, ndarray.get())) return -1;
		UniqueObj dtype{ PyObject_GetAttrString(obj, "dtype") };
		if (!dtype) throw ExcPropagation{};
		UniqueObj type_num{ PyObject_GetAttrString(dtype.get(), "num") };
		if (!type_num) throw ExcPropagation{};
		int npy_type = (int)PyLong_AsLong(type_num.get());
		if (npy_type == -1 && PyErr_Occurred()) throw ExcPropagation{};
		return npy_type;
	}

	inline size_t getArraySize(PyObject* obj)
	{
		UniqueObj attr{ PyObject_GetAttrString(obj, "size") };
		if (!attr) throw ExcPropagation{};
		size_t v = (size_t)PyLong_AsSize_t(attr.get());
		if (v == (size_t)-1 && PyErr_Occurred()) throw ExcPropagation{};
		return v;
	}

	inline size_t getArrayNdim(PyObject* obj)
	{
		UniqueObj attr{ PyObject_GetAttrString(obj, "ndim") };
		if (!attr) throw ExcPropagation{};
		size_t v = (size_t)PyLong_AsSize_t(attr.get());
		if (v == (size_t)-1 && PyErr_Occurred()) throw ExcPropagation{};
		return v;
	}

	template<typename _Ty, typename _Alloc>
	struct ValueBuilder<std::vector<_Ty, _Alloc>,
		typename std::enable_if<numpy_able<_Ty>::value>::type>
	{
		UniqueObj operator()(const std::vector<_Ty, _Alloc>& v)
		{
			_Ty* ptr = nullptr;
			UniqueObj array = newEmptyArray<_Ty>(ptr, v.size());
			std::memcpy(ptr, v.data(), sizeof(_Ty) * v.size());
			return array;
		}

		bool _toCpp(PyObject* obj, std::vector<_Ty, _Alloc>& out)
		{
			if (detail::NpyType<_Ty>::npy_type >= 0 && getArrayDtype(obj) == detail::NpyType<_Ty>::npy_type)
			{
				_Ty* ptr = (_Ty*)getArrayDataPtr(obj);
				out = std::vector<_Ty>{ ptr, ptr + getArraySize(obj) };
				return true;
			}
			else
			{
				UniqueObj iter{ PyObject_GetIter(obj) }, item;
				if (!iter) return false;
				std::vector<_Ty, _Alloc> v;
				while ((item = UniqueObj{ PyIter_Next(iter.get()) }))
				{
					_Ty i;
					if (!toCpp<_Ty>(item.get(), i)) return false;
					v.emplace_back(std::move(i));
				}
				if (PyErr_Occurred())
				{
					return false;
				}
				out = std::move(v);
				return true;
			}
		}
	};

	template<typename _Ty>
	struct ValueBuilder<std::vector<std::pair<_Ty, _Ty>>,
		typename std::enable_if<numpy_able<_Ty>::value>::type>
	{
		UniqueObj operator()(const std::vector<std::pair<_Ty, _Ty>>& v)
		{
			_Ty* ptr = nullptr;
			UniqueObj array = newEmptyArray<_Ty>(ptr, v.size(), 2);
			std::memcpy(ptr, v.data(), sizeof(_Ty) * v.size() * 2);
			return array;
		}
	};
#endif

	template<typename _Ty>
	struct ValueBuilder<std::vector<_Ty>,
		typename std::enable_if<!numpy_able<_Ty>::value && !numpy_pair_test<_Ty>::value>::type>
	{
		UniqueObj operator()(const std::vector<_Ty>& v)
		{
			UniqueObj ret{ PyList_New(v.size()) };
			size_t id = 0;
			for (auto& e : v)
			{
				PyList_SetItem(ret.get(), id++, buildPyValue(e).release());
			}
			return ret;
		}

		bool _toCpp(PyObject* obj, std::vector<_Ty>& out)
		{
			UniqueObj iter{ PyObject_GetIter(obj) }, item;
			if (!iter) return false;
			std::vector<_Ty> v;
			while ((item = UniqueObj{ PyIter_Next(iter.get()) }))
			{
				_Ty i;
				if (!toCpp<_Ty>(item.get(), i)) return false;
				v.emplace_back(std::move(i));
			}
			if (PyErr_Occurred())
			{
				return false;
			}
			out = std::move(v);
			return true;
		}
	};

	template<typename T, typename Out, typename Msg>
	inline void transform(PyObject* iterable, Out out, Msg&& failMsg)
	{
		if (!iterable) throw ConversionFail{ std::forward<Msg>(failMsg) };
		UniqueObj iter{ PyObject_GetIter(iterable) }, item;
		if (!iter) throw ConversionFail{ std::forward<Msg>(failMsg) };
		while ((item = UniqueObj{ PyIter_Next(iter.get()) }))
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
		try
		{
			while ((item = UniqueObj{ PyIter_Next(iter.get()) }))
			{
				fn(toCpp<T>(item.get()));
			}
		}
		catch (const ForeachFailed&)
		{
			throw ConversionFail{ std::forward<Msg>(failMsg) };
		}

		if (PyErr_Occurred())
		{
			throw ExcPropagation{};
		}
	}

	template<typename T, typename Fn, typename Msg>
	inline void foreachVisit(PyObject* iterable, Fn&& fn, Msg&& failMsg)
	{
		if (!iterable) throw ConversionFail{ std::forward<Msg>(failMsg) };
		UniqueObj iter{ PyObject_GetIter(iterable) }, item;
		if (!iter) throw ConversionFail{ std::forward<Msg>(failMsg) };
		try
		{
			while ((item = UniqueObj{ PyIter_Next(iter.get()) }))
			{
				std::visit(fn, toCpp<T>(item.get()));
			}
		}
		catch (const ForeachFailed&)
		{
			throw ConversionFail{ std::forward<Msg>(failMsg) };
		}

		if (PyErr_Occurred())
		{
			throw ExcPropagation{};
		}
	}

	template<typename T, typename Fn, typename Msg>
	inline void foreachWithPy(PyObject* iterable, Fn&& fn, Msg&& failMsg)
	{
		if (!iterable) throw ConversionFail{ std::forward<Msg>(failMsg) };
		UniqueObj iter{ PyObject_GetIter(iterable) }, item;
		if (!iter) throw ConversionFail{ std::forward<Msg>(failMsg) };
		try
		{
			while ((item = UniqueObj{ PyIter_Next(iter.get()) }))
			{
				fn(toCpp<T>(item.get()), item.get());
			}
		}
		catch (const ForeachFailed&)
		{
			throw ConversionFail{ std::forward<Msg>(failMsg) };
		}

		if (PyErr_Occurred())
		{
			throw ConversionFail{ std::forward<Msg>(failMsg) };
		}
	}

#ifdef USE_NUMPY
	template<typename _Ty>
	inline typename std::enable_if<numpy_able<_Ty>::value, UniqueObj>::type
		buildPyValue(const std::vector<_Ty>& v, cast_to_signed_t)
	{
		std::make_signed_t<_Ty>* ptr = nullptr;
		UniqueObj array = newEmptyArray<std::make_signed_t<_Ty>>(ptr, v.size());
		std::memcpy(ptr, v.data(), sizeof(_Ty) * v.size());
		return array;
	}
#endif

	template<typename _Ty>
	inline typename std::enable_if<
		!numpy_able<typename std::iterator_traits<_Ty>::value_type>::value,
		UniqueObj
	>::type buildPyValue(_Ty first, _Ty last)
	{
		UniqueObj ret{ PyList_New(std::distance(first, last)) };
		size_t id = 0;
		for (; first != last; ++first)
		{
			PyList_SetItem(ret.get(), id++, buildPyValue(*first).release());
		}
		return ret;
	}

	template<typename _Ty>
	inline UniqueObj buildPyValue(const std::vector<_Ty>& v, force_list_t)
	{
		UniqueObj ret{ PyList_New(v.size()) };
		for (size_t i = 0; i < v.size(); ++i)
		{
			PyList_SetItem(ret.get(), i, buildPyValue(v[i]).release());
		}
		return ret;
	}

	template<typename _Ty, typename _Tx>
	inline typename std::enable_if<
		!numpy_able<
		typename std::result_of<_Tx(typename std::iterator_traits<_Ty>::value_type)>::type
		>::value,
		UniqueObj
	>::type buildPyValueTransform(_Ty first, _Ty last, _Tx tx)
	{
		UniqueObj ret{ PyList_New(std::distance(first, last)) };
		size_t id = 0;
		for (; first != last; ++first)
		{
			PyList_SetItem(ret.get(), id++, buildPyValue(tx(*first)).release());
		}
		return ret;
	}

	template<typename _Ty, typename _Tx>
	inline UniqueObj buildPyValueTransform(_Ty&& container, _Tx tx)
	{
		return buildPyValueTransform(std::begin(container), std::end(container), tx);
	}

#ifdef USE_NUMPY
	template<typename _Ty>
	inline typename std::enable_if<
		numpy_able<typename std::iterator_traits<_Ty>::value_type>::value,
		UniqueObj
	>::type buildPyValue(_Ty first, _Ty last)
	{
		using value_type = typename std::iterator_traits<_Ty>::value_type;
		size_t size = std::distance(first, last);
		value_type* ptr = nullptr;
		UniqueObj array = newEmptyArray<value_type>(ptr, size);
		std::copy(first, last, ptr);
		return array;
	}

	template<typename _Ty, typename _Tx>
	inline typename std::enable_if<
		numpy_able<
		typename std::result_of<_Tx(typename std::iterator_traits<_Ty>::value_type)>::type
		>::value,
		UniqueObj
	>::type buildPyValueTransform(_Ty first, _Ty last, _Tx tx)
	{
		using value_type = decltype(tx(*first));
		size_t size = std::distance(first, last);
		value_type* ptr = nullptr;
		UniqueObj array = newEmptyArray<value_type>(ptr, size);
		std::transform(first, last, ptr, tx);
		return array;
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
				PyDict_SetItemString(dict, keys[0], v.get());
			}
			return setDictItem(dict, keys + 1, std::forward<_Rest>(rest)...);
		}

		inline void setDictItem(PyObject* dict, const UniqueObj* keys)
		{

		}

		template<typename _Ty, typename... _Rest>
		inline void setDictItem(PyObject* dict, const UniqueObj* keys, _Ty&& value, _Rest&& ... rest)
		{
			{
				UniqueObj v{ buildPyValue(std::forward<_Ty>(value)) };
				PyDict_SetItem(dict, keys[0].get(), v.get());
			}
			return setDictItem(dict, keys + 1, std::forward<_Rest>(rest)...);
		}

		template<typename _Ty>
		struct IsNull
		{
			bool operator()(const _Ty& v)
			{
				return false;
			}
		};

		template<typename _Ty>
		struct IsNull<_Ty*>
		{
			bool operator()(_Ty* v)
			{
				return !v;
			}
		};

		template<>
		struct IsNull<std::nullptr_t>
		{
			bool operator()(std::nullptr_t v)
			{
				return true;
			}
		};

		template<class _Ty>
		inline bool isNull(_Ty&& v)
		{
			return IsNull<std::remove_reference_t<_Ty>>{}(std::forward<_Ty>(v));
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
				PyDict_SetItemString(dict, keys[0], v.get());
			}
			return setDictItemSkipNull(dict, keys + 1, std::forward<_Rest>(rest)...);
		}

		inline void setDictItemSkipNull(PyObject* dict, const UniqueObj* keys)
		{

		}

		template<typename _Ty, typename... _Rest>
		inline void setDictItemSkipNull(PyObject* dict, const UniqueObj* keys, _Ty&& value, _Rest&& ... rest)
		{
			if (!isNull(value))
			{
				UniqueObj v{ buildPyValue(value) };
				PyDict_SetItem(dict, keys[0].get(), v.get());
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
			PyTuple_SetItem(tuple, _n, buildPyValue(std::forward<_Ty>(first)).release());
			return setTupleItem<_n + 1>(tuple, std::forward<_Rest>(rest)...);
		}
	}

	template<typename... _Rest>
	inline UniqueObj buildPyDict(const char** keys, _Rest&&... rest)
	{
		UniqueObj dict{ PyDict_New() };
		detail::setDictItem(dict.get(), keys, std::forward<_Rest>(rest)...);
		return dict;
	}

	template<typename... _Rest>
	inline UniqueObj buildPyDictSkipNull(const char** keys, _Rest&&... rest)
	{
		UniqueObj dict{ PyDict_New() };
		detail::setDictItemSkipNull(dict.get(), keys, std::forward<_Rest>(rest)...);
		return dict;
	}

	template<typename... _Rest>
	inline UniqueObj buildPyDict(const UniqueObj* keys, _Rest&&... rest)
	{
		UniqueObj dict{ PyDict_New() };
		detail::setDictItem(dict.get(), keys, std::forward<_Rest>(rest)...);
		return dict;
	}

	template<typename... _Rest>
	inline UniqueObj buildPyDictSkipNull(const UniqueObj* keys, _Rest&&... rest)
	{
		UniqueObj dict{ PyDict_New() };
		detail::setDictItemSkipNull(dict.get(), keys, std::forward<_Rest>(rest)...);
		return dict;
	}

	template<typename _Ty>
	inline void setPyDictItem(PyObject* dict, const char* key, _Ty&& value)
	{
		UniqueObj v{ buildPyValue(value) };
		PyDict_SetItemString(dict, key, v.get());
	}

	template<typename... _Rest>
	inline UniqueObj buildPyTuple(_Rest&&... rest)
	{
		UniqueObj tuple{ PyTuple_New(sizeof...(_Rest)) };
		detail::setTupleItem<0>(tuple.get(), std::forward<_Rest>(rest)...);
		return tuple;
	}

	template<typename Ty>
	class TypeWrapper;

	template<class Derived>
	struct CObject
	{
		friend class Module;

		PyObject_HEAD;
		PyObject* managedDict = nullptr;
		PyObject* managedWeakList = nullptr;

		static PyObject* _new(PyTypeObject* subtype, PyObject* args, PyObject* kwargs)
		{
			static_assert(!std::is_polymorphic_v<Derived>, "Derived class must not be polymorphic");
			return handleExc([&]()
			{
				py::UniqueObj ret{ PyType_GenericAlloc(subtype, 0) };
				auto temp = ((Derived*)ret.get())->ob_base;
				new ((Derived*)ret.get()) Derived;
				((Derived*)ret.get())->ob_base = temp;
				return ret.release();
			});
		}

		static void dealloc(Derived* self)
		{
			self->~Derived();

			auto* cobj = static_cast<CObject<Derived>*>(self);
			Py_XDECREF(cobj->managedDict);
			Py_XDECREF(cobj->managedWeakList);

			if (PyType_HasFeature(Py_TYPE(self), Py_TPFLAGS_HAVE_GC))
			{
				PyObject_GC_Del(self);
			}
			else
			{
				PyObject_Free(self);
			}
		}

		using _InitArgs = std::tuple<>;

	private:
		template<class InitArgs, size_t ...idx>
		PY_STRONG_INLINE static void initFromPython(Derived* self, PyObject* args, std::index_sequence<idx...>)
		{
			auto temp = self->ob_base;
			auto temp2 = self->managedDict;
			auto temp3 = self->managedWeakList;
			self->~Derived();
			new (self) Derived{ toCpp<std::tuple_element_t<idx, InitArgs>>(PyTuple_GetItem(args, idx))... };
			self->ob_base = temp;
			self->managedDict = temp2;
			self->managedWeakList = temp3;
		}

		static int init(Derived* self, PyObject* args, PyObject* kwargs)
		{
			return handleExc([&]() -> int
			{
				using InitArgs = typename Derived::_InitArgs;
				if constexpr (std::tuple_size_v<InitArgs> == 0)
				{
					if (args && PyTuple_Size(args) != 0)
					{
						throw TypeError{ "function takes " + std::to_string(std::tuple_size_v<InitArgs>) + " arguments (" + std::to_string(PyTuple_Size(args)) + " given)" };
					}
				}

				if (std::tuple_size_v<InitArgs> != PyTuple_Size(args))
				{
					throw TypeError{ "function takes " + std::to_string(std::tuple_size_v<InitArgs>) + " arguments (" + std::to_string(PyTuple_Size(args)) + " given)" };
				}
				if (kwargs)
				{
					throw TypeError{ "function takes positional arguments only" };
				}

				auto temp = self->ob_base;
				initFromPython<InitArgs>(self, args, std::make_index_sequence<std::tuple_size_v<InitArgs>>{});
				self->ob_base = temp;
				return 0;
			});
		}

		friend class TypeWrapper<Derived>;
	};

	template<class Ty>
	struct PObject
	{
		friend class Module;

		PyObject_HEAD;
		PyObject* managedDict = nullptr;
		PyObject* managedWeakList = nullptr;
		Ty value;

		Ty* operator->()
		{
			return &value;
		}

		const Ty* operator->() const
		{
			return &value;
		}

	private:
		static PyObject* _new(PyTypeObject* subtype, PyObject* args, PyObject* kwargs)
		{
			return handleExc([&]()
			{
				py::UniqueObj ret{ PyType_GenericAlloc(subtype, 0) };
				new (&((PObject*)ret.get())->value) Ty;
				return ret.release();
			});
		}

		static void dealloc(PObject* self)
		{
			self->value.~Ty();
			Py_XDECREF(self->managedDict);
			Py_XDECREF(self->managedWeakList);

			if (PyType_HasFeature(Py_TYPE(self), Py_TPFLAGS_HAVE_GC))
			{
				PyObject_GC_Del(self);
			}
			else
			{
				PyObject_Free(self);
			}
		}

		template<class InitArgs, size_t ...idx>
		PY_STRONG_INLINE static void initFromPython(PObject* self, PyObject* args, std::index_sequence<idx...>)
		{
			self->value.~Ty();
			new (&self->value) Ty{ toCpp<std::tuple_element_t<idx, InitArgs>>(PyTuple_GetItem(args, idx))... };
		}

		static int init(PObject* self, PyObject* args, PyObject* kwargs)
		{
			return handleExc([&]() -> int
			{
				using InitArgs = typename Ty::_InitArgs;
				if constexpr (std::tuple_size_v<InitArgs> == 0)
				{
					if (args && PyTuple_Size(args) != 0)
					{
						throw TypeError{ "function takes " + std::to_string(std::tuple_size_v<InitArgs>) + " arguments (" + std::to_string(PyTuple_Size(args)) + " given)" };
					}
				}

				if (std::tuple_size_v<InitArgs> != PyTuple_Size(args))
				{
					throw TypeError{ "function takes " + std::to_string(std::tuple_size_v<InitArgs>) + " arguments (" + std::to_string(PyTuple_Size(args)) + " given)" };
				}
				if (kwargs)
				{
					throw TypeError{ "function takes positional arguments only" };
				}

				initFromPython<InitArgs>(self, args, std::make_index_sequence<std::tuple_size_v<InitArgs>>{});
				return 0;
			});
		}

		friend class TypeWrapper<PObject<Ty>>;
	};

	template<class Ty>
	PObject<Ty>* getPObjectAddress(Ty* value)
	{
		return reinterpret_cast<PObject<Ty>*>(reinterpret_cast<char*>(value) - offsetof(PObject<Ty>, value));;
	}

	template<class Ty>
	const PObject<Ty>* getPObjectAddress(const Ty* value)
	{
		return reinterpret_cast<const PObject<Ty>*>(reinterpret_cast<const char*>(value) - offsetof(PObject<Ty>, value));;
	}

	template<class Derived, class RetTy, class Future = std::future<RetTy>>
	struct ResultIter : public CObject<Derived>
	{
		using ReturnTy = RetTy;
		using FutureTy = Future;
		UniqueObj inputIter;
		std::deque<Future> futures;
		std::deque<SharedObj> inputItems;
		bool echo = false;

		ResultIter() = default;
		ResultIter(ResultIter&&) = default;
		ResultIter& operator=(ResultIter&&) = default;

		ResultIter(const ResultIter&) = delete;
		ResultIter& operator=(const ResultIter&) = delete;

		~ResultIter()
		{
			waitQueue();
		}

		void waitQueue()
		{
			while (!futures.empty())
			{
				auto f = std::move(futures.front());
				futures.pop_front();
				f.get();
			}
		}

		py::UniqueCObj<Derived> iter() const
		{
			Py_INCREF(this);
			return py::UniqueCObj<Derived>{ static_cast<Derived*>(const_cast<ResultIter*>(this)) };
		}

		py::UniqueObj iternext()
		{
			if (!feed() && futures.empty()) throw py::ExcPropagation{};
			auto f = std::move(futures.front());
			futures.pop_front();
			if (echo)
			{
				auto input = std::move(inputItems.front());
				inputItems.pop_front();
				return buildPyTuple(static_cast<Derived*>(this)->buildPy(f.get()), input);
			}
			else
			{
				return static_cast<Derived*>(this)->buildPy(f.get());
			}
		}

		bool feed()
		{
			SharedObj item{ PyIter_Next(inputIter.get()) };
			if (!item)
			{
				if (PyErr_Occurred()) throw ExcPropagation{};
				return false;
			}
			if (echo) inputItems.emplace_back(item);
			futures.emplace_back(static_cast<Derived*>(this)->feedNext(std::move(item)));
			return true;
		}

		Future feedNext(py::SharedObj&& next)
		{
			return {};
		}

		UniqueObj buildPy(RetTy&& v)
		{
			return py::buildPyValue(std::move(v));
		}
	};

	class Module
	{
		PyModuleDef def;
		PyObject* mod = nullptr;

	public:
		Module(const char* name, const char* doc)
		{
			def.m_base = PyModuleDef_HEAD_INIT;
			def.m_name = name;
			def.m_doc = doc;
			def.m_size = -1;
			def.m_methods = nullptr;
			def.m_slots = nullptr;
			def.m_traverse = nullptr;
			def.m_clear = nullptr;
			def.m_free = nullptr;
		}

		template<class Def>
		Module(const char* name, const char* doc, Def&& fn);

		template<class... Defs>
		PyObject* init(Defs&&... defs)
		{
			mod = PyModule_Create(&def);
#ifdef Py_GIL_DISABLED
			PyUnstable_Module_SetGIL(mod, Py_MOD_GIL_NOT_USED);
#endif
			((addType(std::forward<Defs>(defs))), ...);
			return mod;
		}

		template<class TypeDef>
		bool addType(TypeDef&& def);
	};

	template<typename Ty>
	class TypeWrapper
	{
	public:
		static PyTypeObject* obj;

		static constexpr PyObject* getTypeObj() { return (PyObject*)obj; }
	};

	template<typename Ty> PyTypeObject* TypeWrapper<Ty>::obj = nullptr;

	template<typename Ty>
	PyTypeObject* const& Type = TypeWrapper<Ty>::obj;

	template<class Ty = PyObject>
	inline UniqueCObj<Ty> makeNewObject(PyTypeObject* type)
	{
		UniqueCObj<Ty> ret;
		if (PyType_HasFeature(type, Py_TPFLAGS_HAVE_GC))
		{
			ret = UniqueCObj<Ty>{ (Ty*)PyObject_GC_New(CObject<Ty>, type) };
		}
		else
		{
			ret = UniqueCObj<Ty>{ (Ty*)PyObject_New(CObject<Ty>, type) };
		}
		if constexpr (!std::is_same_v<Ty, PyObject>)
		{
			new (ret.get()) Ty;
		}
		return ret;
	}

	template<class Ty>
	inline UniqueCObj<Ty> makeNewObject()
	{
		return makeNewObject<Ty>(Type<Ty>);
	}

	template<class Ty>
	bool ValueBuilder<UniqueCObj<Ty>>::_toCpp(PyObject* obj, UniqueCObj<Ty>& out)
	{
		out = UniqueCObj<Ty>{ (Ty*)obj };
		if (!std::is_same_v<Ty, PyObject> && !PyObject_IsInstance(obj, (PyObject*)Type<Ty>))
		{
			return false;
		}
		Py_INCREF(obj);
		return true;
	}

	class CustomExcHandler
	{
		static std::unordered_map<std::type_index, PyObject*>& _get()
		{
			static std::unordered_map<std::type_index, PyObject*> handlers;
			return handlers;
		}
	public:
		
		template<class CustomExc, class PyExc>
		static void add()
		{
			_get()[std::type_index(typeid(CustomExc))] = PyExc{ "" }.pytype();
		}

		static const std::unordered_map<std::type_index, PyObject*>& get()
		{
			return _get();
		}
	};

	namespace detail
	{
		inline void setPyError(PyObject* errType, const char* errMsg)
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
				PyObject* et = errType;
				val2 = PyObject_CallFunctionObjArgs(et, py::UniqueObj{ buildPyValue(errMsg) }.get(), nullptr);
				PyException_SetCause(val2, val);
				PyErr_SetObject(et, val2);
				Py_DECREF(val2);
			}
			else
			{
				PyErr_SetString(errType, errMsg);
			}
		}
	}

	template<typename _Fn>
	PY_STRONG_INLINE auto handleExc(_Fn&& fn)
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
			detail::setPyError(e.pytype(), e.what());
		}
		catch (const std::exception& e)
		{
			auto customHandlers = CustomExcHandler{}.get();
			auto it = customHandlers.find(std::type_index(typeid(e)));
			if (it == customHandlers.end())
			{
				/*std::cerr << "Uncaughted c++ exception: " << e.what() << std::endl;
				PyErr_SetString(PyExc_RuntimeError, e.what());*/
				throw;
			}
			detail::setPyError(it->second, e.what());
		}
		return nullptr;
	}

	template<typename _Fn>
	PY_STRONG_INLINE auto handleExc(_Fn&& fn)
		-> typename std::enable_if<std::is_same<decltype(fn()), UniqueObj>::value, decltype(fn())>::type
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
			detail::setPyError(e.pytype(), e.what());
		}
		catch (const std::exception& e)
		{
			auto customHandlers = CustomExcHandler{}.get();
			auto it = customHandlers.find(std::type_index(typeid(e)));
			if (it == customHandlers.end())
			{
				/*std::cerr << "Uncaughted c++ exception: " << e.what() << std::endl;
				PyErr_SetString(PyExc_RuntimeError, e.what());*/
				throw;
			}
			detail::setPyError(it->second, e.what());
		}
		return UniqueObj{ nullptr };
	}

	template<typename _Fn>
	PY_STRONG_INLINE auto handleExc(_Fn&& fn)
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
			detail::setPyError(e.pytype(), e.what());
		}
		catch (const std::exception& e)
		{
			auto customHandlers = CustomExcHandler{}.get();
			auto it = customHandlers.find(std::type_index(typeid(e)));
			if (it == customHandlers.end())
			{
				throw;
			}
			detail::setPyError(it->second, e.what());
		}
		/*catch (const std::exception& e)
		{
			std::cerr << "Uncaughted c++ exception: " << e.what() << std::endl;
			PyErr_SetString(PyExc_RuntimeError, e.what());
		}*/
		return -1;
	}

	namespace detail
	{
		template <typename T>
		struct IsFunctionObjectImpl
		{
		private:
			using Yes = char(&)[1];
			using No = char(&)[2];

			struct Fallback
			{
				void operator()();
			};

			struct Derived : T, Fallback
			{
			};

			template <typename U, U>
			struct Check;

			template <typename>
			static Yes Test(...);

			template <typename C>
			static No Test(Check<void(Fallback::*)(), &C::operator()>*);

		public:
			static constexpr bool value{ sizeof(Test<Derived>(0)) == sizeof(Yes) };
		};
	}

	template <typename T>
	struct IsFunctionObject : std::conditional<
		std::is_class<T>::value,
		detail::IsFunctionObjectImpl<T>,
		std::false_type
	>::type
	{
	};

	namespace detail
	{
		template <typename T>
		struct CppWrapperImpl;

		/* global function object */
		template <typename R, typename... Ts>
		struct CppWrapperImpl<R(Ts...)>
		{
			using Type = R(Ts...);
			using FunctionPointerType = R(*)(Ts...);
			using ReturnType = R;
			using ClassType = void;
			using ArgsTuple = std::tuple<Ts...>;

			template <std::size_t N>
			using Arg = typename std::tuple_element<N, ArgsTuple>::type;

			using ClassPtrOrFirstArgType = std::conditional_t<sizeof...(Ts) == 0, void, Arg<0>>;

			static const std::size_t nargs{ sizeof...(Ts) };

			template<Type func, size_t ...idx>
			static constexpr auto callFromPython(void*, PyObject* args, PyObject* kwargs, std::index_sequence<idx...>)
			{
				if (PyTuple_Size(args) != std::tuple_size_v<ArgsTuple>)
				{
					throw TypeError{ "function takes " + std::to_string(std::tuple_size_v<ArgsTuple>) + " arguments (" + std::to_string(PyTuple_Size(args)) + " given)" };
				}
				if (kwargs)
				{
					throw TypeError{ "function takes positional arguments only" };
				}

				return func(toCpp<std::remove_cv_t<std::remove_reference_t<Arg<idx>>>>(PyTuple_GetItem(args, idx))...);
			}

			template<Type func, size_t _nargs = nargs>
			static constexpr std::enable_if_t<_nargs == 1, ReturnType> get(Ts&&... args)
			{
				return func(std::forward<Ts>(args)...);
			}

			template<Type func, size_t _nargs = nargs>
			static constexpr std::enable_if_t<_nargs == 2, int> set(Arg<0> arg0, PyObject* val)
			{
				func(arg0, toCpp<std::remove_cv_t<std::remove_reference_t<Arg<1>>>>(val));
				return 0;
			}
		};

		/* global function pointer */
		template <typename R, typename... Ts>
		struct CppWrapperImpl<R(*)(Ts...)>
		{
			using Type = R(*)(Ts...);
			using FunctionPointerType = R(*)(Ts...);
			using ReturnType = R;
			using ClassType = void;
			using ArgsTuple = std::tuple<Ts...>;

			template <std::size_t N>
			using Arg = typename std::tuple_element<N, ArgsTuple>::type;

			using ClassPtrOrFirstArgType = std::conditional_t<sizeof...(Ts) == 0, void, Arg<0>>;

			static const std::size_t nargs{ sizeof...(Ts) };

			template<Type func, size_t ...idx>
			static constexpr auto call(void*, PyObject* args, PyObject* kwargs, std::index_sequence<idx...>)
			{
				if (PyTuple_Size(args) != std::tuple_size_v<ArgsTuple>)
				{
					throw TypeError{ "function takes " + std::to_string(std::tuple_size_v<ArgsTuple>) + " arguments (" + std::to_string(PyTuple_Size(args)) + " given)" };
				}
				if (kwargs)
				{
					throw TypeError{ "function takes positional arguments only" };
				}

				return func(toCpp<std::remove_cv_t<std::remove_reference_t<Arg<idx>>>>(PyTuple_GetItem(args, idx))...);
			}

			template<Type func, size_t _nargs = nargs>
			static constexpr std::enable_if_t<_nargs == 1, ReturnType> get(Ts&&... args)
			{
				return func(std::forward<Ts>(args)...);
			}

			template<Type func, size_t _nargs = nargs>
			static constexpr std::enable_if_t<_nargs == 2, int> set(Arg<0> arg0, PyObject* val)
			{
				func(arg0, toCpp<std::remove_cv_t<std::remove_reference_t<Arg<1>>>>(val));
				return 0;
			}
		};

		/* member function pointer */
		template <typename C, typename R, typename... Ts>
		struct CppWrapperImpl<R(C::*)(Ts...)>
		{
			using Type = R(C::*)(Ts...);
			using FunctionPointerType = R(*)(C*, Ts...);
			using ReturnType = R;
			using ClassType = C;
			using ArgsTuple = std::tuple<Ts...>;

			template <std::size_t N>
			using Arg = typename std::tuple_element<N, ArgsTuple>::type;

			using ClassPtrOrFirstArgType = C*;

			static constexpr std::size_t nargs{ sizeof...(Ts) };

			template<Type func, size_t ...idx>
			static constexpr auto call(ClassType* self, PyObject* args, PyObject* kwargs, std::index_sequence<idx...>)
			{
				if (PyTuple_Size(args) != std::tuple_size_v<ArgsTuple>)
				{
					throw TypeError{ "function takes " + std::to_string(std::tuple_size_v<ArgsTuple>) + " arguments (" + std::to_string(PyTuple_Size(args)) + " given)" };
				}
				if (kwargs)
				{
					throw TypeError{ "function takes positional arguments only" };
				}

				return (self->*func)(toCpp<std::remove_cv_t<std::remove_reference_t<Arg<idx>>>>(PyTuple_GetItem(args, idx))...);
			}

			template<Type func, size_t _nargs = nargs>
			static constexpr std::enable_if_t<_nargs == 0, ReturnType> len(ClassType* self)
			{
				static_assert(std::is_integral_v<ReturnType>, "len() must return integral type");
				return (self->*func)();
			}

			template<Type func, size_t _nargs = nargs>
			static constexpr std::enable_if_t<_nargs == 1, ReturnType> ssizearg(ClassType* self, Py_ssize_t idx)
			{
				static_assert(std::is_integral_v<std::remove_cv_t<std::remove_reference_t<Arg<0>>>>, "ssizearg() must take one integral argument");
				return (self->*func)(idx);
			}

			template<Type func, size_t _nargs = nargs>
			static constexpr std::enable_if_t<_nargs == 1, ReturnType> binary(ClassType* self, PyObject* arg)
			{
				return (self->*func)(toCpp< std::remove_cv_t<std::remove_reference_t<Arg<0>>>>(arg));
			}

			template<Type func, size_t _nargs = nargs>
			static constexpr std::enable_if_t<_nargs == 0, ReturnType> repr(ClassType* self)
			{
				return (self->*func)();
			}

			template<Type func, size_t _nargs = nargs>
			static constexpr std::enable_if_t<_nargs == 0, ReturnType> get(ClassType* self)
			{
				return (self->*func)();
			}

			template<Type func, size_t _nargs = nargs>
			static constexpr std::enable_if_t<_nargs == 1, int> set(ClassType* self, PyObject* val)
			{
				(self->*func)(toCpp<std::remove_cv_t<std::remove_reference_t<Arg<0>>>>(val));
				return 0;
			}
		};

		/* const member function pointer */
		template <typename C, typename R, typename... Ts>
		struct CppWrapperImpl<R(C::*)(Ts...) const>
		{
			using Type = R(C::*)(Ts...) const;
			using FunctionPointerType = R(*)(C*, Ts...);
			using ReturnType = R;
			using ClassType = C;
			using ArgsTuple = std::tuple<Ts...>;

			template <std::size_t N>
			using Arg = typename std::tuple_element<N, ArgsTuple>::type;

			using ClassPtrOrFirstArgType = C*;

			static constexpr std::size_t nargs{ sizeof...(Ts) };

			template<Type func, size_t ...idx>
			static constexpr auto call(const ClassType* self, PyObject* args, PyObject* kwargs, std::index_sequence<idx...>)
			{
				if (PyTuple_Size(args) != std::tuple_size_v<ArgsTuple>)
				{
					throw TypeError{ "function takes " + std::to_string(std::tuple_size_v<ArgsTuple>) + " arguments (" + std::to_string(PyTuple_Size(args)) + " given)" };
				}
				if (kwargs)
				{
					throw TypeError{ "function takes positional arguments only" };
				}

				return (self->*func)(toCpp<std::remove_const_t<std::remove_reference_t<Arg<idx>>>>(PyTuple_GetItem(args, idx))...);
			}

			template<Type func, size_t _nargs = nargs>
			static constexpr std::enable_if_t<_nargs == 0, ReturnType> len(const ClassType* self)
			{
				static_assert(std::is_integral_v<ReturnType>, "len() must return integral type");
				return (self->*func)();
			}

			template<Type func, size_t _nargs = nargs>
			static constexpr std::enable_if_t<_nargs == 1, ReturnType> ssizearg(const ClassType* self, Py_ssize_t idx)
			{
				static_assert(std::is_integral_v< std::remove_cv_t<std::remove_reference_t<Arg<0>>>>, "ssizearg() must take one integral argument");
				return (self->*func)(idx);
			}

			template<Type func, size_t _nargs = nargs>
			static constexpr std::enable_if_t<_nargs == 1, ReturnType> binary(ClassType* self, PyObject* arg)
			{
				return (self->*func)(toCpp< std::remove_cv_t<std::remove_reference_t<Arg<0>>>>(arg));
			}

			template<Type func, size_t _nargs = nargs>
			static constexpr std::enable_if_t<_nargs == 0, ReturnType> repr(const ClassType* self)
			{
				return (self->*func)();
			}

			template<Type func, size_t _nargs = nargs>
			static constexpr std::enable_if_t<_nargs == 0, ReturnType> get(const ClassType* self)
			{
				return (self->*func)();
			}
		};

		/* member variable pointer */
		template <typename C, typename R>
		struct CppWrapperImpl<R(C::*)>
		{
			using Type = R(C::*);
			using ReturnType = R;
			using ClassType = C;
			using ArgsTuple = std::tuple<>;

			using ClassPtrOrFirstArgType = C*;

			template<Type ptr>
			static constexpr const ReturnType& get(ClassType* self)
			{
				return self->*ptr;
			}

			template<Type ptr>
			static constexpr int set(ClassType* self, PyObject* val)
			{
				self->*ptr = toCpp<ReturnType>(val);
				return 0;
			}
		};

		template<class Base>
		struct CppWrapperInterface : public Base
		{
			using T = typename Base::Type;

			template<T func>
			static constexpr PyCFunctionWithKeywords call()
			{
				return [](PyObject* self, PyObject* args, PyObject* kwargs) -> PyObject*
				{
					return handleExc([&]() -> PyObject*
					{
						if constexpr (std::is_same_v<typename Base::ReturnType, void>)
						{
							Base::template call<func>((typename Base::ClassType*)self, args, kwargs, std::make_index_sequence<std::tuple_size_v<typename Base::ArgsTuple>>{});
							return buildPyValue(nullptr).release();
						}
						else
						{
							return buildPyValue(Base::template call<func>((typename Base::ClassType*)self, args, kwargs, std::make_index_sequence<std::tuple_size_v<typename Base::ArgsTuple>>{})).release();
						}
					});
				};
			}

			template<T ptr>
			static constexpr lenfunc len()
			{
				return (lenfunc)[](PyObject* self)->Py_ssize_t
				{
					return handleExc([&]()
					{
						return Base::template len<ptr>((typename Base::ClassType*)self);
					});
				};
			}

			template<T ptr>
			static constexpr reprfunc repr()
			{
				return (reprfunc)[](PyObject* self)->PyObject*
				{
					return handleExc([&]()
					{
						return buildPyValue(Base::template repr<ptr>((typename Base::ClassType*)self)).release();
					});
				};
			}

			template<T ptr>
			static constexpr ssizeargfunc ssizearg()
			{
				return (ssizeargfunc)[](PyObject* self, Py_ssize_t idx)->PyObject*
				{
					return handleExc([&]()
					{
						return buildPyValue(Base::template ssizearg<ptr>((typename Base::ClassType*)self, idx)).release();
					});
				};
			}

			template<T ptr>
			static constexpr binaryfunc binary()
			{
				return (binaryfunc)[](PyObject* self, PyObject* arg)->PyObject*
				{
					return handleExc([&]()
					{
						return buildPyValue(Base::template binary<ptr>((typename Base::ClassType*)self, arg)).release();
					});
				};
			}

			template<T ptr>
			static constexpr ternaryfunc ternary()
			{
				return (ternaryfunc)[](PyObject* self, PyObject* arg, PyObject* kwarg)->PyObject*
				{
					// ignore kwarg since we don't support it anyway
					return handleExc([&]()
					{
						return buildPyValue(Base::template binary<ptr>((typename Base::ClassType*)self, arg)).release();
					});
				};
			}

			template<T ptr>
			static constexpr getter get()
			{
				return (getter)[](PyObject* self, void* closure)->PyObject*
				{
					return handleExc([&]()
					{
						return buildPyValue(Base::template get<ptr>((typename Base::ClassPtrOrFirstArgType)self)).release();
					});
				};
			}

			template<T ptr>
			static constexpr setter set()
			{
				return (setter)[](PyObject* self, PyObject* val, void* closure) -> int
				{
					return handleExc([&]()
					{
						return Base::template set<ptr>((typename Base::ClassPtrOrFirstArgType)self, val);
					});
				};
			}
		};

#define _PY_DETAILED_TEST_MEMFN(name, func) \
		template <typename T>\
		class name\
		{\
			using one = char;\
			struct two { char x[2]; };\
			template <typename C> static one test(decltype(&C::func));\
			template <typename C> static two test(...);\
		public:\
			enum { value = sizeof(test<T>(0)) == sizeof(char) };\
		};

		_PY_DETAILED_TEST_MEMFN(HasRepr, repr);
		_PY_DETAILED_TEST_MEMFN(HasIter, iter);
		_PY_DETAILED_TEST_MEMFN(HasIterNext, iternext);
		_PY_DETAILED_TEST_MEMFN(HasLen, len);

		template<auto f, auto g, class FClass, class FArgs, class GClass, class GArgs>
		struct CompositorImpl;

		template<auto f, auto g, class FClass, typename... FArgs, class GClass, typename... GArgs>
		struct CompositorImpl<f, g, FClass, std::tuple<FArgs...>, GClass, std::tuple<GArgs...>> : public FClass
		{
			auto call(FArgs&&... args, GArgs&&... args2)
			{
				return std::invoke(g, std::invoke(f, this, std::forward<FArgs>(args)...), std::forward<GArgs>(args2)...);
			}
		};

		template<auto f, auto g, typename... FArgs, class GClass, typename... GArgs>
		struct CompositorImpl<f, g, void, std::tuple<FArgs...>, GClass, std::tuple<GArgs...>>
		{
			static auto call(FArgs&&... args, GArgs&&... args2)
			{
				return std::invoke(g, std::invoke(f, std::forward<FArgs>(args)...), std::forward<GArgs>(args2)...);
			}
		};

		template<auto f, auto g, class FClass, typename... FArgs, class GArg1, typename... GArgs>
		struct CompositorImpl<f, g, FClass, std::tuple<FArgs...>, void, std::tuple<GArg1, GArgs...>> : public FClass
		{
			auto call(FArgs&&... args, GArgs&&... args2)
			{
				return std::invoke(g, std::invoke(f, this, std::forward<FArgs>(args)...), std::forward<GArgs>(args2)...);
			}
		};

		template<auto f, auto g, typename... FArgs, class GArg1, typename... GArgs>
		struct CompositorImpl<f, g, void, std::tuple<FArgs...>, void, std::tuple<GArg1, GArgs...>>
		{
			static auto call(FArgs&&... args, GArgs&&... args2)
			{
				return std::invoke(g, std::invoke(f, std::forward<FArgs>(args)...), std::forward<GArgs>(args2)...);
			}
		};

		template<auto f, auto g>
		struct Compositor : public CompositorImpl<
			f,
			g,
			typename CppWrapperImpl<decltype(f)>::ClassType,
			typename CppWrapperImpl<decltype(f)>::ArgsTuple,
			typename CppWrapperImpl<decltype(g)>::ClassType,
			typename CppWrapperImpl<decltype(g)>::ArgsTuple>
		{
		};

		template<auto f, auto g>
		static constexpr auto compose = &Compositor<f, g>::call;
	}

	template <typename T, typename = void>
	struct CppWrapper : detail::CppWrapperInterface<detail::CppWrapperImpl<T>>
	{
	};

	template <typename T>
	struct CppWrapper<T, typename std::enable_if<IsFunctionObject<T>::value>::type> :
		detail::CppWrapperInterface<detail::CppWrapperImpl<decltype(&T::operator())>>
	{
	};

	struct NativeMethod
	{
		const char* name;
		PyCFunctionWithKeywords fnPtr;
		int flags;

		operator PyMethodDef() const
		{
			return { (char*)name, (PyCFunction)fnPtr, flags, "" };
		}
	};

	struct NativePropety
	{
		const char* name;
		getter getFn;
		setter setFn;

		operator PyGetSetDef() const
		{
			return { (char*)name, getFn, setFn, "", nullptr };
		}
	};

	template<auto memFn, int flags>
	constexpr auto makeMethodDef()
	{
		return NativeMethod{ nullptr, CppWrapper<decltype(memFn)>::template call<memFn>(), flags };
	}

	template<auto getter>
	constexpr auto makePropertyDef()
	{
		return NativePropety{ nullptr, CppWrapper<decltype(getter)>::template get<getter>(), nullptr };
	}

	template<auto getter, auto setter>
	constexpr auto makePropertyDef()
	{
		return NativePropety{ nullptr, CppWrapper<decltype(getter)>::template get<getter>(), CppWrapper<decltype(setter)>::template set<setter>() };
	}

	template<auto memFn, int flags>
	static constexpr NativeMethod methodDef = makeMethodDef<memFn, flags>();

	template<auto getter>
	static constexpr NativePropety getDef = makePropertyDef<getter>();

	template<auto getter, auto setter>
	static constexpr NativePropety getsetDef = makePropertyDef<getter, setter>();

	template<auto base, auto getter>
	static constexpr NativePropety getDefComposition = makePropertyDef<detail::compose<base, getter>>();

	template<auto base, auto getter, auto setter>
	static constexpr NativePropety getsetDefComposition = makePropertyDef<detail::compose<base, getter>, detail::compose<base, setter>>();

	namespace detail
	{
		template<const auto&... vs>
		struct value_tuple
		{
			static constexpr size_t size = sizeof...(vs);
		};

		template<const auto& first, class Ty>
		struct prepend;

		template<const auto& first, const auto&... vs>
		struct prepend<first, value_tuple<vs...>>
		{
			using type = value_tuple<first, vs...>;
		};

		template<const auto&... vs>
		struct FilterMethods;

		template<>
		struct FilterMethods<>
		{
			using Type = value_tuple<>;
		};

		template<const auto& first, const auto&... rest>
		struct FilterMethods<first, rest...>
		{
			using FirstType = std::remove_cv_t<std::remove_reference_t<decltype(first)>>;
			using Tail = typename FilterMethods<rest...>::Type;
			using Type = std::conditional_t<
				std::is_same_v<FirstType, NativeMethod>,
				typename prepend<first, Tail>::type,
				Tail>;
		};

		template<const auto&... vs>
		struct FilterProperties;

		template<>
		struct FilterProperties<>
		{
			using Type = value_tuple<>;
		};

		template<const auto& first, const auto&... rest>
		struct FilterProperties<first, rest...>
		{
			using FirstType = std::remove_cv_t<std::remove_reference_t<decltype(first)>>;
			using Tail = typename FilterProperties<rest...>::Type;
			using Type = std::conditional_t<
				std::is_same_v<FirstType, NativePropety>,
				typename prepend<first, Tail>::type,
				Tail>;
		};

		template<class Ty>
		struct MethodDefBuilder;

		template<const auto&... vs>
		struct MethodDefBuilder<value_tuple<vs...>>
		{
			static constexpr auto get()
			{
				return std::array<PyMethodDef, sizeof...(vs) + 1>{ ((PyMethodDef)vs)..., { nullptr } };
			}
		};

		template<class Ty>
		struct PropertyDefBuilder;

		template<const auto&... vs>
		struct PropertyDefBuilder<value_tuple<vs...>>
		{
			static constexpr auto get()
			{
				return std::array<PyGetSetDef, sizeof...(vs) + 1>{ ((PyGetSetDef)vs)..., { nullptr } };
			}
		};

		template<class Ty>
		struct IsPObject : std::false_type
		{
		};

		template<class Ty>
		struct IsPObject<PObject<Ty>> : std::true_type
		{
		};
	}

	template<class Ty, class BaseTy, const auto& ... defs>
	class TypeDefinition
	{
		friend class Module;

		using Methods = typename detail::FilterMethods<defs...>::Type;
		using Properties = typename detail::FilterProperties<defs...>::Type;

	public:
		const char* typeName = nullptr;
		const char* typeNameInModule = nullptr;
		int typeFlags = Py_TPFLAGS_DEFAULT;
		std::vector<const char*> methodNames;
		std::vector<const char*> propertyNames;
		std::vector<PyType_Slot> slots;

		using Class = Ty;
		using BaseClass = BaseTy;
		static_assert(std::is_base_of_v<CObject<Class>, Class> || detail::IsPObject<Class>::value, "Only CObject or PObject has its TypeDefinition.");
		static_assert(std::is_base_of_v<CObject<BaseClass>, BaseClass> || detail::IsPObject<Class>::value || std::is_same_v<BaseClass, void>, "BaseClass must be derived from CObject, PObject or void.");

		inline static auto methodDefs = detail::MethodDefBuilder<Methods>::get();
		inline static auto propertyDefs = detail::PropertyDefBuilder<Properties>::get();

		constexpr TypeDefinition(
			const char* _typeName = nullptr,
			const char* _typeNameInModule = nullptr,
			int _typeFlags = Py_TPFLAGS_DEFAULT,
			const std::vector<const char*>& _methodNames = {},
			const std::vector<const char*>& _propertyNames = {},
			const std::vector<PyType_Slot>& _slots = {}
		)
			: typeName{ _typeName },
			typeNameInModule{ _typeNameInModule },
			typeFlags{ _typeFlags },
			methodNames{ _methodNames },
			propertyNames{ _propertyNames },
			slots{ _slots }
		{
		}

		template<auto memFn>
		constexpr auto method(const char* name) const
		{
			constexpr auto fn = []() {
				if constexpr (detail::IsPObject<Ty>::value) return detail::compose<&Ty::value, memFn>;
				else return memFn;
			}();
			auto ret = TypeDefinition<Ty, BaseTy, defs..., methodDef<fn, METH_VARARGS | METH_KEYWORDS>>{
				typeName, typeNameInModule, typeFlags,
				methodNames, propertyNames, slots
			};
			ret.methodNames.emplace_back(name);
			return ret;
		}

		template<auto memFn>
		constexpr auto staticMethod(const char* name) const
		{
			auto ret = TypeDefinition<Ty, BaseTy, defs..., methodDef<memFn, METH_VARARGS | METH_KEYWORDS | METH_STATIC>>{
				typeName, typeNameInModule, typeFlags,
				methodNames, propertyNames, slots
			};
			ret.methodNames.emplace_back(name);
			return ret;
		}

		template<auto getter>
		constexpr auto property(const char* name) const
		{
			constexpr auto g = []() {
				if constexpr (detail::IsPObject<Ty>::value) return detail::compose<&Ty::value, getter>;
				else return getter;
			}();

			auto ret = TypeDefinition<Ty, BaseTy, defs..., getDef<g>>{
				typeName, typeNameInModule, typeFlags,
				methodNames, propertyNames, slots
			};
			ret.propertyNames.emplace_back(name);
			return ret;
		}

		template<auto getter, auto setter>
		constexpr auto property(const char* name) const
		{
			constexpr auto g = []() {
				if constexpr (detail::IsPObject<Ty>::value) return detail::compose<&Ty::value, getter>;
				else return getter;
			}();
			constexpr auto s = []() {
				if constexpr (detail::IsPObject<Ty>::value) return detail::compose<&Ty::value, setter>;
				else return setter;
			}();
			auto ret = TypeDefinition<Ty, BaseTy, defs..., getsetDef<g, s>>{
				typeName, typeNameInModule, typeFlags,
				methodNames, propertyNames, slots
			};
			ret.propertyNames.emplace_back(name);
			return ret;
		}

		template<auto base, auto getter>
		constexpr auto property2(const char* name) const
		{
			constexpr auto b = []() {
				if constexpr (detail::IsPObject<Ty>::value) return detail::compose<&Ty::value, base>;
				else return base;
			}();
			auto ret = TypeDefinition<Ty, BaseTy, defs..., getDefComposition<b, getter>>{
				typeName, typeNameInModule, typeFlags,
				methodNames, propertyNames, slots
			};
			ret.propertyNames.emplace_back(name);
			return ret;
		}

		template<auto base, auto getter, auto setter>
		constexpr auto property2(const char* name) const
		{
			constexpr auto b = []() {
				if constexpr (detail::IsPObject<Ty>::value) return detail::compose<&Ty::value, base>;
				else return base;
			}();
			auto ret = TypeDefinition<Ty, BaseTy, defs..., getsetDefComposition<b, getter, setter>>{
				typeName, typeNameInModule, typeFlags,
				methodNames, propertyNames, slots
			};
			ret.propertyNames.emplace_back(name);
			return ret;
		}

		template<auto memFn>
		constexpr auto getAttrO() const
		{
			constexpr auto fn = []() {
				if constexpr (detail::IsPObject<Ty>::value) return detail::compose<&Ty::value, memFn>;
				else return memFn;
			}();
			auto* ptr = py::CppWrapper<std::remove_const_t<decltype(fn)>>::template binary<fn>();
			auto ret = TypeDefinition<Ty, BaseTy, defs...>{
				typeName, typeNameInModule, typeFlags,
				methodNames, propertyNames, slots
			};
			ret.slots.emplace_back(PyType_Slot{ Py_tp_getattro, (void*)ptr });
			return ret;
		}

		template<auto memFn>
		constexpr auto sqLen() const
		{
			constexpr auto fn = []() {
				if constexpr (detail::IsPObject<Ty>::value) return detail::compose<&Ty::value, memFn>;
				else return memFn;
			}();
			auto* ptr = py::CppWrapper<std::remove_const_t<decltype(fn)>>::template len<fn>();
			auto ret = TypeDefinition<Ty, BaseTy, defs...>{
				typeName, typeNameInModule, typeFlags,
				methodNames, propertyNames, slots
			};
			ret.slots.emplace_back(PyType_Slot{ Py_sq_length, (void*)ptr });
			return ret;
		}

		template<auto memFn>
		constexpr auto sqGetItem() const
		{
			constexpr auto fn = []() {
				if constexpr (detail::IsPObject<Ty>::value) return detail::compose<&Ty::value, memFn>;
				else return memFn;
			}();
			auto* ptr = py::CppWrapper<std::remove_const_t<decltype(fn)>>::template ssizearg<fn>();
			auto ret = TypeDefinition<Ty, BaseTy, defs...>{
				typeName, typeNameInModule, typeFlags,
				methodNames, propertyNames, slots
			};
			ret.slots.emplace_back(PyType_Slot{ Py_sq_item, (void*)ptr });
			return ret;
		}

		template<auto memFn>
		constexpr auto mpLen() const
		{
			constexpr auto fn = []() {
				if constexpr (detail::IsPObject<Ty>::value) return detail::compose<&Ty::value, memFn>;
				else return memFn;
			}();
			auto* ptr = py::CppWrapper<std::remove_const_t<decltype(fn)>>::template len<fn>();
			auto ret = TypeDefinition<Ty, BaseTy, defs...>{
				typeName, typeNameInModule, typeFlags,
				methodNames, propertyNames, slots
			};
			ret.slots.emplace_back(PyType_Slot{ Py_mp_length, (void*)ptr });
			return ret;
		}

		template<auto memFn>
		constexpr auto mpGetItem() const
		{
			constexpr auto fn = []() {
				if constexpr (detail::IsPObject<Ty>::value) return detail::compose<&Ty::value, memFn>;
				else return memFn;
			}();
			auto* ptr = py::CppWrapper<std::remove_const_t<decltype(fn)>>::template binary<fn>();
			auto ret = TypeDefinition<Ty, BaseTy, defs...>{
				typeName, typeNameInModule, typeFlags,
				methodNames, propertyNames, slots
			};
			ret.slots.emplace_back(PyType_Slot{ Py_mp_subscript, (void*)ptr });
			return ret;
		}

		template<auto memFn>
		constexpr auto repr() const
		{
			constexpr auto fn = []() {
				if constexpr (detail::IsPObject<Ty>::value) return detail::compose<&Ty::value, memFn>;
				else return memFn;
			}();
			auto* ptr = py::CppWrapper<std::remove_const_t<decltype(fn)>>::template repr<fn>();
			auto ret = TypeDefinition<Ty, BaseTy, defs...>{
				typeName, typeNameInModule, typeFlags,
				methodNames, propertyNames, slots
			};
			ret.slots.emplace_back(PyType_Slot{ Py_tp_repr, (void*)ptr });
			return ret;
		}

		template<auto memFn>
		constexpr auto iter() const
		{
			constexpr auto fn = []() {
				if constexpr (detail::IsPObject<Ty>::value) return detail::compose<&Ty::value, memFn>;
				else return memFn;
			}();
			auto* ptr = py::CppWrapper<std::remove_const_t<decltype(fn)>>::template repr<fn>();
			auto ret = TypeDefinition<Ty, BaseTy, defs...>{
				typeName, typeNameInModule, typeFlags,
				methodNames, propertyNames, slots
			};
			ret.slots.emplace_back(PyType_Slot{ Py_tp_iter, (void*)ptr });
			return ret;
		}

		template<auto memFn>
		constexpr auto iternext() const
		{
			constexpr auto fn = []() {
				if constexpr (detail::IsPObject<Ty>::value) return detail::compose<&Ty::value, memFn>;
				else return memFn;
			}();
			auto* ptr = py::CppWrapper<std::remove_const_t<decltype(fn)>>::template repr<fn>();
			auto ret = TypeDefinition<Ty, BaseTy, defs...>{
				typeName, typeNameInModule, typeFlags,
				methodNames, propertyNames, slots
			};
			ret.slots.emplace_back(PyType_Slot{ Py_tp_iternext, (void*)ptr });
			return ret;
		}

		template<auto memFn>
		constexpr auto call() const
		{
			constexpr auto fn = []() {
				if constexpr (detail::IsPObject<Ty>::value) return detail::compose<&Ty::value, memFn>;
				else return memFn;
			}();
			auto* ptr = py::CppWrapper<std::remove_const_t<decltype(fn)>>::template ternary<fn>();
			auto ret = TypeDefinition<Ty, BaseTy, defs...>{
				typeName, typeNameInModule, typeFlags,
				methodNames, propertyNames, slots
			};
			ret.slots.emplace_back(PyType_Slot{ Py_tp_call, (void*)ptr });
			return ret;
		}

		bool hasDefinition(int slot) const
		{
			for (const auto& s : slots)
			{
				if (s.slot == slot)
				{
					return true;
				}
			}
			return false;
		}
	};

	template<const auto& ... defs>
	class ModuleDefinition
	{
		friend class Module;

		using Methods = typename detail::FilterMethods<defs...>::Type;
		using Properties = typename detail::FilterProperties<defs...>::Type;

	public:
		std::vector<const char*> methodNames;
		std::vector<const char*> propertyNames;

		inline static auto methodDefs = detail::MethodDefBuilder<Methods>::get();
		inline static auto propertyDefs = detail::PropertyDefBuilder<Properties>::get();

		constexpr ModuleDefinition(
			const std::vector<const char*>& _methodNames = {},
			const std::vector<const char*>& _propertyNames = {}
		)
			: methodNames{ _methodNames },
			propertyNames{ _propertyNames }
		{
		}

		template<auto memFn>
		constexpr ModuleDefinition<defs..., methodDef<memFn, METH_VARARGS | METH_KEYWORDS>> method(const char* name) const
		{
			auto ret = ModuleDefinition<defs..., methodDef<memFn, METH_VARARGS | METH_KEYWORDS>>{
				methodNames, propertyNames
			};
			ret.methodNames.emplace_back(name);
			return ret;
		}
	};

	namespace detail
	{
		template<class Ty>
		struct IsTypeDefinition : std::false_type
		{
		};

		template<class Ty, class BaseTy, const auto& ... defs>
		struct IsTypeDefinition<TypeDefinition<Ty, BaseTy, defs...>> : std::true_type
		{
		};

		template<class Ty>
		struct IsModuleDefinition : std::false_type
		{
		};

		template<const auto& ... defs>
		struct IsModuleDefinition<ModuleDefinition<defs...>> : std::true_type
		{
		};
	}

	template<class Ty, class BaseTy = void>
	constexpr auto define(const char* name, const char* nameInModule, int flags = Py_TPFLAGS_DEFAULT)
	{
		return TypeDefinition<Ty, BaseTy>{ name, nameInModule, flags };
	}

	template<class Ty, class BaseTy = void>
	constexpr auto defineP(const char* name, const char* nameInModule, int flags = Py_TPFLAGS_DEFAULT)
	{
		return TypeDefinition<PObject<Ty>, std::conditional_t<std::is_same_v<BaseTy, void>, void, PObject<BaseTy>>>{ name, nameInModule, flags };
	}

	inline auto defineModule()
	{
		return ModuleDefinition{};
	}


	template<class Def>
	Module::Module(const char* name, const char* doc, Def&& d)
		: Module{ name, doc }
	{
		static_assert(detail::IsModuleDefinition<Def>::value, "Module constructor only accepts ModuleDefinition.");

		if (!d.methodNames.empty())
		{
			auto* defs = d.methodDefs.data();
			for (size_t i = 0; i < d.methodNames.size(); ++i)
			{
				defs[i].ml_name = (char*)d.methodNames[i];
			}
			def.m_methods = defs;
		}
	}


	template<class TypeDefR>
	bool Module::addType(TypeDefR&& def)
	{
		using TypeDef = std::remove_cv_t<std::remove_reference_t<TypeDefR>>;
		static_assert(detail::IsTypeDefinition<TypeDef>::value, "addType only accepts TypeDefinition.");
		using Ty = typename TypeDef::Class;
		using BaseTy = typename TypeDef::BaseClass;

		PyType_Spec spec;
		spec.name = def.typeName;
		spec.basicsize = sizeof(Ty);
		spec.itemsize = 0;
		spec.flags = def.typeFlags;
		spec.slots = nullptr;

		static PyMemberDef members[] = {
			{ "__dictoffset__", T_PYSSIZET, offsetof(Ty, managedDict), READONLY },
			{ "__weaklistoffset__", T_PYSSIZET, offsetof(Ty, managedWeakList), READONLY },
			{ nullptr },
		};

		std::vector<PyType_Slot> slots;
		slots.emplace_back(PyType_Slot{ Py_tp_dealloc, (void*)Ty::dealloc });
		slots.emplace_back(PyType_Slot{ Py_tp_new, (void*)Ty::_new });
		slots.emplace_back(PyType_Slot{ Py_tp_alloc, (void*)PyType_GenericAlloc });
		slots.emplace_back(PyType_Slot{ Py_tp_init, (void*)Ty::init });
		slots.emplace_back(PyType_Slot{ Py_tp_doc, (void*)"" });
		slots.emplace_back(PyType_Slot{ Py_tp_members, (void*)members });

		if constexpr (detail::HasIter<Ty>::value)
		{
			if (!def.hasDefinition(Py_tp_iter))
			{
				slots.emplace_back(PyType_Slot{ Py_tp_iter, (void*)CppWrapper<decltype(&Ty::iter)>::template repr<&Ty::iter>() });
			}
		}

		if constexpr (detail::HasIterNext<Ty>::value)
		{
			if (!def.hasDefinition(Py_tp_iternext))
			{
				slots.emplace_back(PyType_Slot{ Py_tp_iternext, (void*)CppWrapper<decltype(&Ty::iternext)>::template repr<&Ty::iternext>() });
			}
		}

		if constexpr (detail::HasRepr<Ty>::value)
		{
			if (!def.hasDefinition(Py_tp_repr))
			{
				slots.emplace_back(PyType_Slot{ Py_tp_repr, (void*)CppWrapper<decltype(&Ty::repr)>::template repr<&Ty::repr>() });
			}
		}

		if (!def.methodNames.empty())
		{
			auto* defs = def.methodDefs.data();
			for (size_t i = 0; i < def.methodNames.size(); ++i)
			{
				defs[i].ml_name = (char*)def.methodNames[i];
			}
			slots.emplace_back(PyType_Slot{ Py_tp_methods, (void*)defs });
		}

		if (def.propertyNames.size() > 0)
		{
			auto* defs = def.propertyDefs.data();
			for (size_t i = 0; i < def.propertyNames.size(); ++i)
			{
				defs[i].name = (char*)def.propertyNames[i];
			}
			slots.emplace_back(PyType_Slot{ Py_tp_getset, (void*)def.propertyDefs.data() });
		}

		slots.insert(slots.end(), def.slots.begin(), def.slots.end());
		slots.emplace_back(PyType_Slot{ 0, nullptr });
		spec.slots = slots.data();
		PyTypeObject* typeObj;
		if constexpr (std::is_same_v<BaseTy, void>)
		{
			typeObj = (PyTypeObject*)PyType_FromSpec(&spec);
		}
		else
		{
			PyTypeObject* baseTypeObj = TypeWrapper<BaseTy>::obj;
			if (!baseTypeObj)
			{
				std::cerr << "Base type " << def.typeName << " is not registered in module." << std::endl;
				return false;
			}
			typeObj = (PyTypeObject*)PyType_FromSpecWithBases(&spec, (PyObject*)baseTypeObj);
		}
		if (!typeObj) return false;

		TypeWrapper<Ty>::obj = typeObj;
		Py_INCREF(typeObj);
		PyModule_AddObject(mod, def.typeNameInModule, (PyObject*)typeObj);
		return true;
	}

	template<class Ty>
	Ty* checkType(PyObject* obj, const char* errorMsg = nullptr)
	{
		auto* ptr = py::TypeWrapper<Ty>::obj;
		if (!ptr)
		{
			throw TypeError{ "type not found in module" };
		}
		if (!PyObject_TypeCheck(obj, ptr))
		{
			if (errorMsg) throw TypeError{ errorMsg };
			else return nullptr;
		}
		return (Ty*)obj;
	}

	template<class Ty>
	py::UniqueCObj<Ty> checkType(py::UniqueObj obj, const char* errorMsg = nullptr)
	{
		auto* ptr = py::TypeWrapper<Ty>::obj;
		if (!ptr)
		{
			throw TypeError{ "type not found in module" };
		}
		if (!PyObject_TypeCheck(obj.get(), ptr))
		{
			if (errorMsg) throw TypeError{ errorMsg };
			else return {};
		}
		return py::UniqueCObj<Ty>{ (Ty*)obj.release() };
	}
}
