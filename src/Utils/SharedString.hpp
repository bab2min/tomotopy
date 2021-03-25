#pragma once

#include <string>
#include "serializer.hpp"

namespace tomoto
{
	class SharedString
	{
		const char* ptr = nullptr;
		size_t len = 0;

		void incref()
		{
			if (ptr)
			{
				++*(size_t*)ptr;
			}
		}

		void decref()
		{
			if (ptr)
			{
				if (--*(size_t*)ptr == 0)
				{
					delete[] ptr;
					ptr = nullptr;
				}
			}
		}

		void init(const char* _begin, const char* _end)
		{
			ptr = new char[_end - _begin + 9];
			*(size_t*)ptr = 1;
			len = _end - _begin;
			std::memcpy((void*)(ptr + 8), _begin, _end - _begin);
			((char*)ptr)[_end - _begin + 8] = 0;
		}

	public:

		SharedString()
		{
		}

		explicit SharedString(const char* _begin, const char* _end)
		{
			init(_begin, _end);
		}

		explicit SharedString(const char* _ptr)
		{
			if (_ptr)
			{
				init(_ptr, _ptr + std::strlen(_ptr));
			}
		}

		explicit SharedString(const std::string& str)
		{
			if (!str.empty())
			{
				init(str.data(), str.data() + str.size());
			}
		}

		SharedString(const SharedString& o) noexcept
			: ptr{ o.ptr }, len{ o.len }
		{
			incref();
		}

		SharedString(SharedString&& o) noexcept
		{
			std::swap(ptr, o.ptr);
			std::swap(len, o.len);
		}

		~SharedString()
		{
			decref();
		}

		SharedString& operator=(const SharedString& o)
		{
			if (this != &o)
			{
				decref();
				ptr = o.ptr;
				len = o.len;
				incref();
			}
			return *this;
		}

		SharedString& operator=(SharedString&& o) noexcept
		{
			std::swap(ptr, o.ptr);
			std::swap(len, o.len);
			return *this;
		}

		size_t size() const
		{
			if (ptr) return len;
			return 0;
		}

		bool empty() const
		{
			return ptr == nullptr || size() == 0;
		}

		operator std::string() const
		{
			if (!ptr) return {};
			return { ptr + 8, ptr + 8 + len };
		}

		const char* c_str() const
		{
			if (!ptr) return "";
			return ptr + 8;
		}

		const char* data() const
		{
			return c_str();
		}

		const char* begin() const
		{
			return data();
		}

		const char* end() const
		{
			return data() + size();
		}

		std::string substr(size_t start, size_t len) const
		{
			return { c_str() + start, c_str() + start + len };
		}

		bool operator==(const SharedString& o) const
		{
			if (ptr == o.ptr) return true;
			if (size() != o.size()) return false;
			return std::equal(begin(), end(), o.begin());
		}

		bool operator==(const std::string& o) const
		{
			if (size() != o.size()) return false;
			return std::equal(begin(), end(), o.begin());
		}

		bool operator!=(const SharedString& o) const
		{
			return !operator==(o);
		}

		bool operator!=(const std::string& o) const
		{
			return !operator==(o);
		}
	};

	namespace serializer
	{
		template<>
		struct Serializer<SharedString>
		{
			using VTy = SharedString;
			void write(std::ostream& ostr, const VTy& v)
			{
				writeToStream(ostr, (uint32_t)v.size());
				if (!ostr.write((const char*)v.data(), v.size()))
					throw std::ios_base::failure(std::string("writing type 'SharedString' is failed"));
			}

			void read(std::istream& istr, VTy& v)
			{
				auto size = readFromStream<uint32_t>(istr);
				std::vector<char> t(size);
				if (!istr.read((char*)t.data(), t.size()))
					throw std::ios_base::failure(std::string("reading type 'SharedString' is failed"));
				v = SharedString{ t.data(), t.data() + t.size() };
			}
		};
	}
}

namespace std
{
	template <> struct hash<tomoto::SharedString>
	{
		size_t operator()(const tomoto::SharedString& x) const
		{
			return hash<string>{}(x);
		}
	};
}
