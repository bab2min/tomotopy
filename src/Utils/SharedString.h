#pragma once

#include <string>
#include "serializer.hpp"

namespace tomoto
{
	class SharedString
	{
		const char* ptr = nullptr;
		size_t len = 0;

		void incref();

		void decref();

		void init(const char* _begin, const char* _end);

	public:

		SharedString();
		explicit SharedString(const char* _begin, const char* _end);
		explicit SharedString(const char* _ptr);
		explicit SharedString(const std::string& str);
		SharedString(const SharedString& o) noexcept;
		SharedString(SharedString&& o) noexcept;
		~SharedString();
		SharedString& operator=(const SharedString& o);
		SharedString& operator=(SharedString&& o) noexcept;

		size_t size() const
		{
			if (ptr) return len;
			return 0;
		}

		bool empty() const
		{
			return size() == 0;
		}

		operator std::string() const;

		const char* c_str() const;

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

		std::string substr(size_t start, size_t len) const;

		bool operator==(const SharedString& o) const;
		bool operator==(const std::string& o) const;

		bool operator!=(const SharedString& o) const;
		bool operator!=(const std::string& o) const;
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
