#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <cstdio>

namespace tomoto
{
	namespace text
	{
		template<typename ... _Args>
		std::string format(const std::string& format, _Args ... args)
		{
			size_t size = snprintf(nullptr, 0, format.c_str(), args ...) + 1;
			std::vector<char> buf(size);
			snprintf(buf.data(), size, format.c_str(), args ...);
			return std::string{ buf.data(), buf.data() + size - 1 };
		}

		template<class _Iter, class _Target = decltype(*_Iter{}) >
		std::string join(_Iter first, _Iter last, const std::string& delimiter = ",")
		{
			if (first == last) return "";
			std::ostringstream stream;
			std::copy(first, last, std::ostream_iterator<_Target>(stream, delimiter.c_str()));
			std::string s = stream.str();
			s.erase(s.end() - delimiter.size(), s.end());
			return s;
		}
	}
}