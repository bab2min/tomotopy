#include <cstdint>
#include "Mmap.h"

namespace tomoto
{
	namespace utils
	{
		static std::u16string utf8To16(const std::string& str)
		{
			std::u16string ret;
			for (auto it = str.begin(); it != str.end(); ++it)
			{
				uint32_t code = 0;
				uint32_t byte = (uint8_t)*it;
				if ((byte & 0xF8) == 0xF0)
				{
					code = (uint32_t)((byte & 0x07) << 18);
					if (++it == str.end()) throw std::invalid_argument{ "unexpected ending" };
					if (((byte = *it) & 0xC0) != 0x80) throw std::invalid_argument{ "unexpected trailing byte" };
					code |= (uint32_t)((byte & 0x3F) << 12);
					if (++it == str.end()) throw std::invalid_argument{ "unexpected ending" };
					if (((byte = *it) & 0xC0) != 0x80) throw std::invalid_argument{ "unexpected trailing byte" };
					code |= (uint32_t)((byte & 0x3F) << 6);
					if (++it == str.end()) throw std::invalid_argument{ "unexpected ending" };
					if (((byte = *it) & 0xC0) != 0x80) throw std::invalid_argument{ "unexpected trailing byte" };
					code |= (byte & 0x3F);
				}
				else if ((byte & 0xF0) == 0xE0)
				{
					code = (uint32_t)((byte & 0x0F) << 12);
					if (++it == str.end()) throw std::invalid_argument{ "unexpected ending" };
					if (((byte = *it) & 0xC0) != 0x80) throw std::invalid_argument{ "unexpected trailing byte" };
					code |= (uint32_t)((byte & 0x3F) << 6);
					if (++it == str.end()) throw std::invalid_argument{ "unexpected ending" };
					if (((byte = *it) & 0xC0) != 0x80) throw std::invalid_argument{ "unexpected trailing byte" };
					code |= (byte & 0x3F);
				}
				else if ((byte & 0xE0) == 0xC0)
				{
					code = (uint32_t)((byte & 0x1F) << 6);
					if (++it == str.end()) throw std::invalid_argument{ "unexpected ending" };
					if (((byte = *it) & 0xC0) != 0x80) throw std::invalid_argument{ "unexpected trailing byte" };
					code |= (byte & 0x3F);
				}
				else if ((byte & 0x80) == 0x00)
				{
					code = byte;
				}
				else
				{
					throw std::invalid_argument{ "unicode error" };
				}

				if (code < 0x10000)
				{
					ret.push_back((char16_t)code);
				}
				else if (code < 0x10FFFF)
				{
					code -= 0x10000;
					ret.push_back((char16_t)(0xD800 | (code >> 10)));
					ret.push_back((char16_t)(0xDC00 | (code & 0x3FF)));
				}
				else
				{
					throw std::invalid_argument{ "unicode error" };
				}
			}
			return ret;
		}
	}
}

namespace tomoto
{
	namespace utils
	{
		MMap::MMap(const std::string& filepath)
		{
#ifdef _WIN32
			hFile = CreateFileW((const wchar_t*)utf8To16(filepath).c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_READONLY, nullptr);
			if (hFile == INVALID_HANDLE_VALUE) throw std::ios_base::failure("Cannot open '" + filepath + "'");
			hFileMap = CreateFileMapping(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
			if (hFileMap == nullptr) throw std::ios_base::failure("Cannot open '" + filepath + "' Code:" + std::to_string(GetLastError()));
			view = (const char*)MapViewOfFile(hFileMap, FILE_MAP_READ, 0, 0, 0);
			if (!view) throw std::ios_base::failure("Cannot MapViewOfFile() Code:" + std::to_string(GetLastError()));
			DWORD high;
			len = GetFileSize(hFile, &high);
			len |= (uint64_t)high << 32;
#else 
			fd = open(filepath.c_str(), O_RDONLY);
			if (fd == -1) throw std::ios_base::failure("Cannot open '" + filepath + "'");
			struct stat sb;
			if (fstat(fd, &sb) < 0) throw std::ios_base::failure("Cannot open '" + filepath + "'");
			len = sb.st_size;
			view = (const char*)mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
			if (view == MAP_FAILED) throw std::ios_base::failure("Mapping failed");
#endif
		}

#ifdef _WIN32
		MMap::MMap(MMap&& o) noexcept
			: view{ o.view }, len{ o.len }
		{
			o.view = nullptr;
			std::swap(hFile, o.hFile);
			std::swap(hFileMap, o.hFileMap);
		}
#else 
		MMap::MMap(MMap&& o) noexcept 
			: len{ o.len }, fd{ std::move(o.fd) }
		{
			std::swap(view, o.view);
		}
#endif

		MMap& MMap::operator=(MMap&& o) noexcept
		{
			std::swap(view, o.view);
			std::swap(len, o.len);
#ifdef _WIN32
			std::swap(hFile, o.hFile);
			std::swap(hFileMap, o.hFileMap);
#else 
			std::swap(fd, o.fd);
#endif
			return *this;
		}

		MMap::~MMap()
		{
#ifdef _WIN32
			if (hFileMap)
			{
				UnmapViewOfFile(view);
				view = nullptr;
			}
#else 
			if (view)
			{
				munmap((void*)view, len);
			}
#endif
		}
	}
}
