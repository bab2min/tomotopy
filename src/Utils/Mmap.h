#pragma once
#include <string>
#include <iostream>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
namespace tomoto
{
	namespace utils
	{
		namespace detail
		{
			class HandleGuard
			{
				HANDLE handle = nullptr;
			public:
				HandleGuard(HANDLE _handle = nullptr) : handle(_handle)
				{
				}

				HandleGuard(const HandleGuard&) = delete;
				HandleGuard& operator =(const HandleGuard&) = delete;

				HandleGuard(HandleGuard&& o) noexcept
				{
					std::swap(handle, o.handle);
				}

				HandleGuard& operator=(HandleGuard&& o) noexcept
				{
					std::swap(handle, o.handle);
					return *this;
				}

				~HandleGuard()
				{
					if (handle && handle != INVALID_HANDLE_VALUE)
					{
						CloseHandle(handle);
						handle = nullptr;
					}
				}

				operator HANDLE() const
				{
					return handle;
				}
			};
		}

		class MMap
		{
			const char* view = nullptr;
			uint64_t len = 0;
			detail::HandleGuard hFile, hFileMap;
		public:
			MMap(const std::string& filepath);
			MMap(const MMap&) = delete;
			MMap& operator=(const MMap&) = delete;
			MMap(MMap&& o) noexcept;
			MMap& operator=(MMap&& o) noexcept;
			~MMap();

			const char* get() const { return view; }
			size_t size() const { return len; }
		};
	}
}
#else
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>

namespace tomoto
{
	namespace utils
	{
		namespace detail
		{
			class FDGuard
			{
				int fd = 0;
			public:
				FDGuard(int _fd = 0) : fd(_fd)
				{
				}

				FDGuard(const FDGuard&) = delete;
				FDGuard& operator =(const FDGuard&) = delete;

				FDGuard(FDGuard&& o)
				{
					std::swap(fd, o.fd);
				}

				FDGuard& operator=(FDGuard&& o)
				{
					std::swap(fd, o.fd);
					return *this;
				}

				~FDGuard()
				{
					if (fd && fd != -1)
					{
						close(fd);
						fd = 0;
					}
				}

				operator int() const
				{
					return fd;
				}
			};
		}

		class MMap
		{
			const char* view = nullptr;
			size_t len = 0;
			detail::FDGuard fd;
		public:
			MMap(const std::string& filepath);
			MMap(const MMap&) = delete;
			MMap& operator=(const MMap&) = delete;
			MMap(MMap&& o) noexcept;
			MMap& operator=(MMap&& o) noexcept;
			~MMap();

			const char* get() const { return view; }
			size_t size() const { return len; }
		};
	}
}
#endif
