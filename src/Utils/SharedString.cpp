#include "SharedString.h"

namespace tomoto
{
    void SharedString::incref()
    {
        if (ptr)
        {
            ++*(size_t*)ptr;
        }
    }

    void SharedString::decref()
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

    void SharedString::init(const char* _begin, const char* _end)
    {
        ptr = new char[_end - _begin + 9];
        *(size_t*)ptr = 1;
        len = _end - _begin;
        std::memcpy((void*)(ptr + 8), _begin, _end - _begin);
        ((char*)ptr)[_end - _begin + 8] = 0;
    }

    SharedString::SharedString()
    {
    }

    SharedString::SharedString(const char* _begin, const char* _end)
    {
        init(_begin, _end);
    }

    SharedString::SharedString(const char* _ptr)
    {
        if (_ptr)
        {
            init(_ptr, _ptr + std::strlen(_ptr));
        }
    }

    SharedString::SharedString(const std::string& str)
    {
        if (!str.empty())
        {
            init(str.data(), str.data() + str.size());
        }
    }

    SharedString::SharedString(const SharedString& o) noexcept
        : ptr{ o.ptr }, len{ o.len }
    {
        incref();
    }

    SharedString::SharedString(SharedString&& o) noexcept
    {
        std::swap(ptr, o.ptr);
        std::swap(len, o.len);
    }

    SharedString::~SharedString()
    {
        decref();
    }

    SharedString& SharedString::operator=(const SharedString& o)
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

    SharedString& SharedString::operator=(SharedString&& o) noexcept
    {
        std::swap(ptr, o.ptr);
        std::swap(len, o.len);
        return *this;
    }

    SharedString::operator std::string() const
    {
        if (!ptr) return {};
        return { ptr + 8, ptr + 8 + len };
    }

    const char* SharedString::c_str() const
    {
        if (!ptr) return "";
        return ptr + 8;
    }

    std::string SharedString::substr(size_t start, size_t len) const
    {
        return { c_str() + start, c_str() + start + len };
    }

    bool SharedString::operator==(const SharedString& o) const
    {
        if (ptr == o.ptr) return true;
        if (size() != o.size()) return false;
        return std::equal(begin(), end(), o.begin());
    }

    bool SharedString::operator==(const std::string& o) const
    {
        if (size() != o.size()) return false;
        return std::equal(begin(), end(), o.begin());
    }

    bool SharedString::operator!=(const SharedString& o) const
    {
        return !operator==(o);
    }

    bool SharedString::operator!=(const std::string& o) const
    {
        return !operator==(o);
    }
}
