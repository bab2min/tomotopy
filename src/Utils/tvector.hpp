#pragma once
#include <algorithm>
#include <cstring>
#include "serializer.hpp"

namespace tomoto
{
	/**
	tvector : std::vector-like container which can handle non-owning std::array as well as traditional std::vector function
	*/
	template <typename T, class _Alloc = std::allocator<T>>
	class tvector 
	{
	public:
		typedef T                                     value_type;
		typedef T &                                   reference;
		typedef const T &                             const_reference;
		typedef T *                                   pointer;
		typedef const T *                             const_pointer;
		typedef T *                                   iterator;
		typedef const T *                             const_iterator;
		typedef std::reverse_iterator<iterator>       reverse_iterator;
		typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
		typedef std::ptrdiff_t                             difference_type;
		typedef std::size_t                          size_type;

		tvector() noexcept
		{
		}

		tvector(std::nullptr_t) noexcept
		{
		}

		// non-owning, just pointing constructor
		tvector(pointer ptr, size_type size) noexcept
		{
			_first = ptr;
			_last = ptr + size;
			_rsvEnd = nullptr;
		}

		explicit tvector(size_type n, const T &val = T())
		{
			_first = _Alloc{}.allocate(n);
			for (size_type i = 0; i < n; ++i) _first[i] = val;
			_last = _first + n;
			_rsvEnd = _last;
		}

		tvector(iterator first, iterator last)
		{
			size_type count = last - first;
			_first = _Alloc{}.allocate(count);
			for (size_type i = 0; i < count; ++i, ++first)
				_first[i] = *first;
			_last = _first + count;
			_rsvEnd = _last;
		}

		tvector(std::initializer_list<T> lst)
		{
			_first = _Alloc{}.allocate(lst.size());
			_last = _first;
			for (auto &item : lst) *_last++ = item;
			_rsvEnd = _last;
		}

		tvector(const tvector<T> & other)
		{
			if (other._rsvEnd)
			{
				_first = _Alloc{}.allocate(other.capacity());
				_last = _first;
				for (size_type i = 0; i < other.size(); ++i) *_last++ = other._first[i];
				_rsvEnd = _first + other.capacity();
			}
			else
			{
				if (_rsvEnd) _Alloc{}.deallocate(_first, capacity());
				_first = other._first;
				_last = other._last;
				_rsvEnd = nullptr;
			}
		}

		tvector(tvector<T> && other) noexcept
		{
			swap(other);
		}

		~tvector()
		{
			if (_first && _rsvEnd) _Alloc{}.deallocate(_first, capacity());
		}

		tvector<T> & operator = (const tvector<T> &other)
		{
			if (other._rsvEnd)
			{
				buy(other.size());
				for (size_type i = 0; i < other.size(); ++i)
					_first[i] = other._first[i];
				_last = _first + other.size();
			}
			else
			{
				if (_rsvEnd) _Alloc{}.deallocate(_first, capacity());
				_first = other._first;
				_last = other._last;
				_rsvEnd = nullptr;
			}
			return *this;
		}

		tvector<T> & operator = (tvector<T> &&other)
		{
			swap(other);
			return *this;
		}

		tvector<T> & operator = (std::initializer_list<T> lst)
		{
			buy(lst.size());
			_last = _first;
			for (auto &item : lst)
				*_last++ = item;
			return *this;
		}

		void assign(size_type count, const T &value)
		{
			size_type i;
			buy(count);
			for (i = 0; i < count; ++i)
				_first[i] = value;
			_last = _first + count;
		}

		void assign(iterator first, iterator last)
		{
			size_type count = last - first;
			buy(count);
			for (size_type i = 0; i < count; ++i, ++first)
				_first[i] = *first;
			_last = _first + count;
		}

		void assign(std::initializer_list<T> lst)
		{
			size_type count = lst.size();
			buy(count);
			_last = _first;
			for (auto &item : lst)
				*_last++ = item;
		}

		// iterators:
		iterator begin() noexcept {	return _first; }

		const_iterator begin() const noexcept { return _first; }

		const_iterator cbegin() const noexcept { return _first; }

		iterator end() noexcept { return _last; }

		const_iterator end() const noexcept { return _last; }

		const_iterator cend() const noexcept { return _last; }

		reverse_iterator rbegin() noexcept { return reverse_iterator(_last); }

		const_reverse_iterator rbegin() const noexcept { return reverse_iterator(_last); }

		const_reverse_iterator crbegin() const noexcept { return reverse_iterator(_last); }

		reverse_iterator rend() noexcept { return reverse_iterator(_first); }

		const_reverse_iterator rend() const noexcept { return reverse_iterator(_first); }

		const_reverse_iterator crend() const noexcept { return reverse_iterator(_first); }

		// 23.3.11.3, capacity:
		bool empty() const noexcept 
		{
			return _last == _first;
		}

		size_type size() const noexcept 
		{
			return _last - _first;
		}


		size_type max_size() const noexcept 
		{
			return 0xFFFFFFFF;
		}


		size_type capacity() const noexcept 
		{
			return _rsvEnd - _first;
		}

		void resize(size_type sz, const T &c = T()) 
		{
			if (sz > size())
			{
				if (!isOwner()) throw std::out_of_range("cannot increase size of non-owning mode");
				reallocate(sz);
				for (size_type i = size(); i < sz; ++i) _first[i] = c;
			}
			else
			{
				for (size_type i = size(); i < sz; ++i) _first[i].~T();
			}
			_last = _first + sz;
		}

		void reserve(size_type _sz) 
		{
			if (_sz > capacity()) {
				reallocate(_sz);
			}
		}

		void shrink_to_fit() 
		{
			reallocate(size());
		}

		// element access
		reference operator [](size_type idx) 
		{
			return _first[idx];
		}

		const_reference operator [](size_type idx) const 
		{
			return _first[idx];
		}

		reference at(size_type pos) 
		{
			if (pos < size())
				return _first[pos];
			else
				throw std::out_of_range("accessed position is out of range");
		}

		const_reference at(size_type pos) const
		{
			if (pos < size())
				return _first[pos];
			else
				throw std::out_of_range("accessed position is out of range");
		}

		reference front() 
		{
			return _first[0];
		}

		const_reference front() const 
		{
			return _first[0];
		}

		reference back() 
		{
			return _last[-1];
		}

		const_reference back() const 
		{
			return _last[-1];
		}

		// 23.3.11.4, data access:
		T * data() noexcept 
		{
			return _first;
		}

		const T * data() const noexcept 
		{
			return _first;
		}

		bool isOwner() const noexcept
		{
			return _rsvEnd || (_rsvEnd == nullptr && _first == nullptr);
		}

		// 23.3.11.5, modifiers:
		template <class ... Args>
		void emplace_back(Args && ... args) 
		{
			buy(size() + 1);
			*_last++ = std::move(T(std::forward<Args>(args) ...));
		}

		void push_back(const T &val) 
		{
			buy(size() + 1);
			*_last++ = val;
		}

		void push_back(T &&val) 
		{
			buy(size() + 1);
			*_last++ = std::move(val);
		}

		void pop_back() 
		{
			(_last--)[-1].~T();
		}

		template <class ... Args>
		iterator emplace(const_iterator it, Args && ... args) 
		{
			iterator iit = &_first[it - _first];
			buy(size() + 1);
			memmove(iit + 1, iit, (size() - (it - _first)) * sizeof(T));
			(*iit) = std::move(T(std::forward<Args>(args) ...));
			_last++;
			return iit;
		}

		iterator insert(const_iterator it, const T &val) 
		{
			iterator iit = &_first[it - _first];
			buy(size() + 1);
			memmove(iit + 1, iit, (size() - (it - _first)) * sizeof(T));
			(*iit) = val;
			_last++;
			return iit;
		}


		iterator insert(const_iterator it, T &&val) 
		{
			iterator iit = &_first[it - _first];
			buy(size() + 1);
			memmove(iit + 1, iit, (size() - (it - _first)) * sizeof(T));
			(*iit) = std::move(val);
			_last++;
			return iit;
		}


		iterator insert(const_iterator it, size_type cnt, const T &val) 
		{
			iterator f = &_first[it - _first];
			if (!cnt) return f;
			buy(size() + cnt);
			memmove(f + cnt, f, (size() - (it - _first)) * sizeof(T));
			_last += cnt;
			for (iterator it = f; cnt--; ++it)
				(*it) = val;
			return f;
		}


		template <class InputIt>
		iterator insert(const_iterator it, InputIt first, InputIt last) 
		{
			iterator f = &_first[it - _first];
			size_type cnt = last - first;
			if (!cnt) return f;
			buy(size() + cnt);
			memmove(f + cnt, f, (size() - (it - _first)) * sizeof(T));
			for (iterator it = f; first != last; ++it, ++first)
				(*it) = *first;
			_last += cnt;
			return f;
		}

		iterator insert(const_iterator it, std::initializer_list<T> lst) 
		{
			size_type cnt = lst.size();
			iterator f = &_first[it - _first];
			if (!cnt) return f;
			buy(size() + cnt);
			memmove(f + cnt, f, (size() - (it - _first)) * sizeof(T));
			iterator iit = f;
			for (auto &item : lst) {
				(*iit) = item;
				++iit;
			}
			_last += cnt;
			return f;
		}

		iterator erase(const_iterator it) 
		{
			iterator iit = &_first[it - _first];
			(*iit).~T();
			memmove(iit, iit + 1, (size() - (it - _first) - 1) * sizeof(T));
			--_last;
			return iit;
		}

		iterator erase(const_iterator first, const_iterator last) 
		{
			iterator f = &_first[first - _first];
			if (first == last) return f;
			for (; first != last; ++first)
				(*first).~T();
			memmove(f, last, (size() - (last - _first)) * sizeof(T));
			_last -= last - first;
			return f;
		}

		void swap(tvector<T> &rhs) 
		{
			std::swap(_first, rhs._first);
			std::swap(_last, rhs._last);
			std::swap(_rsvEnd, rhs._rsvEnd);
		}

		void clear() noexcept 
		{
			for (size_type i = 0; i < size(); ++i) _first[i].~T();
			_last = _first;
		}

		bool operator == (const tvector<T> &rhs) const 
		{
			if (size() != rhs.size()) return false;
			for (size_type i = 0; i < size(); ++i)
				if (_first[i] != rhs._first[i])
					return false;
			return true;
		}


		bool operator != (const tvector<T> &rhs) const 
		{
			return !operator==(rhs);
		}


		bool operator < (const tvector<T> &rhs) const 
		{
			size_type i, ub = std::min(size(), rhs.size());
			for (i = 0; i < ub; ++i)
				if (_first[i] != rhs._first[i])
					return _first[i] < rhs._first[i];
			return size() < rhs.size();
		}


		bool operator <= (const tvector<T> &rhs) const 
		{
			size_type i, ub = std::min(size(), rhs.size());
			for (i = 0; i < ub; ++i)
				if (_first[i] != rhs._first[i])
					return _first[i] < rhs._first[i];
			return size() < rhs.size();
		}


		bool operator > (const tvector<T> &rhs) const 
		{
			size_type i, ub = std::min(size(), rhs.size());
			for (i = 0; i < ub; ++i)
				if (_first[i] != rhs._first[i])
					return _first[i] > rhs._first[i];
			return size() > rhs.size();
		}


		bool operator >= (const tvector<T> &rhs) const 
		{
			size_type i, ub = std::min(size(), rhs.size());
			for (i = 0; i < ub; ++i)
				if (_first[i] != rhs._first[i])
					return _first[i] > rhs._first[i];
			return size() >= rhs.size();
		}

		template <class _Cont, class _Iter>
		static void trade(_Cont& dest, _Iter srcBegin, _Iter srcEnd)
		{
			/*static_assert(std::is_same<typename std::remove_reference<std::iterator_traits<_Iter>::value_type>::type, tvector<T>*>::value, 
				"value_type of InputIt must be tvector<T>* type");*/
			size_type totalLen = 0;
			for (auto it = srcBegin; it != srcEnd; ++it) totalLen += (*it)->size();
			auto dend = dest.size();
			dest.resize(dest.size() + totalLen);
			T* dp = dest.data() + dend;
			for (auto it = srcBegin; it != srcEnd; ++it)
			{
				auto& tv = **it;
				std::copy(tv.begin(), tv.end(), dp);
				tv = tvector<T>{ dp, tv.size() };
				dp += tv.size();
			}
		}

	private:
		pointer _first = nullptr; // ptr for first data
		pointer _last = nullptr; // ptr last data
		pointer _rsvEnd = nullptr; // ptr reserved last data

		inline void buy(size_type needSize)
		{
			if (!isOwner()) throw std::out_of_range("cannot increase size of non-owning mode");
			if (needSize > capacity())
			{
				reallocate(calcGrowth(needSize));
			}
		}

		inline void reallocate(size_type newSize)
		{
			size_type s = size();
			T *tarr = _Alloc{}.allocate(newSize);
			if (_first)
			{
				memcpy(tarr, _first, s * sizeof(T));
				_Alloc{}.deallocate(_first, capacity());
			}
			_first = tarr;
			_last = _first + s;
			_rsvEnd = _first + newSize;
		}

		inline size_type calcGrowth(size_type newSize)
		{
			const size_type oldcapacity = capacity();
			if (oldcapacity > max_size() - oldcapacity / 2)
			{
				return newSize;
			}

			const size_type geometric = oldcapacity + oldcapacity / 2;
			if (geometric < newSize)
			{
				return newSize;
			}
			return geometric;
		}
	};

	namespace serializer
	{
		template<typename _Ty>
		struct Serializer<tvector<_Ty>, typename std::enable_if<std::is_fundamental<_Ty>::value>::type>
		{
			using VTy = tvector<_Ty>;
			void write(std::ostream& ostr, const VTy& v)
			{
				writeToStream(ostr, (uint32_t)v.size());
				if (!ostr.write((const char*)v.data(), sizeof(_Ty) * v.size()))
					throw std::ios_base::failure(std::string("writing type '") + typeid(_Ty).name() + std::string("' is failed"));
			}

			void read(std::istream& istr, VTy& v)
			{
				auto size = readFromStream<uint32_t>(istr);
				v.resize(size);
				if (!istr.read((char*)v.data(), sizeof(_Ty) * size))
					throw std::ios_base::failure(std::string("reading type '") + typeid(_Ty).name() + std::string("' is failed"));
			}
		};

		template<typename _Ty>
		struct Serializer<tvector<_Ty>, typename std::enable_if<!std::is_fundamental<_Ty>::value>::type>
		{
			using VTy = tvector<_Ty>;
			void write(std::ostream& ostr, const VTy& v)
			{
				writeToStream(ostr, (uint32_t)v.size());
				for (auto& e : v) Serializer<_Ty>{}.write(ostr, e);
			}

			void read(std::istream& istr, VTy& v)
			{
				auto size = readFromStream<uint32_t>(istr);
				v.resize(size);
				for (auto& e : v) Serializer<_Ty>{}.read(istr, e);
			}
		};
	}
}