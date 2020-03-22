#pragma once
#include <cassert>
#include <vector>
#include <functional>
#include <typeinfo>
#include <algorithm>
#include <memory>
#include <mutex>

namespace tomoto
{
	template<typename T, typename... Args, 
		typename std::enable_if<!std::is_array<T>::value, int>::type = 0>
	std::unique_ptr<T> make_unique(Args&&... args)
	{
		return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
	}

	template<typename T,
		typename std::enable_if<std::is_array<T>::value, int>::type = 0>
	std::unique_ptr<T> make_unique(size_t size)
	{
		return std::unique_ptr<T>(new typename std::remove_extent<T>::type [size]);
	}

	template<typename T>
	constexpr T * as_mutable(const T * value) noexcept {
		return const_cast<T *>(value);
	}

	template<bool _lock>
	class OptionalLock : public std::lock_guard<std::mutex>
	{
	public:
		using std::lock_guard<std::mutex>::lock_guard;
	};

	template<>
	class OptionalLock<false>
	{
	public:
		OptionalLock(const std::mutex& mtx)
		{}
	};

	template<bool _Dec, typename _Ty>
	struct CntUpdater
	{
		inline _Ty operator()(_Ty& val, _Ty inc)
		{
			return val += inc;
		}
	};

	template<typename _Ty>
	struct CntUpdater<true, _Ty>
	{
		inline _Ty operator()(_Ty& val, _Ty inc)
		{
			return val = std::max(val + inc, (_Ty)0);
		}
	};

	template<bool _Dec, typename _Ty> _Ty updateCnt(_Ty& val, _Ty inc)
	{
		auto ret = CntUpdater<_Dec, _Ty>{}(val, inc);
		assert(ret >= 0);
		return ret;
	}

	template<class UnaryFunction>
	UnaryFunction forRandom(size_t N, size_t seed, UnaryFunction f)
	{
		static size_t primes[16] = {
			65537, 65539, 65543, 65551, 65557, 65563,
			65579, 65581, 65587, 65599, 65609, 65617,
			65629, 65633, 65647, 65651
		};
		if (!N) return f;
		size_t P = primes[seed & 0xF];
		if (N % P == 0)
		{
			P = primes[(seed + 1) & 0xF];
			if (N % P == 0) P = primes[(seed + 2) & 0xF];
			if (N % P == 0) P = primes[(seed + 3) & 0xF];
		}
		P %= N;
		for (size_t i = 0; i < N; ++i) {
			f(((i + seed) * P) % N);
		}
		return f;
	}

	template<class RandomIt, class UnaryFunction>
	UnaryFunction forEachRandom(RandomIt first, RandomIt last, size_t seed, UnaryFunction f)
	{
		static size_t primes[16] = {
			65537, 65539, 65543, 65551, 65557, 65563,
			65579, 65581, 65587, 65599, 65609, 65617,
			65629, 65633, 65647, 65651
		};

		const size_t N = std::distance(first, last);
		if (!N) return f;
		size_t P = primes[seed & 0xF];
		if (N % P == 0)
		{
			P = primes[(seed + 1) & 0xF];
			if (N % P == 0) P = primes[(seed + 2) & 0xF];
			if (N % P == 0) P = primes[(seed + 3) & 0xF];
		}
		P %= N;
		for (size_t i = 0; i < N; ++i) {
			f(first[((i + seed) * P) % N]);
		}
		return f;
	}

	template<class _Cont, class _Ty>
	size_t insertIntoEmpty(_Cont& cont, _Ty&& e)
	{
		size_t pos = 0;
		for (auto& c : cont)
		{
			if (!(bool)c)
			{
				c = e;
				return pos;
			}
			++pos;
		}
		cont.emplace_back(e);
		return pos;
	}

	/*
	* _Container src: (in/out) container to be sorted
	* vector<integer> order: (out) a vector mapping old idx to new idx (order[oldIdx] => newIdx)
	* _Less cmp: (in) comparator
	*/
	template<typename _Container, typename _OrderType = uint32_t, typename _Less = std::less<typename _Container::value_type>>
	void sortAndWriteOrder(_Container& src, std::vector<_OrderType>& order, size_t rotate = 0, _Less cmp = _Less{})
	{
		typedef std::pair<typename _Container::value_type, _OrderType> voPair_t;
		std::vector<voPair_t> pv(src.size());
		for (_OrderType i = 0; i < src.size(); ++i)
		{
			pv[i] = std::make_pair(src[i], i);
		}

		std::sort(pv.begin(), pv.end(), [&cmp](const voPair_t& a, const voPair_t& b)
		{
			return cmp(a.first, b.first);
		});
		if (rotate) std::rotate(pv.begin(), pv.begin() + rotate, pv.end());
		order = std::vector<_OrderType>(src.size());
		for (size_t i = 0; i < src.size(); ++i)
		{
			src[i] = pv[i].first;
			order[pv[i].second] = i;
		}
	}

	template <typename _BaseIter>
	struct FilteredIter : _BaseIter
	{
		using filterTy = std::function<bool(const typename std::iterator_traits<_BaseIter>::value_type&)>;

		FilteredIter() = default;
		FilteredIter(filterTy filter, _BaseIter base, _BaseIter end = {})
			: _BaseIter(base), _filter(filter), _end(end)
		{
			while (*this != _end && !_filter(**this)) {
				++*this;
			}
		}

		FilteredIter& operator++()
		{
			do
			{
				_BaseIter::operator++();
			} while (*this != _end && !_filter(**this));
			return *this;
		}

		FilteredIter operator++(int)
		{
			FilteredIter copy = *this;
			++*this;
			return copy;
		}

	private:
		filterTy _filter;
		_BaseIter _end;
	};

	template <typename _BaseIter>
	FilteredIter<_BaseIter> makeFilteredIter(
		typename FilteredIter<_BaseIter>::filterTy filter,
		_BaseIter base, _BaseIter end = {})
	{
		return { filter, base, end };
	}

	template <typename _UnaryFunc, typename _Iterator>
	class TransformIter : public _Iterator
	{
	private:
		_UnaryFunc f;
	public:
		using reference = typename std::result_of<const _UnaryFunc(typename std::iterator_traits<_Iterator>::reference)>::type;
		TransformIter(const _Iterator& _iter = {}, _UnaryFunc _f = {})
			: _Iterator(_iter), f(_f)
		{}
		
		reference operator*()
		{
			return f(_Iterator::operator*());
		}

		const reference operator*() const
		{
			return f(_Iterator::operator*());
		}

		reference operator[](std::size_t idx)
		{
			return f(_Iterator::operator[](idx));
		}

		const reference operator[](std::size_t idx) const
		{
			return f(_Iterator::operator[](idx));
		}

		TransformIter& operator++()
		{
			_Iterator::operator++();
			return *this;
		}
	};

	template <typename _UnaryFunc, typename _Iterator> 
	TransformIter<_UnaryFunc, _Iterator> makeTransformIter(const _Iterator& iter, _UnaryFunc f)
	{
		return { iter, f };
	}
}
