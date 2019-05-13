#pragma once
#include <vector>
#include <functional>
#include <typeinfo>
#include <algorithm>

namespace tomoto
{
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
		return CntUpdater<_Dec, _Ty>{}(val, inc);
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

	template<class _Container, class _OrderType = uint32_t>
	void sortAndWriteOrder(_Container& src, std::vector<_OrderType>& order)
	{
		std::vector<std::pair<typename _Container::value_type, _OrderType>> pv(src.size());
		for (_OrderType i = 0; i < src.size(); ++i)
		{
			pv[i] = std::make_pair(src[i], i);
		}

		std::sort(pv.begin(), pv.end());
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