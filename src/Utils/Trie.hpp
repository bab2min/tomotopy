#pragma once
#include <map>
#include <unordered_map>
#include <deque>
#include <functional>
#include <iterator>

namespace tomoto
{
	template<class _Map>
	class ConstAccess : public _Map
	{
	public:
		auto operator[](typename _Map::key_type key) const -> typename _Map::mapped_type
		{
			auto it = this->find(key);
			if (it == this->end()) return {};
			else return it->second;
		}

		auto operator[](typename _Map::key_type key) -> typename _Map::mapped_type&
		{
			auto it = this->find(key);
			if (it == this->end()) return this->emplace(key, typename _Map::mapped_type{}).first->second;
			else return it->second;
		}
	};

	template<class _Key, class _Value, class _KeyStore = ConstAccess<std::map<_Key, int32_t>>, class _Trie = void>
	struct Trie
	{
		using Node = typename std::conditional<std::is_same<_Trie, void>::value, Trie, _Trie>::type;
		_KeyStore next = {};
		int32_t fail = 0;
		_Value val = {};

		Trie() {}
		~Trie() {}

		Node* getNext(_Key i) const
		{
			return next[i] ? (Node*)this + next[i] : nullptr;
		}

		Node* getFail() const
		{
			return fail ? (Node*)this + fail : nullptr;
		}

		template<typename _TyIter, typename _FnAlloc>
		void build(_TyIter first, _TyIter last, const _Value& _val, _FnAlloc&& alloc)
		{
			if (first == last)
			{
				if (!val) val = _val;
				return;
			}

			auto v = *first;
			if (!getNext(v))
			{
				next[v] = alloc() - this;
			}
			getNext(v)->build(++first, last, _val, alloc);
		}

		template<typename _TyIter>
		Node* findNode(_TyIter begin, _TyIter end)
		{
			if (begin == end) return this;
			auto n = getNext(*begin);
			if (n) return n->findNode(++begin, end);
			return nullptr;
		}

		template<class _Func>
		void traverse(_Func&& func)
		{
			if (val)
			{
				if (func(val)) return;
			}
			for (auto& p : next)
			{
				if (getNext(p.first))
				{
					getNext(p.first)->traverse(std::forward<_Func>(func));
				}
			}
			return;
		}

		template<typename _Fn>
		void traverse_with_keys(_Fn&& fn, std::vector<_Key>& rkeys)
		{
			fn((Node*)this, rkeys);

			for (auto& p : next)
			{
				if (p.first)
				{
					rkeys.emplace_back(p.first);
					getNext(p.first)->traverse_with_keys(fn, rkeys);
					rkeys.pop_back();
				}
			}
		}

		template<class _Iterator>
		std::pair<_Value*, size_t> findMaximumMatch(_Iterator begin, _Iterator end, size_t idxCnt = 0)
		{
			if (begin == end) return std::make_pair(val ? &val : nullptr, idxCnt);
			auto n = getNext(*begin);
			if (n)
			{
				auto v = n->findMaximumMatch(++begin, end, idxCnt + 1);
				if (v.first) return v;
			}
			return std::make_pair(val ? &val : nullptr, idxCnt);
		}

		Node* findFail(_Key i) const
		{
			if (!fail) // if this is Root
			{
				return (Node*)this;
			}
			else
			{
				if (getFail()->getNext(i)) // if 'i' node exists
				{
					return getFail()->getNext(i);
				}
				else // or loop for failure of this
				{
					return getFail()->findFail(i);
				}
			}
		}

		void fillFail()
		{
			std::deque<Node*> dq;
			for (dq.emplace_back(this); !dq.empty(); dq.pop_front())
			{
				auto p = dq.front();
				for (auto&& kv : p->next)
				{
					auto i = kv.first;
					if (!p->getNext(i)) continue;
					p->getNext(i)->fail = p->findFail(i) - p->getNext(i);
					dq.emplace_back(p->getNext(i));

					if (!p->val)
					{
						for (auto n = p; n->fail; n = n->getFail())
						{
							if (n->val)
							{
								p->val = (_Value)-1;
								break;
							}
						}
					}
				}
			}
		}
	};

	template<class _Key, class _Value, class _KeyStore = ConstAccess<std::map<_Key, int32_t>>>
	struct TrieEx : public Trie<_Key, _Value, _KeyStore, TrieEx<_Key, _Value, _KeyStore>>
	{
		int32_t parent = 0;

		template<typename _TyIter, typename _FnAlloc>
		void build(_TyIter first, _TyIter last, const _Value& _val, _FnAlloc&& alloc)
		{
			if (first == last)
			{
				if (!this->val) this->val = _val;
				return;
			}

			auto v = *first;
			if (!getNext(v))
			{
				this->next[v] = alloc() - this;
				this->getNext(v)->parent = -this->next[v];
			}
			this->getNext(v)->build(++first, last, _val, alloc);
		}

		template<typename _FnAlloc>
		TrieEx* makeNext(const _Key& k, _FnAlloc&& alloc)
		{
			if (!this->next[k])
			{
				this->next[k] = alloc() - this;
				this->getNext(k)->parent = -this->next[k];
				auto f = this->getFail();
				if (f)
				{
					f = f->makeNext(k, std::forward<_FnAlloc>(alloc));
					this->getNext(k)->fail = f - this->getNext(k);
				}
				else
				{
					this->getNext(k)->fail = this - this->getNext(k);
				}
			}
			return this + this->next[k];
		}

		TrieEx* getParent() const
		{
			if (!parent) return nullptr;
			return (TrieEx*)this + parent;
		}
	};
}