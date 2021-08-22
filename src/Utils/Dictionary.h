#pragma once
#include <random>
#include <exception>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <sstream>
#include <cassert>
#include "serializer.hpp"

namespace tomoto
{
	using Vid = uint32_t;
	static constexpr Vid non_vocab_id = (Vid)-1;
	using Tid = uint16_t;
	static constexpr Vid non_topic_id = (Tid)-1;
	using Float = float;

	struct VidPair : public std::pair<Vid, Vid>
	{
		using std::pair<Vid, Vid>::pair;
	};

	class Dictionary
	{
	protected:
		std::unordered_map<std::string, Vid> dict;
		std::vector<std::string> id2word;
	public:
		Vid add(const std::string& word)
		{
			auto it = dict.find(word);
			if (it == dict.end())
			{
				dict.emplace(word, (Vid)dict.size());
				id2word.emplace_back(word);
				return (Vid)(dict.size() - 1);
			}
			return it->second;
		}

		size_t size() const { return dict.size(); }
		
		const std::string& toWord(Vid vid) const
		{
			assert(vid < id2word.size());
			return id2word[vid];
		}
		
		Vid toWid(const std::string& word) const
		{
			auto it = dict.find(word);
			if (it == dict.end()) return non_vocab_id;
			return it->second;
		}

		void serializerWrite(std::ostream& writer) const
		{
			serializer::writeMany(writer, serializer::to_key("Dict"), id2word);
		}
		
		void serializerRead(std::istream& reader)
		{
			serializer::readMany(reader, serializer::to_key("Dict"), id2word);
			for (size_t i = 0; i < id2word.size(); ++i)
			{
				dict.emplace(id2word[i], (Vid)i);
			}
		}

		void swap(Dictionary& rhs)
		{
			std::swap(dict, rhs.dict);
			std::swap(id2word, rhs.id2word);
		}

		void reorder(const std::vector<Vid>& order)
		{
			for (auto& p : dict)
			{
				p.second = order[p.second];
				id2word[p.second] = p.first;
			}
		}

		const std::vector<std::string>& getRaw() const
		{
			return id2word;
		}

		Vid mapToNewDict(Vid v, const Dictionary& newDict) const
		{
			return newDict.toWid(toWord(v));
		}

		std::vector<Vid> mapToNewDict(const std::vector<Vid>& v, const Dictionary& newDict) const
		{
			std::vector<Vid> r(v.size());
			for (size_t i = 0; i < v.size(); ++i)
			{
				r[i] = mapToNewDict(v[i], newDict);
			}
			return r;
		}

		std::vector<Vid> mapToNewDictAdd(const std::vector<Vid>& v, Dictionary& newDict) const
		{
			std::vector<Vid> r(v.size());
			for (size_t i = 0; i < v.size(); ++i)
			{
				r[i] = mapToNewDict(v[i], newDict);
			}
			return r;
		}
	};

}

namespace std
{
	template<>
	struct hash<tomoto::VidPair>
	{
		size_t operator()(const tomoto::VidPair& p) const
		{
			return hash<size_t>{}(p.first) ^ hash<size_t>{}(p.second);
		}
	};
}