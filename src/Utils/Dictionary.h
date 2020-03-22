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
	using Tid = uint16_t;
	using Float = float;

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
				dict.emplace(std::make_pair(word, dict.size()));
				id2word.emplace_back(word);
				return dict.size() - 1;
			}
			return it->second;
		}

		size_t size() const { return dict.size(); }
		
		std::string toWord(Vid vid) const
		{
			assert(vid < id2word.size());
			return id2word[vid];
		}
		
		Vid toWid(const std::string& word) const
		{
			auto it = dict.find(word);
			if (it == dict.end()) return (Vid)-1;
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
				dict.emplace(id2word[i], i);
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
	};

}
