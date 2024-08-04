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
	static constexpr Vid rm_vocab_id = (Vid)-2;
	using Tid = uint16_t;
	static constexpr Tid non_topic_id = (Tid)-1;
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

		Dictionary();
		~Dictionary();

		Dictionary(const Dictionary&);
		Dictionary& operator=(const Dictionary&);

		Dictionary(Dictionary&&) noexcept;
		Dictionary& operator=(Dictionary&&) noexcept;

		Vid add(const std::string& word);

		size_t size() const { return dict.size(); }
		
		const std::string& toWord(Vid vid) const;
		
		Vid toWid(const std::string& word) const;

		void serializerWrite(std::ostream& writer) const;
		
		void serializerRead(std::istream& reader);

		uint64_t computeHash(uint64_t seed) const;

		void swap(Dictionary& rhs);

		void reorder(const std::vector<Vid>& order);

		const std::vector<std::string>& getRaw() const;

		Vid mapToNewDict(Vid v, const Dictionary& newDict) const;

		std::vector<Vid> mapToNewDict(const std::vector<Vid>& v, const Dictionary& newDict) const;

		std::vector<Vid> mapToNewDictAdd(const std::vector<Vid>& v, Dictionary& newDict) const;
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
