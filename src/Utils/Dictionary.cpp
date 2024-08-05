#include "Dictionary.h"

namespace tomoto
{
    Dictionary::Dictionary() = default;
    Dictionary::~Dictionary() = default;

    Dictionary::Dictionary(const Dictionary&) = default;
    Dictionary& Dictionary::operator=(const Dictionary&) = default;

    Dictionary::Dictionary(Dictionary&&) noexcept = default;
    Dictionary& Dictionary::operator=(Dictionary&&) noexcept = default;

    Vid Dictionary::add(const std::string& word)
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
    
    const std::string& Dictionary::toWord(Vid vid) const
    {
        assert(vid < id2word.size());
        return id2word[vid];
    }
    
    Vid Dictionary::toWid(const std::string& word) const
    {
        auto it = dict.find(word);
        if (it == dict.end()) return non_vocab_id;
        return it->second;
    }

    void Dictionary::serializerWrite(std::ostream& writer) const
    {
        serializer::writeMany(writer, serializer::to_key("Dict"), id2word);
    }
    
    void Dictionary::serializerRead(std::istream& reader)
    {
        serializer::readMany(reader, serializer::to_key("Dict"), id2word);
        for (size_t i = 0; i < id2word.size(); ++i)
        {
            dict.emplace(id2word[i], (Vid)i);
        }
    }

    uint64_t Dictionary::computeHash(uint64_t seed) const
	{
        return serializer::computeHashMany(seed, id2word);
	}

    void Dictionary::swap(Dictionary& rhs)
    {
        std::swap(dict, rhs.dict);
        std::swap(id2word, rhs.id2word);
    }

    void Dictionary::reorder(const std::vector<Vid>& order)
    {
        for (auto& p : dict)
        {
            p.second = order[p.second];
            id2word[p.second] = p.first;
        }
    }

    const std::vector<std::string>& Dictionary::getRaw() const
    {
        return id2word;
    }

    Vid Dictionary::mapToNewDict(Vid v, const Dictionary& newDict) const
    {
        return newDict.toWid(toWord(v));
    }

    std::vector<Vid> Dictionary::mapToNewDict(const std::vector<Vid>& v, const Dictionary& newDict) const
    {
        std::vector<Vid> r(v.size());
        for (size_t i = 0; i < v.size(); ++i)
        {
            r[i] = mapToNewDict(v[i], newDict);
        }
        return r;
    }

    std::vector<Vid> Dictionary::mapToNewDictAdd(const std::vector<Vid>& v, Dictionary& newDict) const
    {
        std::vector<Vid> r(v.size());
        for (size_t i = 0; i < v.size(); ++i)
        {
            r[i] = mapToNewDict(v[i], newDict);
        }
        return r;
    }
}
