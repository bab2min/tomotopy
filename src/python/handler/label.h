#pragma once

#define USE_NUMPY
#ifdef _DEBUG
//#undef _DEBUG
#define DEBUG_LOG(t) do{ cerr << t << endl; }while(0)
#include "PyUtils.h"
//#define _DEBUG
#else 
#define DEBUG_LOG(t)
#include "PyUtils.h"
#endif

#include "../../Labeling/FoRelevance.h"

struct CorpusObject;

struct CandidateObject;

class CandWordIterator
{
	const CandidateObject* co = nullptr;
	size_t idx = 0;
public:
	using difference_type = ptrdiff_t;
	using value_type = const std::string;
	using reference = const std::string&;
	using pointer = const std::string*;
	using iterator_category = std::random_access_iterator_tag;

	CandWordIterator(const CandidateObject* _co = nullptr, size_t _idx = 0)
		: co{ _co }, idx{ _idx }
	{
	}

	CandWordIterator& operator++()
	{
		idx++;
		return *this;
	}

	const std::string& operator *() const;

	bool operator==(const CandWordIterator& o) const
	{
		return co == o.co && idx == o.idx;
	}

	bool operator!=(const CandWordIterator& o) const
	{
		return co != o.co || idx != o.idx;
	}

	std::ptrdiff_t operator-(const CandWordIterator& o) const
	{
		return (std::ptrdiff_t)idx - (std::ptrdiff_t)o.idx;
	}
};

struct LDAModelObject;

struct CandidateObject : public py::CObject<CandidateObject>
{
	py::UniqueCObj<LDAModelObject> tm;
	py::UniqueCObj<CorpusObject> corpus;
	tomoto::label::Candidate cand;

	std::string repr() const;

	CandWordIterator begin() const
	{
		return { this, 0 };
	}

	CandWordIterator end() const
	{
		return { this, cand.w.size() };
	}

	py::UniqueObj getWords() const;
	std::string getName() const;
	void setName(const std::string& val);
	float getScore() const;
	size_t getCf() const;
	size_t getDf() const;
};

void addLabelTypes(py::Module& module);