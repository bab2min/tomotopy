#pragma once

#ifdef _DEBUG
//#undef _DEBUG
#define DEBUG_LOG(t) do{ cerr << t << endl; }while(0)
#include "PyUtils.h"
//#define _DEBUG
#else 
#define DEBUG_LOG(t)
#include "PyUtils.h"
#endif

#include "../Labeling/FoRelevance.h"

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

struct CandidateObject
{
	PyObject_HEAD;
	TopicModelObject* tm;
	CorpusObject* corpus;
	tomoto::label::Candidate cand;

	static int init(CandidateObject* self, PyObject* args, PyObject* kwargs);
	static void dealloc(CandidateObject* self);
	static PyObject* repr(CandidateObject* self);

	CandWordIterator begin() const
	{
		return { this, 0 };
	}

	CandWordIterator end() const
	{
		return { this, cand.w.size() };
	}
};

extern PyTypeObject Candidate_type;

void addLabelTypes(PyObject* mModule);