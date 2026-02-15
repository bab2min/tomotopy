
#include "module.h"
#include "utils.h"
#include "label.h"
#include "../../Labeling/Phraser.hpp"
#include "../../Labeling/FoRelevance.h"

using namespace std;

void byte2Char(const tomoto::SharedString& str, vector<uint32_t>& startPos, vector<uint16_t>& length);

namespace py
{
	template<>
	struct ValueBuilder<tomoto::SharedString>
	{
		py::UniqueObj operator()(const tomoto::SharedString& v)
		{
			return py::UniqueObj{ PyUnicode_FromStringAndSize(v.data(), v.size()) };
		}
	};

	template<>
	struct ValueBuilder<tomoto::RawDoc>
	{
		py::UniqueObj operator()(const tomoto::RawDoc& v)
		{
			py::UniqueObj ret{ PyTuple_New(5) };
			PyTuple_SetItem(ret.get(), 0, buildPyValue(v.words, py::force_list).release());
			PyTuple_SetItem(ret.get(), 1, buildPyValue(v.rawStr).release());
			PyTuple_SetItem(ret.get(), 2, buildPyValue(v.origWordPos, py::force_list).release());
			PyTuple_SetItem(ret.get(), 3, buildPyValue(v.origWordLen, py::force_list).release());
			py::UniqueObj dict{ PyDict_New() };
			for (auto& p : v.misc)
			{
				PyObject* o = (PyObject*)std::get<std::shared_ptr<void>>(p.second).get();
				Py_INCREF(o);
				PyDict_SetItemString(dict.get(), p.first.c_str(), o);
			}
			PyTuple_SetItem(ret.get(), 4, dict.release());
			return ret;
		}
	};
}

VocabObject::~VocabObject()
{
	if (dep)
	{
		dep = {};
	}
	else if (vocabs)
	{
		delete vocabs;
		vocabs = nullptr;
	}
}

py::UniqueObj VocabObject::getstate() const
{
	static const char* keys[] = { "id2word" };
	return py::buildPyDict(keys, vocabs->getRaw());
}

void VocabObject::setstate(PyObject* args)
{
	PyObject* dict = PyTuple_GetItem(args, 0);
	PyObject* id2word = PyDict_GetItemString(dict, "id2word");
	this->~VocabObject();
	vocabs = new tomoto::Dictionary;
	dep = {};
	size = -1;
	py::foreach<std::string>(id2word, [&](const std::string& str)
	{
		vocabs->add(str);
	}, "");
	if (PyErr_Occurred()) throw py::ExcPropagation{};
}

size_t VocabObject::len() const
{
	if (size == -1) return vocabs->size();
	return size;
}

std::string_view VocabObject::getitem(size_t key) const
{
	if (key >= len()) throw py::IndexError{ std::to_string(key) };
	return vocabs->toWord(key);
}

py::UniqueObj VocabObject::repr() const
{
	py::UniqueObj args{ py::buildPyTuple((PyObject*)this) };
	py::UniqueObj l{ PyObject_CallObject((PyObject*)&PyList_Type, args.get()) };
	return py::UniqueObj{ PyObject_Repr(l.get()) };
}

class RawDocWrapper
{
	const tomoto::RawDoc& raw;
public:
	RawDocWrapper(const tomoto::RawDoc& _raw) : raw{ _raw }
	{
	}

	size_t size() const
	{
		return raw.words.size();
	}

	tomoto::Vid operator[](size_t idx) const
	{
		return raw.words[idx];
	}

	auto begin() const -> decltype(raw.words.begin())
	{
		return raw.words.begin();
	}

	auto end() const -> decltype(raw.words.end())
	{
		return raw.words.end();
	}

	auto rbegin() const -> decltype(raw.words.rbegin())
	{
		return raw.words.rbegin();
	}

	auto rend() const -> decltype(raw.words.rend())
	{
		return raw.words.rend();
	}
};

size_t CorpusObject::findUid(const std::string& uid) const
{
	if (isIndependent() || isSubDocs())
	{
		auto it = invmap.find(uid);
		if (it == invmap.end()) return -1;
		return it->second;
	}
	else
	{
		return tm->inst->getDocIdByUid(uid);
	}
}

const tomoto::RawDocKernel* CorpusObject::getDoc(size_t idx) const
{
	if (isIndependent()) return &docs[idx];
	if (made) return docsMade[idx].get();
	else return tm->inst->getDoc(isSubDocs() ? docIdcs[idx] : idx);
}

CorpusObject::CorpusObject()
{
	new (&docs) vector<tomoto::RawDoc>();
	new (&depObj) py::UniqueObj();
}

CorpusObject::CorpusObject(PyObject* vocabInst, PyObject* dep)
{
	new (&docs) vector<tomoto::RawDoc>();
	new (&depObj) py::UniqueObj();

	if (vocabInst && vocabInst != Py_None)
	{
		Py_INCREF(vocabInst);
		vocab = py::UniqueCObj<VocabObject>{ (VocabObject*)vocabInst };
		vocab->vocabs = new tomoto::Dictionary;
		vocab->size = -1;
	}
	else
	{
		Py_INCREF(dep);
		depObj = py::UniqueObj{ dep };
	}
}

CorpusObject::~CorpusObject()
{
	if (isIndependent()) docs.~vector();
	else if (made) docsMade.~vector();
	else docIdcs.~vector();
	depObj.~UniqueCObj();
}

py::UniqueObj CorpusObject::getstate() const
{
	if (!isIndependent())
		throw py::RuntimeError{ "Cannot pickle the corpus bound to a topic model. Try to use a topic model's `save()` method." };
	static const char* keys[] = { "_docs", "_vocab" };
	return py::buildPyDict(keys, docs, vocab);
}

void CorpusObject::setstate(PyObject* args)
{
	PyObject* dict = PyTuple_GetItem(args, 0);
	PyObject* vocab = PyDict_GetItemString(dict, "_vocab");
	Py_INCREF(vocab);
	this->vocab = py::UniqueCObj<VocabObject>{ (VocabObject*)vocab };
	PyObject* docs = PyDict_GetItemString(dict, "_docs");
	py::UniqueObj iter{ PyObject_GetIter(docs) }, next;
	if (!iter) throw py::ExcPropagation{};
	while ((next = py::UniqueObj{ PyIter_Next(iter.get()) }))
	{
		auto size = PyTuple_Size(next.get());
		PyObject* words = nullptr;
		PyObject* raw = nullptr;
		PyObject* pos = nullptr;
		PyObject* len = nullptr;
		PyObject* kwargs = nullptr;
		if (size == 2)
		{
			words = PyTuple_GetItem(next.get(), 0);
			kwargs = PyTuple_GetItem(next.get(), 1);
		}
		else if (size == 5)
		{
			words = PyTuple_GetItem(next.get(), 0);
			raw = PyTuple_GetItem(next.get(), 1);
			pos = PyTuple_GetItem(next.get(), 2);
			len = PyTuple_GetItem(next.get(), 3);
			kwargs = PyTuple_GetItem(next.get(), 4);
		}
		tomoto::RawDoc doc;
		doc.words = py::toCpp<vector<tomoto::Vid>>(words);
		if (raw) doc.rawStr = tomoto::SharedString{ py::toCpp<std::string>(raw) };
		if (pos) doc.origWordPos = py::toCpp<vector<uint32_t>>(pos);
		if (len) doc.origWordLen = py::toCpp<vector<uint16_t>>(len);

		PyObject* key, * value;
		Py_ssize_t p = 0;
		while (PyDict_Next(kwargs, &p, &key, &value))
		{
			auto utf8 = py::toCpp<std::string>(key);
			Py_INCREF(value);
			doc.misc[utf8] = std::shared_ptr<void>{ value, [](void* p)
			{
				Py_XDECREF(p);
			} };
		}
		this->docs.emplace_back(move(doc));
	}
	if (PyErr_Occurred()) throw py::ExcPropagation{};
}

size_t CorpusObject::addDoc(PyObject* words, PyObject* raw, PyObject* userData, PyObject* additionalKwargs)
{
	if (!isIndependent())
		throw py::RuntimeError{ "Cannot modify the corpus bound to a topic model." };
	tomoto::RawDoc doc;

	py::UniqueObj stopwords{ PyObject_GetAttrString((PyObject*)this, "_stopwords") };

	if (PyObject_HasAttrString((PyObject*)this, "_tokenizer")
		&& PyObject_IsTrue(py::UniqueObj{ PyObject_GetAttrString((PyObject*)this, "_tokenizer") }.get()))
	{
		if (words && words != Py_None) throw py::ValueError{ "only `raw` is required when `tokenizer` is provided." };
		if (!PyObject_IsTrue(raw)) return py::buildPyValue(-1);

		py::UniqueObj tokenizer{ PyObject_GetAttrString((PyObject*)this, "_tokenizer") };

		py::UniqueObj args{ PyTuple_New(1) };
		Py_INCREF(raw);
		PyTuple_SetItem(args.get(), 0, raw);
		py::UniqueObj kwargs{ PyDict_New() };
		PyDict_SetItemString(kwargs.get(), "user_data", userData);

		py::UniqueObj ret{ PyObject_Call(tokenizer.get(), args.get(), kwargs.get())};
		if (!ret) throw py::ExcPropagation{};
		py::foreach<PyObject*>(ret.get(), [&](PyObject* t)
		{
			if (PyUnicode_Check(t))
			{
				doc.words.emplace_back(vocab->vocabs->add(py::toCpp<std::string>(t)));
			}
			else if (t == Py_None)
			{
				doc.words.emplace_back(tomoto::non_vocab_id);
			}
			else if (PyTuple_Size(t) == 3)
			{
				PyObject* word = PyTuple_GetItem(t, 0);
				PyObject* pos = PyTuple_GetItem(t, 1);
				PyObject* len = PyTuple_GetItem(t, 2);
				if (!((PyUnicode_Check(word) || word == Py_None) && PyLong_Check(pos) && PyLong_Check(len))) throw py::ValueError{ "`tokenizer` must return an iterable of `str` or `tuple` of (`str`, `int`, `int`)." };
				bool isStopword = false;
				if (stopwords.get() != Py_None)
				{
					py::UniqueObj stopRet{ PyObject_CallObject(stopwords.get(), py::buildPyTuple(word).get())};
					if (!stopRet) throw py::ExcPropagation{};
					isStopword = PyObject_IsTrue(stopRet.get());
				}
				else if (word == Py_None)
				{
					isStopword = true;
				}
				doc.words.emplace_back(isStopword ? tomoto::non_vocab_id : vocab->vocabs->add(py::toCpp<std::string>(word)));
				doc.origWordPos.emplace_back(PyLong_AsLong(pos));
				doc.origWordLen.emplace_back(PyLong_AsLong(len));
			}
			else
			{
				throw py::ValueError{ "`tokenizer` must return an iterable of `str` or `tuple` of (`str`, `int`, `int`)." };
			}
		}, "`tokenizer` must return an iterable of `str` or `tuple` of (`str`, `int`, `int`).");
		doc.rawStr = tomoto::SharedString{ py::toCpp<std::string>(raw) };
	}
	else
	{
		if (raw && raw != Py_None) throw py::ValueError{ "only `words` is required when `tokenizer` is not provided." };
		if (!PyObject_IsTrue(words)) return py::buildPyValue(-1);
		py::foreach<string>(words, [&](const string& w)
		{
			bool isStopword = false;
			if (stopwords.get() != Py_None)
			{
				py::UniqueObj stopRet{ PyObject_CallObject(stopwords.get(), py::buildPyTuple(w).get())};
				if (!stopRet) throw py::ExcPropagation{};
				isStopword = PyObject_IsTrue(stopRet.get());
			}
			doc.words.emplace_back(isStopword ? -1 : vocab->vocabs->add(w));
		}, "");
	}
	PyObject* key, * value;
	Py_ssize_t p = 0;
	while (PyDict_Next(additionalKwargs, &p, &key, &value))
	{
		auto utf8 = py::toCpp<std::string>(key);
		if (utf8 == string{ "uid" })
		{
			if (value == Py_None) continue;
			std::string uid; ;
			if (!py::toCpp<std::string>(value, uid)) throw py::ValueError{ "`uid` must be str type." };
			string suid = uid;
			if (suid.empty()) throw py::ValueError{ "wrong `uid` value : empty str not allowed" };
			if (invmap.find(suid) != invmap.end())
			{
				throw py::ValueError{ "there is a document with uid = " + py::repr(value) + " already." };
			}
			invmap.emplace(suid, docs.size());
			doc.docUid = tomoto::SharedString{ uid };
			continue;
		}

		Py_INCREF(value);
		doc.misc[utf8] = std::shared_ptr<void>{ value, [](void* p)
		{
			Py_XDECREF(p);
		} };
	}
	docs.emplace_back(move(doc));
	return docs.size() - 1;
}

size_t CorpusObject::addDocs(PyObject* tokenizedIter, PyObject* rawIter, PyObject* metadataIter)
{
	if (!isIndependent())
		throw py::RuntimeError{ "Cannot modify the corpus bound to a topic model." };

	size_t cnt = 0;
	py::UniqueObj stopwords{ PyObject_GetAttrString((PyObject*)this, "_stopwords") };

	py::foreach<PyObject*>(tokenizedIter, [&](PyObject* tokenized)
	{
		tomoto::RawDoc doc;
		py::foreach<PyObject*>(tokenized, [&](PyObject* t)
		{
			if (PyUnicode_Check(t))
			{
				doc.words.emplace_back(vocab->vocabs->add(py::toCpp<std::string>(t)));
			}
			else if (t == Py_None)
			{
				doc.words.emplace_back(tomoto::non_vocab_id);
			}
			else if (PyTuple_Size(t) == 3)
			{
				PyObject* word = PyTuple_GetItem(t, 0);
				PyObject* pos = PyTuple_GetItem(t, 1);
				PyObject* len = PyTuple_GetItem(t, 2);
				if (!((PyUnicode_Check(word) || word == Py_None) && PyLong_Check(pos) && PyLong_Check(len))) throw py::ValueError{ "`tokenizer` must return an iterable of `str` or `tuple` of (`str`, `int`, `int`)." };

				bool isStopword = false;
				if (stopwords.get() != Py_None)
				{
					py::UniqueObj stopRet{ PyObject_CallObject(stopwords.get(), py::buildPyTuple(word).get())};
					if (!stopRet) throw py::ExcPropagation{};
					isStopword = PyObject_IsTrue(stopRet.get());
				}
				else if (word == Py_None)
				{
					isStopword = true;
				}
				doc.words.emplace_back(isStopword ? tomoto::non_vocab_id : vocab->vocabs->add(py::toCpp<std::string>(word)));
				doc.origWordPos.emplace_back(PyLong_AsLong(pos));
				doc.origWordLen.emplace_back(PyLong_AsLong(len));
			}
			else
			{
				throw py::ValueError{ "`tokenizer` must return an iterable of `str` or `tuple` of (`str`, `int`, `int`)." };
			}
		}, "`tokenizer` must return an iterable of `str` or `tuple` of (`str`, `int`, `int`).");
		py::UniqueObj raw{ PyIter_Next(rawIter) };
		if (!raw) throw py::ExcPropagation{};
		py::UniqueObj metadata{ PyIter_Next(metadataIter) };
		if (!metadata) throw py::ExcPropagation{};

		doc.rawStr = tomoto::SharedString{ py::toCpp<std::string>(raw.get()) };

		PyObject* key, * value;
		Py_ssize_t p = 0;
		while (PyDict_Next(metadata.get(), &p, &key, &value))
		{
			auto utf8 = py::toCpp<std::string>(key);
			if (utf8 == string{ "uid" })
			{
				if (value == Py_None) continue;
				std::string uid;
				if (!py::toCpp<std::string>(value, uid)) throw py::ValueError{ "`uid` must be str type." };
				string suid = uid;
				if (suid.empty()) throw py::ValueError{ "wrong `uid` value : empty str not allowed" };
				if (invmap.find(suid) != invmap.end())
				{
					throw py::ValueError{ "there is a document with uid = " + py::repr(value) + " already." };
				}
				invmap.emplace(suid, docs.size());
				doc.docUid = tomoto::SharedString{ uid };
				continue;
			}

			Py_INCREF(value);
			doc.misc[utf8] = std::shared_ptr<void>{ value, [](void* p)
			{
				Py_XDECREF(p);
			} };
		}
		docs.emplace_back(move(doc));
		cnt++;
	}, "");

	return cnt;
}

py::UniqueObj CorpusObject::extractNgrams(size_t minCf, size_t minDf, size_t maxLen, size_t maxCand,
	float minScore, bool normalized, size_t workers) const
{
	if (!isIndependent())
		throw py::RuntimeError{ "Cannot modify the corpus bound to a topic model." };
	size_t vSize = vocab->vocabs->size();
	vector<size_t> cf(vSize),
		df(vSize),
		odf(vSize);
	for (auto& d : docs)
	{
		for (auto w : d.words)
		{
			if (w == tomoto::non_vocab_id) continue;
			odf[w] = 1;
			cf[w]++;
		}

		for (size_t i = 0; i < df.size(); ++i) df[i] += odf[i];
		fill(odf.begin(), odf.end(), 0);
	}

	auto tx = [](const tomoto::RawDoc& raw)
	{
		return RawDocWrapper{ raw };
	};
	auto docBegin = tomoto::makeTransformIter(docs.begin(), tx);
	auto docEnd = tomoto::makeTransformIter(docs.end(), tx);
	auto cands = tomoto::phraser::extractPMINgrams(docBegin, docEnd,
		cf, df,
		minCf, minDf, 2, maxLen, maxCand, minScore, normalized
	);

	auto ret = py::UniqueObj{ PyList_New(0) };
	for (auto& c : cands)
	{
		auto item = py::makeNewObject<CandidateObject>();
		item->corpus = py::UniqueCObj<CorpusObject>{ (CorpusObject*)this };
		Py_INCREF(this);
		item->cand = move(c);
		PyList_Append(ret.get(), (PyObject*)item.get());
	}
	return ret;
}

// TODO: It loses some ngram patterns. Fix me!
size_t CorpusObject::concatNgrams(PyObject* cands, const std::string& delimiter)
{
	if (!isIndependent())
		throw py::RuntimeError{ "Cannot modify the corpus bound to a topic model." };

	py::UniqueObj iter{ PyObject_GetIter(cands) };
	if (!iter) throw py::ValueError{ "`cands` must be an iterable of `tomotopy.label.Candidate`" };
	vector<tomoto::label::Candidate> pcands;
	vector<tomoto::Vid> pcandVids;
	{
		py::UniqueObj item;
		while ((item = py::UniqueObj{ PyIter_Next(iter.get()) }))
		{
			if (!PyObject_TypeCheck(item, py::Type<CandidateObject>))
			{
				throw py::ValueError{ "`cands` must be an iterable of `tomotopy.label.Candidate`" };
			}
			CandidateObject* cand = (CandidateObject*)item.get();
			if (cand->corpus.get() == this)
			{
				pcands.emplace_back(cand->cand);
			}
			else if(cand->corpus)
			{
				tomoto::label::Candidate c = cand->cand;
				c.w = cand->corpus->vocab->vocabs->mapToNewDict(c.w, *vocab->vocabs);
				if (find(c.w.begin(), c.w.end(), tomoto::non_vocab_id) != c.w.end())
				{
					auto repr = py::repr(item.get());
					if (PyErr_WarnEx(PyExc_RuntimeWarning, 
						("Candidate is ignored because it is not found in the corpus.\n" + repr).c_str(), 1
					)) throw py::ExcPropagation{};
					continue;
				}
				pcands.emplace_back(move(c));
			}
			else if (cand->tm)
			{
				tomoto::label::Candidate c = cand->cand;
				c.w = cand->tm->inst->getVocabDict().mapToNewDict(c.w, *vocab->vocabs);
				if (find(c.w.begin(), c.w.end(), tomoto::non_vocab_id) != c.w.end())
				{
					auto repr = py::repr(item.get());
					if (PyErr_WarnEx(PyExc_RuntimeWarning,
						("Candidate is ignored because it is not found in the corpus.\n" + repr).c_str(), 1
					)) throw py::ExcPropagation{};
					continue;
				}
				pcands.emplace_back(move(c));
			}
			pcandVids.emplace_back(vocab->vocabs->add(tomoto::text::join(cand->begin(), cand->end(), delimiter)));
		}
	}

	vector<tomoto::Trie<tomoto::Vid, size_t>> candTrie(1);
	candTrie.reserve(std::accumulate(pcands.begin(), pcands.end(), 0, [](size_t s, const tomoto::label::Candidate& c)
	{
		return s + c.w.size() * 2;
	}));
	auto& root = candTrie.front();

	size_t idx = 0;
	for (auto& c : pcands)
	{
		root.build(c.w.begin(), c.w.end(), ++idx, [&]()
		{
			candTrie.emplace_back();
			return &candTrie.back();
		});
	}
	root.fillFail();

	size_t totUpdated = 0;
	for (auto& doc : docs)
	{
		auto* node = &root;
		for (size_t i = 0; i < doc.words.size(); ++i)
		{
			auto* nnode = node->getNext(doc.words[i]);
			while (!nnode)
			{
				node = node->getFail();
				if (node) nnode = node->getNext(doc.words[i]);
				else break;
			}
				
			if (nnode)
			{
				node = nnode;
				if (nnode->val && nnode->val != (size_t)-1)
				{
					size_t found = nnode->val - 1;
					doc.words[i] = pcandVids[found];
					size_t len = pcands[found].w.size();
					if (len > 1)
					{
						std::fill(doc.words.begin() + i + 1 - len, doc.words.begin() + i, tomoto::rm_vocab_id);
						if (doc.origWordLen.size() > i)
						{
							doc.origWordLen[i] = (doc.origWordPos[i] + doc.origWordLen[i]) - doc.origWordPos[i - len + 1];
							doc.origWordPos[i] = doc.origWordPos[i - len + 1];
						}
					}
					totUpdated++;
				}
			}
			else
			{
				node = &root;
			}
		}

		// remove tomoto::rm_vocab_id
		size_t j = 0;
		for (size_t i = 0; i < doc.words.size(); ++i)
		{
			if (doc.words[i] != tomoto::rm_vocab_id)
			{
				doc.words[j] = doc.words[i];
				if (doc.origWordLen.size() > i)
				{
					doc.origWordLen[j] = doc.origWordLen[i];
					doc.origWordPos[j] = doc.origWordPos[i];
				}
				++j;
			}
		}
		doc.words.resize(j);
		if (doc.origWordLen.size() > j) doc.origWordLen.resize(j);
		if (doc.origWordPos.size() > j) doc.origWordPos.resize(j);
	}
	return totUpdated;
}

size_t CorpusObject::len() const
{
	if (isIndependent()) return docs.size();
	if (made) return docsMade.size();
	if (isSubDocs()) return docIdcs.size();
	return tm->inst->getNumDocs();
}

py::UniqueObj CorpusObject::getitem(PyObject* idx) const
{
	// indexing by int
	Py_ssize_t v = PyLong_AsLongLong(idx);
	if (v != -1 || !(PyErr_Occurred() && (PyErr_Clear(), true)))
	{
		if (v >= len() || -v > len()) throw py::IndexError{ to_string(v) };
		if (v < 0) v += len();
		auto doc = py::UniqueCObj<DocumentObject>{ (DocumentObject*)PyObject_CallFunctionObjArgs((PyObject*)py::Type<DocumentObject>, this, nullptr) };
		if (!doc) throw py::ExcPropagation{};
		doc->doc = getDoc(v);
		return doc;
	}
	// indexing by uid
	else if (PyUnicode_Check(idx))
	{
		string v = py::toCpp<std::string>(idx);
		auto doc = py::UniqueCObj<DocumentObject>{ (DocumentObject*)PyObject_CallFunctionObjArgs((PyObject*)py::Type<DocumentObject>, this, nullptr) };
		if (!doc) throw py::ExcPropagation{};
		size_t iidx = findUid(v);
		if (iidx == (size_t)-1) throw py::KeyError{ "Cannot find a document with uid = " + py::repr(idx)  }; 
		doc->doc = getDoc(iidx);
		return doc;
	}
	// slicing
	else if (PySlice_Check(idx))
	{
		Py_ssize_t start, end, step, size;
		if (PySlice_GetIndicesEx(idx, len(), &start, &end, &step, &size))
		{
			throw py::ExcPropagation{};
		}

		if (isIndependent())
		{
			auto ret = py::UniqueCObj<CorpusObject>{ (CorpusObject*)PyObject_CallFunctionObjArgs((PyObject*)py::Type<CorpusObject>, vocab.get(), nullptr)};
			if (!ret) throw py::ExcPropagation{};
			for (Py_ssize_t i = start; i < end; i += step)
			{
				ret->docs.emplace_back(docs[i]);
			}
			return ret;
		}
		else if (made)
		{
			auto ret = py::UniqueCObj<CorpusObject>{ (CorpusObject*)PyObject_CallFunctionObjArgs((PyObject*)py::Type<CorpusObject>, tm.get(), nullptr)};
			if (!ret) throw bad_exception{};
			for (Py_ssize_t i = start; i < end; i += step)
			{
				ret->docsMade.emplace_back(docsMade[i]);
				ret->invmap.emplace(docsMade[i]->docUid, i);
			}
			return ret;
		}
		else if(isSubDocs())
		{
			auto ret = py::UniqueCObj<CorpusObject>{ (CorpusObject*)PyObject_CallFunctionObjArgs((PyObject*)py::Type<CorpusObject>, tm.get(), nullptr) };
			if (!ret) throw bad_exception{};
			for (Py_ssize_t i = start; i < end; i += step)
			{
				ret->docIdcs.emplace_back(docIdcs[i]);
				ret->invmap.emplace(ret->tm->inst->getDoc(i)->docUid, i);
			}
			return ret;
		}
		else 
		{
			auto ret = py::UniqueCObj<CorpusObject>{ (CorpusObject*)PyObject_CallFunctionObjArgs((PyObject*)py::Type<CorpusObject>, tm.get(), nullptr) };
			if (!ret) throw bad_exception{};
			for (Py_ssize_t i = start; i < end; i += step)
			{
				ret->docIdcs.emplace_back(i);
				ret->invmap.emplace(ret->tm->inst->getDoc(i)->docUid, i);
			}
			return ret;
		}
	}
	// indexing by list of uid or int
	else if (py::UniqueObj{ PyObject_GetIter(idx) })
	{
		vector<size_t> idcs;
		py::foreach<PyObject*>(idx, [&](PyObject* o)
		{
			Py_ssize_t v = PyLong_AsLongLong(o);
			if (v != -1 || !(PyErr_Occurred() && (PyErr_Clear(), true)))
			{
				if (v >= len() || -v > len())
				{
					throw py::IndexError{ "len = " + to_string(len()) + ", idx = " + to_string(v) };
				}
				if (v < 0) v += len();
				idcs.emplace_back((size_t)v);
			}
			else if (PyUnicode_Check(o))
			{
				string k = py::toCpp<string>(o);
				size_t idx = findUid(k);
				if (idx == (size_t)-1) throw py::KeyError{ "Cannot find a document with uid = " + py::repr(o) };
				idcs.emplace_back(idx);
			}
			else
			{
				throw py::IndexError{ string{"Unsupported index "} + py::repr(o)};
			}
		}, "");

		if (isIndependent())
		{
			auto ret = py::UniqueCObj<CorpusObject>{ (CorpusObject*)PyObject_CallFunctionObjArgs((PyObject*)py::Type<CorpusObject>, vocab.get(), nullptr)};
			if (!ret) throw py::ExcPropagation{};
			for (auto i : idcs)
			{
				ret->docs.emplace_back(docs[i]);
			}
			return ret;
		}
		else if (made)
		{
			auto ret = py::UniqueCObj<CorpusObject>{ (CorpusObject*)PyObject_CallFunctionObjArgs((PyObject*)py::Type<CorpusObject>, tm.get(), nullptr)};
			if (!ret) throw py::ExcPropagation{};
			for (auto i : ret->docIdcs)
			{
				ret->docsMade.emplace_back(docsMade[i]);
				ret->invmap.emplace(docsMade[i]->docUid, i);
			}
			return ret;
		}
		else
		{
			auto ret = py::UniqueCObj<CorpusObject>{ (CorpusObject*)PyObject_CallFunctionObjArgs((PyObject*)py::Type<CorpusObject>, tm.get(), nullptr)};
			if (!ret) throw py::ExcPropagation{};
			ret->docIdcs = move(idcs);
			for (auto i : ret->docIdcs)
			{
				ret->invmap.emplace(tm->inst->getDoc(i)->docUid, i);
			}
			return ret;
		}
	}
	else
	{
		throw py::IndexError{ string{"Unsupported indexing type "} + py::repr(idx) };
	}
}

py::UniqueCObj<CorpusIterObject> CorpusObject::iter()
{
	auto ret = py::makeNewObject<CorpusIterObject>();
	if (!ret) throw py::ExcPropagation{};
	ret->corpus = py::UniqueCObj<CorpusObject>{ this };
	Py_INCREF(this);
	return ret;
}

const tomoto::Dictionary& CorpusObject::getVocabDict() const
{
	if (isIndependent()) return *vocab->vocabs;
	return tm->inst->getVocabDict();
}

py::UniqueCObj<CorpusIterObject> CorpusIterObject::iter()
{
	Py_INCREF(this);
	return py::UniqueCObj<CorpusIterObject>{ this };
}

py::UniqueCObj<DocumentObject> CorpusIterObject::iternext()
{
	if (idx >= corpus->len()) throw py::ExcPropagation{};
	py::UniqueObj args = py::buildPyTuple(corpus);
	auto doc = py::UniqueCObj<DocumentObject>{ (DocumentObject*)PyObject_CallObject((PyObject*)py::Type<DocumentObject>, args.get()) };
	if (!doc) throw py::ExcPropagation{};
	doc->doc = corpus->getDoc(idx);
	idx++;
	return doc;
}

DocWordIterator wordBegin(const tomoto::RawDocKernel* doc, bool independent)
{
	if (independent)
	{
		auto rdoc = (const tomoto::RawDoc*)doc;
		return { rdoc->words.data(), nullptr, 0 };
	}
	auto rdoc = (const tomoto::DocumentBase*)doc;
	return { rdoc->words.data(), rdoc->wOrder.empty() ? nullptr : rdoc->wOrder.data(), 0 };
}

DocWordIterator wordEnd(const tomoto::RawDocKernel* doc, bool independent)
{
	if (independent)
	{
		auto rdoc = (const tomoto::RawDoc*)doc;
		return { rdoc->words.data(), nullptr, rdoc->words.size() };
	}
	auto rdoc = (const tomoto::DocumentBase*)doc;
	return { rdoc->words.data(), rdoc->wOrder.empty() ? nullptr : rdoc->wOrder.data(), rdoc->words.size() };
}

DocumentObject::DocumentObject(PyObject* corpus)
{
	this->corpus = py::UniqueCObj<CorpusObject>{ (CorpusObject*)corpus };
	Py_INCREF(corpus);
}

DocumentObject::~DocumentObject()
{
	if (!corpus->isIndependent() && owner)
	{
		delete getBoundDoc();
	}
}

size_t DocumentObject::len() const
{
	if (!doc) return 0;
	if (corpus->isIndependent())
	{
		return getRawDoc()->words.size();
	}
	return getBoundDoc()->words.size();
}

std::optional<std::string_view> DocumentObject::getitem(size_t idx) const
{
	if (idx >= len()) throw py::IndexError{ std::to_string(idx) };
	if (corpus->isIndependent())
	{
		if (getRawDoc()->words[idx] == tomoto::non_vocab_id)
		{
			return std::nullopt;
		}
		return corpus->getVocabDict().toWord(getRawDoc()->words[idx]);
	}
	else
	{
		idx = getBoundDoc()->wOrder.empty() ? idx : getBoundDoc()->wOrder[idx];
		return corpus->getVocabDict().toWord(getBoundDoc()->words[idx]);
	}
}

py::UniqueObj DocumentObject::getAllWords() const
{
	if (corpus->isIndependent()) return py::buildPyValue(getRawDoc()->words);
	else return buildPyValueReorder(getBoundDoc()->words, getBoundDoc()->wOrder);
}

const tomoto::SharedString& DocumentObject::getRaw() const
{
	return doc->rawStr;
}

py::UniqueObj DocumentObject::getSpan() const
{
	auto starts = doc->origWordPos;
	auto lengthes = doc->origWordLen;
	byte2Char(doc->rawStr, starts, lengthes);

	auto ret = py::UniqueObj{ PyList_New(starts.size()) };
	for (size_t i = 0; i < starts.size(); ++i)
	{
		size_t begin = starts[i], end = begin + lengthes[i];
		PyList_SetItem(ret.get(), i, py::buildPyTuple(begin, end).release());
	}
	return ret;
}

tomoto::Float DocumentObject::getWeight() const
{
	return doc->weight;
}

const tomoto::SharedString& DocumentObject::getUid() const
{
	return doc->docUid;
}

py::UniqueObj DocumentObject::getattro(PyObject* attr) const
{
	if (!corpus->isIndependent()) return py::UniqueObj{ PyObject_GenericGetAttr((PyObject*)this, attr) };
	std::string a;
	if (!py::toCpp<std::string>(attr, a)) throw py::AttributeError{ "invalid attribute name" };
	string name = a;
	auto it = getRawDoc()->misc.find(name);
	if (it == getRawDoc()->misc.end()) return py::UniqueObj{ PyObject_GenericGetAttr((PyObject*)this, attr) };
	auto ret = py::UniqueObj{ (PyObject*)std::get<std::shared_ptr<void>>(it->second).get() };
	Py_INCREF(ret.get());
	return ret;
}

std::string DocumentObject::repr() const
{
	string ret = "<tomotopy.Document with words=\"";

	for (size_t i = 0; i < len(); ++i)
	{
		size_t w;
		if (corpus->isIndependent())
		{
			w = getRawDoc()->words[i];
			if (w == tomoto::non_vocab_id) continue;
		}
		else
		{
			w = getBoundDoc()->wOrder.empty() ? getBoundDoc()->words[i] : getBoundDoc()->words[getBoundDoc()->wOrder[i]];
		}
		ret += corpus->getVocabDict().toWord(w);
		ret.push_back(' ');
	}
	ret.pop_back();
	ret += "\">";
	return ret;
}

std::vector<std::pair<tomoto::Tid, tomoto::Float>> DocumentObject::getTopics(size_t topN, bool fromPseudoDoc) const
{
	if (corpus->isIndependent()) throw py::RuntimeError{ "This method can only be called by documents bound to the topic model." };
	if (!corpus->tm->inst) throw py::RuntimeError{ "inst is null" };
	if (!corpus->tm->isPrepared) throw py::RuntimeError{ "train() should be called first for calculating the topic distribution" };

	if (owner && !initialized)
	{
		if (PyErr_WarnEx(PyExc_RuntimeWarning, "This document has no topic information. Call `infer()` method passing this document as an argument first!", 1)) throw py::ExcPropagation{};
	}

	if (fromPseudoDoc) return getTopicsFromPseudoDoc(topN);
	return corpus->tm->inst->getTopicsByDocSorted(getBoundDoc(), topN);
}

std::vector<float> DocumentObject::getTopicDist(bool normalize, bool fromPseudoDoc) const
{
	if (corpus->isIndependent()) throw py::RuntimeError{ "This method can only be called by documents bound to the topic model." };
	if (!corpus->tm->inst) throw py::RuntimeError{ "inst is null" };
	if (!corpus->tm->isPrepared) throw py::RuntimeError{ "train() should be called first for calculating the topic distribution" };

	if (owner && !initialized)
	{
		if (PyErr_WarnEx(PyExc_RuntimeWarning, "This document has no topic information. Call `infer()` method passing this document as an argument first!", 1)) throw py::ExcPropagation{};
	}
	if (fromPseudoDoc) return getTopicDistFromPseudoDoc(normalize);
	return corpus->tm->inst->getTopicsByDoc(getBoundDoc(), normalize);
}

std::vector<std::pair<std::string, tomoto::Float>> DocumentObject::getWords(size_t topN) const
{
	if (corpus->isIndependent()) throw py::RuntimeError{ "This method can only be called by documents bound to the topic model." };
	if (!corpus->tm->inst) throw py::RuntimeError{ "inst is null" };
	return corpus->tm->inst->getWordsByDocSorted(getBoundDoc(), topN);
}

py::UniqueObj DocumentObject::getZ() const
{
	if (corpus->isIndependent()) throw py::AttributeError{ "doc has no `topics` field!" };
	if (!doc) throw py::RuntimeError{ "doc is null!" };
	if (auto ret = getZFromHLDA()) return ret;
	if (auto ret = getZFromHDP()) return ret;
	if (auto ret = getZFromLDA()) return ret;
	throw py::AttributeError{ "doc has no `topics` field!" };
}

std::string_view DocumentObject::getMetadata() const
{
	if (corpus->isIndependent()) throw py::AttributeError{ "doc has no `metadata` field!" };
	if (!doc) throw py::RuntimeError{ "doc is null!" };
	if (auto ret = getMetadataFromDMR()) return *ret;
	throw py::AttributeError{ "doc has no `metadata` field!" };
}

double DocumentObject::getLL() const
{
	if (corpus->isIndependent()) throw py::RuntimeError{ "This method can only be called by documents bound to the topic model." };
	if (!corpus->tm->inst) throw py::RuntimeError{ "inst is null" };
	return corpus->tm->inst->getDocLL(getBoundDoc());
}

PhraserObject::PhraserObject()
	: trie_nodes{ 1 } // root node
{
}

PhraserObject::PhraserObject(PyObject* candidates, const std::string& delimiter)
{
	if (!candidates || candidates == Py_None) return;

	py::UniqueObj iter{ PyObject_GetIter(candidates) }, item;
	if (!iter) throw py::ValueError{ "`candidates` must be an iterable of Candidates." };

	CorpusObject* base_corpus = nullptr;
	auto alloc = [&]() { trie_nodes.emplace_back(); return &trie_nodes.back(); };

	while ((item = py::UniqueObj{ PyIter_Next(iter.get()) }))
	{
		if (PyObject_TypeCheck(item, py::Type<CandidateObject>))
		{
			auto c = (CandidateObject*)item.get();
			if (!vocabs.size())
			{
				vocabs = c->corpus->getVocabDict();
				base_corpus = c->corpus.get();
			}

			if (trie_nodes.capacity() < trie_nodes.size() + c->cand.w.size())
			{
				trie_nodes.reserve(max(trie_nodes.size() + c->cand.w.size(), trie_nodes.size() * 2));
			}

			if (c->corpus.get() == base_corpus)
			{
				trie_nodes[0].build(c->cand.w.begin(), c->cand.w.end(), cand_info.size() + 1, alloc);
			}
			else
			{
				auto new_cw = c->corpus->getVocabDict().mapToNewDictAdd(c->cand.w, vocabs);
				trie_nodes[0].build(new_cw.begin(), new_cw.end(), cand_info.size() + 1, alloc);
			}

			string name = c->cand.name;
			if (name.empty())
			{
				for (auto& w : *c)
				{
					if (!name.empty()) name += delimiter;
					name += w;
				}
			}
			cand_info.emplace_back(name, c->cand.w.size());
		}
		else if (PyTuple_Size(item.get()) == 2)
		{
			string name;
			if (!py::toCpp<string>(PyTuple_GetItem(item.get(), 1), name))
			{
				throw std::invalid_argument{ "`candidates` must be an iterable of `(list of str, str)`." };
			}
			vector<tomoto::Vid> ws;
			py::foreach<string>(PyTuple_GetItem(item.get(), 0), [&](const string& w)
			{
				ws.emplace_back(vocabs.add(w));
			}, "`candidates` must be an iterable of `(list of str, str)`.");

			if (trie_nodes.capacity() < trie_nodes.size() + ws.size())
			{
				trie_nodes.reserve(max(trie_nodes.size() + ws.size(), trie_nodes.size() * 2));
			}
			trie_nodes[0].build(ws.begin(), ws.end(), cand_info.size() + 1, alloc);
			cand_info.emplace_back(name, ws.size());
		}
		else
		{
			throw py::ValueError{ "`candidates` must be an iterable of Candidates." };
		}
	}
	if (PyErr_Occurred()) throw py::ExcPropagation{};
	trie_nodes[0].fillFail();
	trie_nodes.shrink_to_fit();
}


std::string PhraserObject::repr() const
{
	string ret = "Phraser(... with ";
	ret += to_string(cand_info.size());
	ret += " items)";
	return ret;
}

py::UniqueObj PhraserObject::call(PyObject* words) const
{
	py::UniqueObj ret{ PyList_New(0) };
	deque<tomoto::Vid> buffer;
	size_t c_found = 0;
	auto* node = trie_nodes.data();
	py::foreachWithPy<string>(words, [&](const string& w, PyObject* pw)
	{
		auto wid = vocabs.toWid(w);

		if (wid != tomoto::non_vocab_id)
		{
			auto* nnode = node->getNext(wid);
			if (!nnode)
			{
				if (c_found)
				{
					auto& info = cand_info[c_found - 1];
					if (buffer.size() >= info.second)
					{
						PyList_Append(ret.get(), py::buildPyValue(info.first).get());
						buffer.erase(buffer.begin(), buffer.begin() + info.second);
					}
					c_found = 0;
				}
			}

			while (!nnode)
			{
				size_t curDepth = node->depth;
				node = node->getFail();
				for (size_t d = node ? node->depth : 0; !buffer.empty() && d < curDepth; ++d)
				{
					PyList_Append(ret.get(), py::buildPyValue(vocabs.toWord(buffer.front())).get());
					buffer.pop_front();
				}
				if (node) nnode = node->getNext(wid);
				else break;
			}

			if (nnode)
			{
				node = nnode;
				if (nnode->val && nnode->val != (size_t)-1)
				{
					c_found = nnode->val;
				}
				buffer.emplace_back(wid);
				return;
			}
		}

		for (auto v : buffer) PyList_Append(ret.get(), py::buildPyValue(vocabs.toWord(v)).get());
		buffer.clear();
		PyList_Append(ret.get(), pw);
		node = trie_nodes.data();
	}, "`words` must be an iterable of `str`s.");
	if (c_found)
	{
		auto& info = cand_info[c_found - 1];
		if (buffer.size() >= info.second)
		{
			PyList_Append(ret.get(), py::buildPyValue(info.first).get());
			buffer.erase(buffer.begin(), buffer.begin() + info.second);
			c_found = 0;
		}
	}
	for (auto v : buffer) PyList_Append(ret.get(), py::buildPyValue(vocabs.toWord(v)).get());
	return ret;
}

py::UniqueObj PhraserObject::findall(PyObject* words) const
{
	py::UniqueObj ret{ PyList_New(0) };
	size_t c_found = 0, stack_size = 0, cur_pos = 0;
	auto* node = trie_nodes.data();
	py::foreach<string>(words, [&](const string& w)
	{
		auto wid = vocabs.toWid(w);

		if (wid != tomoto::non_vocab_id)
		{
			auto* nnode = node->getNext(wid);
			if (!nnode)
			{
				if (c_found)
				{
					auto& info = cand_info[c_found - 1];
					if (stack_size >= info.second)
					{
						assert(cur_pos - stack_size < PyObject_Length(words));
						assert(cur_pos - stack_size + info.second <= PyObject_Length(words));
						PyList_Append(ret.get(), py::buildPyTuple(info.first, py::buildPyTuple(cur_pos - stack_size, cur_pos - stack_size + info.second)).get());
						stack_size -= info.second;

						size_t targetDepth = node->depth - info.second;
						while (node->depth > targetDepth)
						{
							node = node->getFail();
						}
						nnode = node->getNext(wid);
					}
					c_found = 0;
				}

				while (!nnode)
				{
					size_t curDepth = node->depth;
					node = node->getFail();
					stack_size -= curDepth - (node ? node->depth : 0);
					if (node) nnode = node->getNext(wid);
					else break;
				}
			}

			if (nnode)
			{
				node = nnode;
				if (nnode->val && nnode->val != (size_t)-1)
				{
					c_found = nnode->val;
				}
				stack_size++;
				cur_pos++;
				return;
			}
		}
		stack_size = 0;
		node = trie_nodes.data();
		cur_pos++;
	}, "`words` must be an iterable of `str`s.");
	if (c_found)
	{
		auto& info = cand_info[c_found - 1];
		if (stack_size >= info.second)
		{
			PyList_Append(ret.get(), py::buildPyTuple(info.first, py::buildPyTuple(cur_pos - stack_size, cur_pos - stack_size + info.second)).get());
			c_found = 0;
		}
	}
	return ret;
}

void PhraserObject::save(const std::string& path) const
{
	ofstream ofs{ path, ios_base::binary };
	if (!ofs) throw py::OSError{ string{"cannot write to '"} + path + "'" };
	tomoto::serializer::writeMany(ofs, tomoto::serializer::to_keyz("tph1"),
		vocabs,
		cand_info,
		trie_nodes
	);
}

py::UniqueCObj<PhraserObject> PhraserObject::load(PyObject* cls, const std::string& path)
{
	if (!cls) cls = (PyObject*)py::Type<PhraserObject>;
	else if (!PyObject_IsSubclass(cls, (PyObject*)py::Type<PhraserObject>)) throw runtime_error{ "`cls` must be a derived class of `Phraser`." };

	ifstream ifs{ path };
	if (!ifs) throw py::OSError{ string{"cannot read from '"} + path + "'" };
	py::UniqueCObj<PhraserObject> ret{ (PhraserObject*)PyObject_CallObject(cls, nullptr) };
	if (!ret) throw py::ExcPropagation{};
	tomoto::serializer::readMany(ifs, tomoto::serializer::to_keyz("tph1"),
		ret->vocabs,
		ret->cand_info,
		ret->trie_nodes
	);
	return ret;
}

void addUtilsTypes(py::Module& module)
{
	module.addType(py::define<VocabObject>("_VocabDict", "_UtilsVocabDict", Py_TPFLAGS_BASETYPE)
		.sqLen<&VocabObject::len>()
		.sqGetItem<&VocabObject::getitem>()
		.method<&VocabObject::getstate>("__getstate__")
		.method<&VocabObject::setstate>("__setstate__")
	);

	module.addType(py::define<CorpusObject>("tomotopy._UtilsCorpus", "_UtilsCorpus", Py_TPFLAGS_BASETYPE)
		.mpLen<&CorpusObject::len>()
		.mpGetItem<&CorpusObject::getitem>()
		.method<&CorpusObject::getstate>("__getstate__")
		.method<&CorpusObject::setstate>("__setstate__")
		.method<&CorpusObject::addDoc>("add_doc")
		.method<&CorpusObject::addDocs>("add_docs")
		.method<&CorpusObject::extractNgrams>("extract_ngrams")
		.method<&CorpusObject::concatNgrams>("concat_ngrams"));

	module.addType(py::define<CorpusIterObject>("tomotopy._UtilsCorpusIter", "_UtilsCorpusIter", Py_TPFLAGS_DEFAULT)
		.iter<&CorpusIterObject::iter>()
		.iternext<&CorpusIterObject::iternext>());

	module.addType(py::define<DocumentObject>("tomotopy._Document", "_Document", Py_TPFLAGS_DEFAULT)
		.sqLen<&DocumentObject::len>()
		.sqGetItem<&DocumentObject::getitem>()
		.getAttrO<&DocumentObject::getattro>()
		.repr<&DocumentObject::repr>()
		.method<&DocumentObject::getTopics>("get_topics")
		.method<&DocumentObject::getTopicDist>("get_topic_dist")
		.method<&DocumentObject::getSubTopics>("get_sub_topics")
		.method<&DocumentObject::getSubTopicDist>("get_sub_topic_dist")
		.method<&DocumentObject::getWords>("get_words")
		.method<&DocumentObject::getCountVector>("get_count_vector")
		.method<&DocumentObject::getLL>("get_ll")
		.property<&DocumentObject::getAllWords>("words")
		.property<&DocumentObject::getWeight>("weight")
		.property<&DocumentObject::getZ>("topics")
		.property<&DocumentObject::getUid>("uid")
		.property<&DocumentObject::getRaw>("raw")
		.property<&DocumentObject::getSpan>("span")
		.property<&DocumentObject::getMetadata>("metadata")
		.property<&DocumentObject::getMultiMetadata>("multi_metadata")
		.property<&DocumentObject::getNumericMetadata>("numeric_metadata")
		.property<&DocumentObject::getZ2>("subtopics")
		.property<&DocumentObject::getWindows>("windows")
		.property<&DocumentObject::getPath>("path")
		.property<&DocumentObject::getBeta>("beta")
		.property<&DocumentObject::getY>("vars")
		.property<&DocumentObject::getLabels>("labels")
		.property<&DocumentObject::getEta>("eta")
		.property<&DocumentObject::getTimepoint>("timepoint")
		.property<&DocumentObject::getPseudoDocId>("pseudo_doc_id"));

	module.addType(py::define<PhraserObject>("tomotopy._Phraser", "_Phraser", Py_TPFLAGS_BASETYPE)
		.call<&PhraserObject::call>()
		.method<&PhraserObject::findall>("findall")
		.method<&PhraserObject::save>("save")
		.staticMethod<&PhraserObject::load>("load")
		.repr<&PhraserObject::repr>());
}
