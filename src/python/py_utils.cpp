
#include "module.h"
#include "utils.h"
#include "label.h"
#include "../Labeling/FoRelevance.h"

using namespace std;

namespace py
{
	template<>
	struct ValueBuilder<tomoto::SharedString>
	{
		PyObject* operator()(const tomoto::SharedString& v)
		{
			return PyUnicode_FromStringAndSize(v.data(), v.size());
		}
	};

	template<>
	struct ValueBuilder<tomoto::RawDoc>
	{
		PyObject* operator()(const tomoto::RawDoc& v)
		{
			PyObject* ret = PyTuple_New(5);
			PyTuple_SET_ITEM(ret, 0, buildPyValue(v.words));
			PyTuple_SET_ITEM(ret, 1, buildPyValue(v.rawStr));
			PyTuple_SET_ITEM(ret, 2, buildPyValue(v.origWordPos));
			PyTuple_SET_ITEM(ret, 3, buildPyValue(v.origWordLen));
			PyObject* dict = PyDict_New();
			for (auto& p : v.misc)
			{
				PyObject* o = (PyObject*)p.second.template get<std::shared_ptr<void>>().get();
				Py_INCREF(o);
				PyDict_SetItemString(dict, p.first.c_str(), o);
			}
			PyTuple_SET_ITEM(ret, 4, dict);
			return ret;
		}
	};
}


int VocabObject::init(VocabObject* self, PyObject* args, PyObject* kwargs)
{
	self->vocabs = nullptr;
	return 0;
}

void VocabObject::dealloc(VocabObject* self)
{
	if (self->dep)
	{
		Py_XDECREF(self->dep);
		self->dep = nullptr;
	}
	else if (self->vocabs)
	{
		delete self->vocabs;
		self->vocabs = nullptr;
	}
	Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject* VocabObject::getstate(VocabObject* self, PyObject*)
{
	static const char* keys[] = { "id2word" };
	return py::buildPyDict(keys, self->vocabs->getRaw());
}

PyObject* VocabObject::setstate(VocabObject* self, PyObject* args)
{
	try
	{
		PyObject* dict = PyTuple_GetItem(args, 0);
		PyObject* id2word = PyDict_GetItemString(dict, "id2word");
		if (!self->dep && self->vocabs) delete self->vocabs;
		self->vocabs = new tomoto::Dictionary;
		self->dep = nullptr;
		self->size = -1;
		py::foreach<const char*>(id2word, [&](const char* str)
		{
			if (!str) throw bad_exception{};
			self->vocabs->add(str);
		}, "");
		if (PyErr_Occurred()) throw bad_exception{};
	}
	catch (const bad_exception&)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
	Py_INCREF(Py_None);
	return Py_None;
}

Py_ssize_t VocabObject::len(VocabObject* self)
{
	try
	{
		if(self->size == -1) return self->vocabs->size();
		return self->size;
	}
	catch (const bad_exception&)
	{
		return -1;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return -1;
	}
}

PyObject* VocabObject::getitem(VocabObject* self, Py_ssize_t key)
{
	try
	{
		if (key >= len(self))
		{
			PyErr_SetString(PyExc_IndexError, "");
			throw bad_exception{};
		}
		return py::buildPyValue(self->vocabs->toWord(key));
	}
	catch (const bad_exception&)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

PyObject* VocabObject::repr(VocabObject* self)
{
	py::UniqueObj args{ Py_BuildValue("(O)", self) };
	py::UniqueObj l{ PyObject_CallObject((PyObject*)&PyList_Type, args) };
	PyObject* r = PyObject_Repr(l);
	return r;
}

static PyMethodDef UtilsVocab_methods[] =
{
	{ "__getstate__", (PyCFunction)VocabObject::getstate, METH_NOARGS, "" },
	{ "__setstate__", (PyCFunction)VocabObject::setstate, METH_VARARGS, "" },
	{ nullptr }
};


static PySequenceMethods UtilsVocab_seq_methods = {
	(lenfunc)VocabObject::len,
	nullptr,
	nullptr,
	(ssizeargfunc)VocabObject::getitem,
};


PyTypeObject UtilsVocab_type = {
	PyVarObject_HEAD_INIT(nullptr, 0)
	"tomotopy._UtilsVocabDict",             /* tp_name */
	sizeof(VocabObject), /* tp_basicsize */
	0,                         /* tp_itemsize */
	(destructor)VocabObject::dealloc, /* tp_dealloc */
	0,                         /* tp_print */
	0,                         /* tp_getattr */
	0,                         /* tp_setattr */
	0,                         /* tp_reserved */
	(reprfunc)VocabObject::repr,                         /* tp_repr */
	0,                         /* tp_as_number */
	&UtilsVocab_seq_methods,       /* tp_as_sequence */
	0,                         /* tp_as_mapping */
	0,                         /* tp_hash  */
	0,                         /* tp_call */
	0,                         /* tp_str */
	0,                         /* tp_getattro */
	0,                         /* tp_setattro */
	0,                         /* tp_as_buffer */
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
	"",           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	0,              /* tp_iter */
	0,                         /* tp_iternext */
	UtilsVocab_methods,             /* tp_methods */
	0,						 /* tp_members */
	0,                         /* tp_getset */
	0,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)VocabObject::init,      /* tp_init */
	PyType_GenericAlloc,
	PyType_GenericNew,
};


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

CorpusObject* CorpusObject::_new(PyTypeObject* subtype, PyObject* args, PyObject* kwargs)
{
	CorpusObject* obj = (CorpusObject*)subtype->tp_alloc(subtype, 0);
	new (&obj->docs) vector<tomoto::RawDoc>;
	new (&obj->invmap) unordered_map<string, size_t>;
	obj->made = false;
	return obj;
}

int CorpusObject::init(CorpusObject* self, PyObject* args, PyObject* kwargs)
{
	PyObject* dep = nullptr;
	static const char* kwlist[] = { "dep", nullptr };

	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O", (char**)kwlist,
		&dep)) return -1;

	if (!dep)
	{
		dep = PyObject_CallObject((PyObject*)&UtilsVocab_type, nullptr);
		((VocabObject*)dep)->vocabs = new tomoto::Dictionary;
	}
	else Py_INCREF(dep);

	self->depObj = dep;
	return 0;
}

void CorpusObject::dealloc(CorpusObject* self)
{
	if (self->isIndependent()) self->docs.~vector();
	else if (self->made) self->docsMade.~vector();
	else self->docIdcs.~vector();
	self->invmap.~unordered_map();
	Py_XDECREF(self->depObj);
	self->depObj = nullptr;
}

PyObject* CorpusObject::getstate(CorpusObject* self, PyObject*)
{
	try
	{
		if (!self->isIndependent()) 
			throw runtime_error{ "Cannot pickle the corpus bound to a topic model. Try to use a topic model's `save` method." };
		static const char* keys[] = { "_docs", "_vocab" };
		return py::buildPyDict(keys, py::UniqueObj{ py::buildPyValue(self->docs) }, (PyObject*)self->vocab);
	}
	catch (const bad_exception&)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

PyObject* CorpusObject::setstate(CorpusObject* self, PyObject* args)
{
	try
	{
		PyObject* dict = PyTuple_GetItem(args, 0);
		PyObject* vocab = PyDict_GetItemString(dict, "_vocab");
		self->vocab = (VocabObject*)vocab;
		Py_INCREF(self->vocab);
		PyObject* docs = PyDict_GetItemString(dict, "_docs");
		py::UniqueObj iter{ PyObject_GetIter(docs) }, next;
		if (!iter) throw bad_exception{};
		while ((next = py::UniqueObj{ PyIter_Next(iter) }))
		{
			auto size = PyTuple_Size(next);
			PyObject* words = nullptr;
			PyObject* raw = nullptr;
			PyObject* pos = nullptr;
			PyObject* len = nullptr;
			PyObject* kwargs = nullptr;
			if (size == 2)
			{
				words = PyTuple_GetItem(next, 0);
				kwargs = PyTuple_GetItem(next, 1);
			}
			else if (size == 5)
			{
				words = PyTuple_GetItem(next, 0);
				raw = PyTuple_GetItem(next, 1);
				pos = PyTuple_GetItem(next, 2);
				len = PyTuple_GetItem(next, 3);
				kwargs = PyTuple_GetItem(next, 4);
			}
			tomoto::RawDoc doc;
			doc.words = py::toCpp<vector<tomoto::Vid>>(words, "");
			if (raw) doc.rawStr = tomoto::SharedString{ PyUnicode_AsUTF8(raw) };
			if (pos) doc.origWordPos = py::toCpp<vector<uint32_t>>(pos, "");
			if (len) doc.origWordLen = py::toCpp<vector<uint16_t>>(len, "");
				
			PyObject *key, *value;
			Py_ssize_t p = 0;
			while (PyDict_Next(kwargs, &p, &key, &value))
			{
				const char* utf8 = PyUnicode_AsUTF8(key);
				Py_INCREF(value);
				doc.misc[utf8] = std::shared_ptr<void>{ value, [](void* p)
				{
					Py_XDECREF(p);
				} };
			}
			self->docs.emplace_back(move(doc));
		}
		if (PyErr_Occurred()) throw bad_exception{};
	}
	catch (const bad_exception&)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
	Py_INCREF(Py_None);
	return Py_None;
}

PyObject* CorpusObject::addDoc(CorpusObject* self, PyObject* args, PyObject* kwargs)
{
	try
	{
		if (!self->isIndependent())
			throw runtime_error{ "Cannot modify the corpus bound to a topic model." };
		if (PyTuple_Size(args) != 3) throw runtime_error{ "function takes 3 positional arguments." };
		PyObject* words = PyTuple_GetItem(args, 0);
		PyObject* raw = PyTuple_GetItem(args, 1);
		PyObject* user_data = PyTuple_GetItem(args, 2);

		tomoto::RawDoc doc;

		py::UniqueObj stopwords{ PyObject_GetAttrString((PyObject*)self, "_stopwords") };

		if (PyObject_HasAttrString((PyObject*)self, "_tokenizer")
			&& PyObject_IsTrue(py::UniqueObj{ PyObject_GetAttrString((PyObject*)self, "_tokenizer") }))
		{
			if (words && words != Py_None) throw runtime_error{ "only `raw` is required when `tokenizer` is provided." };
			if (!PyObject_IsTrue(raw)) return py::buildPyValue(-1);

			py::UniqueObj tokenizer{ PyObject_GetAttrString((PyObject*)self, "_tokenizer") };
				
			py::UniqueObj args{ PyTuple_New(1) };
			Py_INCREF(raw);
			PyTuple_SET_ITEM(args.get(), 0, raw);
			py::UniqueObj kwargs{ PyDict_New() };
			PyDict_SetItemString(kwargs, "user_data", user_data);

			py::UniqueObj ret{ PyObject_Call(tokenizer, args, kwargs) };
			if (!ret) throw bad_exception{};
			py::foreach<PyObject*>(ret, [&](PyObject* t)
			{
				if (PyUnicode_Check(t))
				{
					doc.words.emplace_back(self->vocab->vocabs->add(PyUnicode_AsUTF8(t)));
				}
				else if (PyTuple_Size(t) == 3)
				{
					PyObject* word = PyTuple_GetItem(t, 0);
					PyObject* pos = PyTuple_GetItem(t, 1);
					PyObject* len = PyTuple_GetItem(t, 2);
					if(!(PyUnicode_Check(word) && PyLong_Check(pos) && PyLong_Check(len))) throw runtime_error{ "`tokenizer` must return an iterable of `str` or `tuple` of (`str`, `int`, `int`)." };

					py::UniqueObj stopRet{ PyObject_CallObject(stopwords, py::UniqueObj{ py::buildPyTuple(word) }) };
					if (!stopRet) throw bad_exception{};
					doc.words.emplace_back(PyObject_IsTrue(stopRet) ? -1 : self->vocab->vocabs->add(PyUnicode_AsUTF8(word)));
					doc.origWordPos.emplace_back(PyLong_AsLong(pos));
					doc.origWordLen.emplace_back(PyLong_AsLong(len));
				}
				else
				{
					throw runtime_error{ "`tokenizer` must return an iterable of `str` or `tuple` of (`str`, `int`, `int`)." };
				}
			}, "`tokenizer` must return an iterable of `str` or `tuple` of (`str`, `int`, `int`).");
			doc.rawStr = tomoto::SharedString{ PyUnicode_AsUTF8(raw) };
		}
		else
		{
			if (raw && raw != Py_None) throw runtime_error{ "only `words` is required when `tokenizer` is not provided." };
			if (!PyObject_IsTrue(words)) return py::buildPyValue(-1);
			py::foreach<string>(words, [&](const string& w)
			{
				py::UniqueObj stopRet{ PyObject_CallObject(stopwords, py::UniqueObj{ py::buildPyTuple(w) }) };
				if (!stopRet) throw bad_exception{};
				doc.words.emplace_back(PyObject_IsTrue(stopRet) ? -1 : self->vocab->vocabs->add(w));
			}, "");
		}
		PyObject* key, * value;
		Py_ssize_t p = 0;
		while (PyDict_Next(kwargs, &p, &key, &value))
		{
			const char* utf8 = PyUnicode_AsUTF8(key);
			if (utf8 == string{ "uid" })
			{
				if (value == Py_None) continue;
				const char* uid = PyUnicode_AsUTF8(value);
				if (!uid) throw runtime_error{ "`uid` must be str type." };
				string suid = uid;
				if (suid.empty()) throw runtime_error{ "wrong `uid` value : empty str not allowed" };
				if (self->invmap.find(suid) != self->invmap.end())
				{
					py::UniqueObj repr{ PyObject_Repr(value) };
					throw runtime_error{ string{ "there is a document with uid = " } + PyUnicode_AsUTF8(repr) + " already." };
				}
				self->invmap.emplace(suid, self->docs.size());
			}

			Py_INCREF(value);
			doc.misc[utf8] = std::shared_ptr<void>{ value, [](void* p)
			{
				Py_XDECREF(p);
			} };
		}
		self->docs.emplace_back(move(doc));
		return py::buildPyValue(self->docs.size() - 1);
	}
	catch (const bad_exception&)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

PyObject* CorpusObject::extractNgrams(CorpusObject* self, PyObject* args, PyObject* kwargs)
{
	size_t minCf = 10, minDf = 5, maxLen = 5, maxCand = 5000;
	float minScore = -INFINITY;
	static const char* kwlist[] = { "min_cf", "min_df", "max_len", "max_cand", "min_score", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnnnf", (char**)kwlist,
		&minCf, &minDf, &maxLen, &maxCand, &minScore)) return nullptr;
	try
	{
		if (!self->isIndependent())
			throw runtime_error{ "Cannot modify the corpus bound to a topic model." };
		size_t vSize = self->vocab->vocabs->size();
		vector<size_t> cf(vSize), 
			df(vSize),
			odf(vSize);
		for (auto& d : self->docs)
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
		auto docBegin = tomoto::makeTransformIter(self->docs.begin(), tx);
		auto docEnd = tomoto::makeTransformIter(self->docs.end(), tx);
		auto cands = tomoto::label::extractPMINgrams(docBegin, docEnd,
			cf, df,
			minCf, minDf, 2, maxLen, maxCand, minScore
		);

		PyObject* ret = PyList_New(0);
		for (auto& c : cands)
		{
			PyObject* item = PyObject_CallObject((PyObject*)&Candidate_type, nullptr);
			((CandidateObject*)item)->corpus = self;
			Py_INCREF(self);
			((CandidateObject*)item)->cand = move(c);
			PyList_Append(ret, item);
		}
		return ret;
	}
	catch (const bad_exception&)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

PyObject* CorpusObject::concatNgrams(CorpusObject* self, PyObject* args, PyObject* kwargs)
{
	PyObject* cands;
	const char* delimiter = "_";
	float minScore = -INFINITY;
	static const char* kwlist[] = { "cands", "delimiter", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|s", (char**)kwlist,
		&cands, &delimiter)) return nullptr;
	try
	{
		if (!self->isIndependent())
			throw runtime_error{ "Cannot modify the corpus bound to a topic model." };

		py::UniqueObj iter{ PyObject_GetIter(cands) };
		if (!iter) throw runtime_error{ "`cands` must be an iterable of `tomotopy.label.Candidate`" };
		vector<tomoto::label::Candidate> pcands;
		vector<tomoto::Vid> pcandVids;
		{
			py::UniqueObj item;
			while ((item = py::UniqueObj{ PyIter_Next(iter) }))
			{
				if (!PyObject_TypeCheck(item, &Candidate_type))
				{
					throw runtime_error{ "`cands` must be an iterable of `tomotopy.label.Candidate`" };
				}
				CandidateObject* cand = (CandidateObject*)item.get();
				if (cand->corpus == self)
				{
					pcands.emplace_back(cand->cand);
				}
				else if(cand->corpus)
				{
					tomoto::label::Candidate c = cand->cand;
					c.w = cand->corpus->vocab->vocabs->mapToNewDict(c.w, *self->vocab->vocabs);
					if (find(c.w.begin(), c.w.end(), tomoto::non_vocab_id) != c.w.end())
					{
						auto repr = py::toCpp<std::string>(py::UniqueObj{ PyObject_Repr(item.get()) });
						PRINT_WARN("Candidate is ignored because it is not found in the corpus.\n" + repr);
						continue;
					}
					pcands.emplace_back(move(c));
				}
				else if (cand->tm)
				{
					tomoto::label::Candidate c = cand->cand;
					c.w = cand->tm->inst->getVocabDict().mapToNewDict(c.w, *self->vocab->vocabs);
					if (find(c.w.begin(), c.w.end(), tomoto::non_vocab_id) != c.w.end())
					{
						auto repr = py::toCpp<std::string>(py::UniqueObj{ PyObject_Repr(item.get()) });
						PRINT_WARN("Candidate is ignored because it is not found in the corpus.\n" + repr);
						continue;
					}
					pcands.emplace_back(move(c));
				}
				pcandVids.emplace_back(self->vocab->vocabs->add(tomoto::text::join(cand->begin(), cand->end(), delimiter)));
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
		for (auto& doc : self->docs)
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
						if(len > 1) doc.words.erase(doc.words.begin() + i - len + 1, doc.words.begin() + i);
						totUpdated++;
					}
				}
				else
				{
					node = &root;
				}
			}
		}
		return py::buildPyValue(totUpdated);
	}
	catch (const bad_exception&)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

Py_ssize_t CorpusObject::len(CorpusObject* self)
{
	if (self->isIndependent()) return self->docs.size();
	if (self->made) return self->docsMade.size();
	if (self->isSubDocs()) return self->docIdcs.size();
	return self->tm->inst->getNumDocs();
}

PyObject* CorpusObject::getitem(CorpusObject* self, PyObject* idx)
{
	try 
	{
		// indexing by int
		if (PyLong_Check(idx))
		{
			Py_ssize_t v = PyLong_AsLong(idx);
			if(v >= len(self) || -v > len(self)) throw out_of_range{ "IndexError: " + to_string(v) };
			auto doc = (DocumentObject*)PyObject_CallFunctionObjArgs((PyObject*)&UtilsDocument_type, (PyObject*)self, nullptr);
			if (!doc) throw bad_exception{};
			doc->doc = self->getDoc(v);
			return (PyObject*)doc;
		}
		// indexing by uid
		else if (PyUnicode_Check(idx))
		{
			string v = PyUnicode_AsUTF8(idx);
			auto doc = (DocumentObject*)PyObject_CallFunctionObjArgs((PyObject*)&UtilsDocument_type, (PyObject*)self, nullptr);
			if (!doc) throw bad_exception{};
			size_t idx = self->findUid(v);
			if (idx == (size_t)-1) throw out_of_range{ "Cannot find a document with uid = '" + v + "'" }; 
			doc->doc = self->getDoc(idx);
			return (PyObject*)doc;
		}
		// slicing
		else if (PySlice_Check(idx))
		{
			Py_ssize_t start, end, step, size;
			if (PySlice_GetIndicesEx(idx, len(self), &start, &end, &step, &size))
			{
				throw bad_exception{};
			}

			if (self->isIndependent())
			{
				auto ret = (CorpusObject*)PyObject_CallFunctionObjArgs((PyObject*)&UtilsCorpus_type, (PyObject*)self->vocab, nullptr);
				for (Py_ssize_t i = start; i < end; i += step)
				{
					ret->docs.emplace_back(self->docs[i]);
				}
				return (PyObject*)ret;
			}
			else if (self->made)
			{
				auto ret = (CorpusObject*)PyObject_CallFunctionObjArgs((PyObject*)&UtilsCorpus_type, (PyObject*)self->tm, nullptr);
				for (Py_ssize_t i = start; i < end; i += step)
				{
					ret->docsMade.emplace_back(self->docsMade[i]);
					ret->invmap.emplace(self->docsMade[i]->docUid, i);
				}
				return (PyObject*)ret;
			}
			else if(self->isSubDocs())
			{
				auto ret = (CorpusObject*)PyObject_CallFunctionObjArgs((PyObject*)&UtilsCorpus_type, (PyObject*)self->tm, nullptr);
				for (Py_ssize_t i = start; i < end; i += step)
				{
					ret->docIdcs.emplace_back(self->docIdcs[i]);
					ret->invmap.emplace(ret->tm->inst->getDoc(i)->docUid, i);
				}
				return (PyObject*)ret;
			}
			else 
			{
				auto ret = (CorpusObject*)PyObject_CallFunctionObjArgs((PyObject*)&UtilsCorpus_type, (PyObject*)self->tm, nullptr);
				for (Py_ssize_t i = start; i < end; i += step)
				{
					ret->docIdcs.emplace_back(i);
					ret->invmap.emplace(ret->tm->inst->getDoc(i)->docUid, i);
				}
				return (PyObject*)ret;
			}
		}
		// indexing by list of uid or int
		else if (PyList_Check(idx))
		{
			vector<size_t> idcs;
			py::foreach<PyObject*>(idx, [&](PyObject* o)
			{
				if (PyLong_Check(o))
				{
					Py_ssize_t v = PyLong_AsLong(o);
					if (v >= len(self) || -v > len(self))
					{
						throw out_of_range{ "IndexError. len = " + to_string(len(self)) + ", idx = " + to_string(v) };
					}
					if (v < 0) v += len(self);
					idcs.emplace_back((size_t)v);
				}
				else if (PyUnicode_Check(o))
				{
					string k = py::toCpp<string>(o);
					size_t idx = self->findUid(k);
					if (idx == (size_t)-1) throw out_of_range{ "cannot find a document with uid = " + k };
					idcs.emplace_back(idx);
				}
				else
				{
					py::UniqueObj ty{ PyObject_Type(idx) };
					py::UniqueObj repr{ PyObject_Str(ty) };
					throw runtime_error{ string{"Unsupported indexing type "} + PyUnicode_AsUTF8(repr) };
				}
			}, "");

			if (self->isIndependent())
			{
				auto ret = (CorpusObject*)PyObject_CallFunctionObjArgs((PyObject*)&UtilsCorpus_type, (PyObject*)self->vocab, nullptr);
				for (auto i : idcs)
				{
					ret->docs.emplace_back(self->docs[i]);
				}
				return (PyObject*)ret;
			}
			else if (self->made)
			{
				auto ret = (CorpusObject*)PyObject_CallFunctionObjArgs((PyObject*)&UtilsCorpus_type, (PyObject*)self->tm, nullptr);
				for (auto i : ret->docIdcs)
				{
					ret->docsMade.emplace_back(self->docsMade[i]);
					ret->invmap.emplace(self->docsMade[i]->docUid, i);
				}
				return (PyObject*)ret;
			}
			else
			{
				auto ret = (CorpusObject*)PyObject_CallFunctionObjArgs((PyObject*)&UtilsCorpus_type, (PyObject*)self->tm, nullptr);
				ret->docIdcs = move(idcs);
				for (auto i : ret->docIdcs)
				{
					ret->invmap.emplace(self->tm->inst->getDoc(i)->docUid, i);
				}
				return (PyObject*)ret;
			}
		}
		else
		{
			py::UniqueObj ty{ PyObject_Type(idx) };
			py::UniqueObj repr{ PyObject_Str(ty) };
			throw runtime_error{ string{"Unsupported indexing type "} + PyUnicode_AsUTF8(repr) };
		}
	}
	catch (const bad_exception&)
	{
		return nullptr;
	}
	catch (const out_of_range& e)
	{
		PyErr_SetString(PyExc_KeyError, e.what());
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

PyObject* CorpusObject::iter(CorpusObject* self)
{
	auto ret = (CorpusIterObject*)PyObject_CallObject((PyObject*)&UtilsCorpusIter_type, nullptr);
	if (!ret) return nullptr;
	ret->corpus = self;
	Py_INCREF(self);
	return (PyObject*)ret;
}

const tomoto::Dictionary& CorpusObject::getVocabDict() const
{
	if (isIndependent()) return *vocab->vocabs;
	return tm->inst->getVocabDict();
}

static PyMethodDef UtilsCorpus_methods[] =
{
	{ "__getstate__", (PyCFunction)CorpusObject::getstate, METH_NOARGS, "" },
	{ "__setstate__", (PyCFunction)CorpusObject::setstate, METH_VARARGS, "" },
	{ "add_doc", (PyCFunction)CorpusObject::addDoc, METH_VARARGS | METH_KEYWORDS, "" },
	{ "extract_ngrams", (PyCFunction)CorpusObject::extractNgrams, METH_VARARGS | METH_KEYWORDS, "" },
	{ "concat_ngrams", (PyCFunction)CorpusObject::concatNgrams, METH_VARARGS | METH_KEYWORDS, "" },
	{ nullptr }
};

PyMappingMethods UtilsCorpus_mapping = {
	(lenfunc)CorpusObject::len, //lenfunc mp_length;
	(binaryfunc)CorpusObject::getitem, //binaryfunc mp_subscript;
	nullptr, //objobjargproc mp_ass_subscript;
};

PyTypeObject UtilsCorpus_type = {
	PyVarObject_HEAD_INIT(nullptr, 0)
	"tomotopy._UtilsCorpus",             /* tp_name */
	sizeof(CorpusObject), /* tp_basicsize */
	0,                         /* tp_itemsize */
	(destructor)CorpusObject::dealloc, /* tp_dealloc */
	0,                         /* tp_print */
	0,                         /* tp_getattr */
	0,                         /* tp_setattr */
	0,                         /* tp_reserved */
	0,                         /* tp_repr */
	0,                         /* tp_as_number */
	0,       /* tp_as_sequence */
	&UtilsCorpus_mapping,                         /* tp_as_mapping */
	0,                         /* tp_hash  */
	0,                         /* tp_call */
	0,                         /* tp_str */
	0,                         /* tp_getattro */
	0,                         /* tp_setattro */
	0,                         /* tp_as_buffer */
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
	"",           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	(getiterfunc)CorpusObject::iter,              /* tp_iter */
	0,                         /* tp_iternext */
	UtilsCorpus_methods,             /* tp_methods */
	0,						 /* tp_members */
	0,                         /* tp_getset */
	0,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)CorpusObject::init,      /* tp_init */
	PyType_GenericAlloc,
	(newfunc)CorpusObject::_new,
};


int CorpusIterObject::init(CorpusIterObject* self, PyObject* args, PyObject* kwargs)
{
	return 0;
}

void CorpusIterObject::dealloc(CorpusIterObject* self)
{
	Py_XDECREF(self->corpus);
	self->corpus = nullptr;
	Py_TYPE(self)->tp_free((PyObject*)self);
}

CorpusIterObject* CorpusIterObject::iter(CorpusIterObject* self)
{
	Py_INCREF(self);
	return self;
}


PyObject* CorpusIterObject::iternext(CorpusIterObject* self)
{
	if (self->idx >= CorpusObject::len(self->corpus)) return nullptr;
	py::UniqueObj args{ py::buildPyTuple((PyObject*)self->corpus) };
	auto doc = (DocumentObject*)PyObject_CallObject((PyObject*)&UtilsDocument_type, args);
	if (!doc) return nullptr;
	doc->doc = self->corpus->getDoc(self->idx);
	self->idx++;
	return (PyObject*)doc;
}

PyTypeObject UtilsCorpusIter_type = {
	PyVarObject_HEAD_INIT(nullptr, 0)
	"tomotopy._UtilsCorpusIter",             /* tp_name */
	sizeof(CorpusIterObject), /* tp_basicsize */
	0,                         /* tp_itemsize */
	(destructor)CorpusIterObject::dealloc, /* tp_dealloc */
	0,                         /* tp_print */
	0,                         /* tp_getattr */
	0,                         /* tp_setattr */
	0,                         /* tp_reserved */
	0,                         /* tp_repr */
	0,                         /* tp_as_number */
	0,       /* tp_as_sequence */
	0,                         /* tp_as_mapping */
	0,                         /* tp_hash  */
	0,                         /* tp_call */
	0,                         /* tp_str */
	0,                         /* tp_getattro */
	0,                         /* tp_setattro */
	0,                         /* tp_as_buffer */
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
	"",           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	(getiterfunc)CorpusIterObject::iter,              /* tp_iter */
	(iternextfunc)CorpusIterObject::iternext,                         /* tp_iternext */
	0,             /* tp_methods */
	0,						 /* tp_members */
	0,                         /* tp_getset */
	0,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)CorpusIterObject::init,      /* tp_init */
	PyType_GenericAlloc,
	PyType_GenericNew,
};

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

int DocumentObject::init(DocumentObject* self, PyObject* args, PyObject* kwargs)
{
	try
	{
		PyObject* corpus = nullptr;
		static const char* kwlist[] = { "corpus", nullptr };

		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O", (char**)kwlist,
			&corpus)) return -1;

		self->corpus = (CorpusObject*)corpus;
		Py_INCREF(corpus);
		self->doc = nullptr;
		return 0;
	}
	catch (const bad_exception&)
	{
		return -1;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return -1;
	}
}

void DocumentObject::dealloc(DocumentObject* self)
{
	if (!self->corpus->isIndependent() && self->owner)
	{
		delete self->getBoundDoc();
	}

	Py_XDECREF(self->corpus);
	self->corpus = nullptr;
	Py_TYPE(self)->tp_free((PyObject*)self);
}

Py_ssize_t DocumentObject::len(DocumentObject* self)
{
	if (!self->doc) return 0;
	if (self->corpus->isIndependent())
	{
		return self->getRawDoc()->words.size();
	}
	return self->getBoundDoc()->words.size();
}

PyObject* DocumentObject::getitem(DocumentObject* self, Py_ssize_t idx)
{
	try
	{
		if (idx >= len(self)) throw out_of_range{ "" };
		if (self->corpus->isIndependent())
		{
			if (self->getRawDoc()->words[idx] == tomoto::non_vocab_id)
			{
				Py_INCREF(Py_None);
				return Py_None;
			}
			return py::buildPyValue(self->corpus->getVocabDict().toWord(self->getRawDoc()->words[idx]));
		}
		else
		{
			idx = self->getBoundDoc()->wOrder.empty() ? idx : self->getBoundDoc()->wOrder[idx];
			return py::buildPyValue(self->corpus->getVocabDict().toWord(self->getBoundDoc()->words[idx]));
		}
	}
	catch (const bad_exception&)
	{
	}
	catch (const out_of_range& e)
	{
		PyErr_SetString(PyExc_IndexError, e.what());
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
	}
	return nullptr;
}

PyObject* DocumentObject::getWords(DocumentObject* self, void* closure)
{
	try
	{
		if (self->corpus->isIndependent()) return py::buildPyValue(self->getRawDoc()->words);
		else return buildPyValueReorder(self->getBoundDoc()->words, self->getBoundDoc()->wOrder);
	}
	catch (const bad_exception&)
	{
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
	}
	return nullptr;
}

PyObject* DocumentObject::getRaw(DocumentObject* self, void* closure)
{
	try
	{
		return py::buildPyValue(self->doc->rawStr);
	}
	catch (const bad_exception&)
	{
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
	}
	return nullptr;
}

PyObject* DocumentObject::getSpan(DocumentObject* self, void* closure)
{
	try
	{
		PyObject* ret = PyList_New(self->doc->origWordPos.size());
		for (size_t i = 0; i < self->doc->origWordPos.size(); ++i)
		{
			size_t begin = self->doc->origWordPos[i], end = begin + self->doc->origWordLen[i];
			PyList_SET_ITEM(ret, i, py::buildPyTuple(begin, end));
		}
		return ret;
	}
	catch (const bad_exception&)
	{
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
	}
	return nullptr;
}

PyObject* DocumentObject::getWeight(DocumentObject* self, void* closure)
{
	try
	{
		return py::buildPyValue(self->doc->weight);
	}
	catch (const bad_exception&)
	{
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
	}
	return nullptr;
}

PyObject* DocumentObject::getUid(DocumentObject* self, void* closure)
{
	try
	{
		return py::buildPyValue(self->doc->docUid);
	}
	catch (const bad_exception&)
	{
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
	}
	return nullptr;
}

PyObject* DocumentObject::getattro(DocumentObject* self, PyObject* attr)
{
	try
	{
		if(!self->corpus->isIndependent()) return PyObject_GenericGetAttr((PyObject*)self, attr);
		const char* a = PyUnicode_AsUTF8(attr);
		if (!a) throw runtime_error{ "invalid attribute name" };
		string name = a;
		auto it = self->getRawDoc()->misc.find(name);
		if (it == self->getRawDoc()->misc.end()) return PyObject_GenericGetAttr((PyObject*)self, attr);
		auto ret = (PyObject*)it->second.template get<std::shared_ptr<void>>().get();
		Py_INCREF(ret);
		return ret;
	}
	catch (const bad_exception&)
	{
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
	}
	return nullptr;
}

PyObject* DocumentObject::repr(DocumentObject* self)
{
	string ret = "<tomotopy.Document with words=\"";

	for (size_t i = 0; i < len(self); ++i)
	{
		size_t w;
		if (self->corpus->isIndependent())
		{
			w = self->getRawDoc()->words[i];
			if (w == tomoto::non_vocab_id) continue;
		}
		else
		{
			w = self->getBoundDoc()->wOrder.empty() ? self->getBoundDoc()->words[i] : self->getBoundDoc()->words[self->getBoundDoc()->wOrder[i]];
		}
		ret += self->corpus->getVocabDict().toWord(w);
		ret.push_back(' ');
	}
	ret.pop_back();
	ret += "\">";
	return py::buildPyValue(ret);
}

static PyObject* Document_getTopics(DocumentObject* self, PyObject* args, PyObject* kwargs)
{
	size_t topN = 10;
	static const char* kwlist[] = { "top_n", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|n", (char**)kwlist, &topN)) return nullptr;
	try
	{
		if (self->corpus->isIndependent()) throw runtime_error{ "This method can only be called by documents bound to the topic model." };
		if (!self->corpus->tm->inst) throw runtime_error{ "inst is null" };
		if (!self->corpus->tm->isPrepared) throw runtime_error{ "train() should be called first for calculating the topic distribution" };
		return py::buildPyValue(self->corpus->tm->inst->getTopicsByDocSorted(self->getBoundDoc(), topN));
	}
	catch (const bad_exception&)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyObject* Document_getTopicDist(DocumentObject* self)
{
	try
	{
		if (self->corpus->isIndependent()) throw runtime_error{ "This method can only be called by documents bound to the topic model." };
		if (!self->corpus->tm->inst) throw runtime_error{ "inst is null" };
		if (!self->corpus->tm->isPrepared) throw runtime_error{ "train() should be called first for calculating the topic distribution" };
		return py::buildPyValue(self->corpus->tm->inst->getTopicsByDoc(self->getBoundDoc()));
	}
	catch (const bad_exception&)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyObject* Document_getWords(DocumentObject* self, PyObject* args, PyObject* kwargs)
{
	size_t topN = 10;
	static const char* kwlist[] = { "top_n", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|n", (char**)kwlist, &topN)) return nullptr;
	try
	{
		if (self->corpus->isIndependent()) throw runtime_error{ "This method can only be called by documents bound to the topic model." };
		if (!self->corpus->tm->inst) throw runtime_error{ "inst is null" };
		return py::buildPyValue(self->corpus->tm->inst->getWordsByDocSorted(self->getBoundDoc(), topN));
	}
	catch (const bad_exception&)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyObject* Document_Z(DocumentObject* self, void* closure)
{
	PyObject* ret;
	try
	{
		if (self->corpus->isIndependent()) throw runtime_error{ "doc doesn't has `topics` field!" };
		if (!self->doc) throw runtime_error{ "doc is null!" };
#ifdef TM_HLDA
		ret = Document_HLDA_Z(self, closure);
		if (ret) return ret;
#endif
#ifdef TM_HDP
		ret = Document_HDP_Z(self, closure);
		if (ret) return ret;
#endif
		ret = Document_LDA_Z(self, closure);
		if (ret) return ret;
		throw runtime_error{ "doc doesn't has `topics` field!" };
	}
	catch (const bad_exception&)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_AttributeError, e.what());
		return nullptr;
	}
}

static PyObject* Document_metadata(DocumentObject* self, void* closure)
{
	PyObject* ret;
	try
	{
		if (self->corpus->isIndependent()) throw runtime_error{ "doc doesn't has `metadata` field!" };
		if (!self->doc) throw runtime_error{ "doc is null!" };
#ifdef TM_GDMR
		ret = Document_GDMR_metadata(self, closure);
		if (ret) return ret;
#endif
#ifdef TM_DMR
		ret = Document_DMR_metadata(self, closure);
		if (ret) return ret;
#endif
		throw runtime_error{ "doc doesn't has `metadata` field!" };
	}
	catch (const bad_exception&)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_AttributeError, e.what());
		return nullptr;
	}
}


PyObject* Document_getLL(DocumentObject* self)
{
	try
	{
		if (self->corpus->isIndependent()) throw runtime_error{ "This method can only be called by documents bound to the topic model." };
		if (!self->corpus->tm->inst) throw runtime_error{ "inst is null" };
		return py::buildPyValue(self->corpus->tm->inst->getDocLL(self->getBoundDoc()));
	}
	catch (const bad_exception&)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyMethodDef UtilsDocument_methods[] =
{
	{ "get_topics", (PyCFunction)Document_getTopics, METH_VARARGS | METH_KEYWORDS, Document_get_topics__doc__ },
	{ "get_topic_dist", (PyCFunction)Document_getTopicDist, METH_NOARGS, Document_get_topic_dist__doc__ },
#ifdef TM_PA
	{ "get_sub_topics", (PyCFunction)Document_getSubTopics, METH_VARARGS | METH_KEYWORDS, Document_get_sub_topics__doc__ },
	{ "get_sub_topic_dist", (PyCFunction)Document_getSubTopicDist, METH_NOARGS, Document_get_sub_topic_dist__doc__ },
#endif
	{ "get_words", (PyCFunction)Document_getWords, METH_VARARGS | METH_KEYWORDS, Document_get_words__doc__ },
	{ "get_count_vector", (PyCFunction)Document_getCountVector, METH_NOARGS, Document_get_count_vector__doc__ },
	{ "get_ll", (PyCFunction)Document_getLL, METH_NOARGS, Document_get_ll__doc__ },
	{ nullptr }
};

static PySequenceMethods UtilsDocument_sequence = {
	(lenfunc)DocumentObject::len,
	nullptr, 
	nullptr, 
	(ssizeargfunc)DocumentObject::getitem, 
};

static PyGetSetDef UtilsDocument_getseters[] = {
	{ (char*)"words", (getter)DocumentObject::getWords, nullptr, Document_words__doc__, nullptr },
	{ (char*)"weight", (getter)DocumentObject::getWeight, nullptr, Document_weight__doc__, nullptr },
	{ (char*)"topics", (getter)Document_Z, nullptr, Document_topics__doc__, nullptr },
	{ (char*)"uid", (getter)DocumentObject::getUid, nullptr, Document_uid__doc__, nullptr },
	{ (char*)"raw", (getter)DocumentObject::getRaw, nullptr, Document_raw__doc__, nullptr },
	{ (char*)"span", (getter)DocumentObject::getSpan, nullptr, Document_span__doc__, nullptr },
#ifdef TM_DMR
	{ (char*)"metadata", (getter)Document_metadata, nullptr, Document_metadata__doc__, nullptr },
#endif
#ifdef TM_PA
	{ (char*)"subtopics", (getter)Document_Z2, nullptr, Document_subtopics__doc__, nullptr },
#endif
#ifdef TM_MGLDA
	{ (char*)"windows", (getter)Document_windows, nullptr, Document_windows__doc__, nullptr },
#endif
#ifdef TM_HLDA
	{ (char*)"path", (getter)Document_path, nullptr, Document_path__doc__, nullptr },
#endif
#ifdef TM_CT
	{ (char*)"beta", (getter)Document_beta, nullptr, Document_beta__doc__, nullptr },
#endif
#ifdef TM_SLDA
	{ (char*)"vars", (getter)Document_y, nullptr, Document_vars__doc__, nullptr },
#endif
#ifdef TM_LLDA
	{ (char*)"labels", (getter)Document_labels, nullptr, Document_labels__doc__, nullptr },
#endif
#ifdef TM_DT
	{ (char*)"eta", (getter)Document_eta, nullptr, Document_eta__doc__, nullptr },
	{ (char*)"timepoint", (getter)Document_timepoint, nullptr, Document_timepoint__doc__, nullptr },
#endif
	{ nullptr },
};


PyTypeObject UtilsDocument_type = {
	PyVarObject_HEAD_INIT(nullptr, 0)
	"tomotopy.Document",             /* tp_name */
	sizeof(DocumentObject), /* tp_basicsize */
	0,                         /* tp_itemsize */
	(destructor)DocumentObject::dealloc, /* tp_dealloc */
	0,                         /* tp_print */
	0,                         /* tp_getattr */
	0,                         /* tp_setattr */
	0,                         /* tp_reserved */
	(reprfunc)DocumentObject::repr,                         /* tp_repr */
	0,                         /* tp_as_number */
	&UtilsDocument_sequence,       /* tp_as_sequence */
	0,                         /* tp_as_mapping */
	0,                         /* tp_hash  */
	0,                         /* tp_call */
	0,                         /* tp_str */
	(getattrofunc)DocumentObject::getattro, /* tp_getattro */
	0,                         /* tp_setattro */
	0,                         /* tp_as_buffer */
	Py_TPFLAGS_DEFAULT,   /* tp_flags */
	Document___init____doc__,           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	0,              /* tp_iter */
	0,                         /* tp_iternext */
	UtilsDocument_methods,             /* tp_methods */
	0,						 /* tp_members */
	UtilsDocument_getseters,                         /* tp_getset */
	0,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)DocumentObject::init,      /* tp_init */
	PyType_GenericAlloc,
	PyType_GenericNew,
};

tomoto::RawDoc::MiscType transformMisc(const tomoto::RawDoc::MiscType& misc, PyObject* transform)
{
	if (!transform || transform == Py_None) return misc;
	py::UniqueObj args{ py::buildPyValue(misc) };
	py::UniqueObj ret{ PyObject_CallFunctionObjArgs(transform, args.get(), nullptr) };
	if (!ret) throw bad_exception{};
	return py::toCpp<tomoto::RawDoc::MiscType>(ret, "`transform` must return an instance of `dict`.");
}

vector<size_t> insertCorpus(TopicModelObject* self, PyObject* _corpus, PyObject* transform)
{
	vector<size_t> ret;
	if (!_corpus || _corpus == Py_None) return ret;
	if (!PyObject_TypeCheck(_corpus, &UtilsCorpus_type)) throw runtime_error{ "`corpus` must be an instance of `tomotopy.utils.Corpus`" };
	auto corpus = (CorpusObject*)_corpus;
	bool insert_into_empty = self->inst->updateVocab(corpus->getVocabDict().getRaw());
	for (auto& rdoc : corpus->docs)
	{
		tomoto::RawDoc doc;
		doc.rawStr = rdoc.rawStr;
		doc.weight = rdoc.weight;
		doc.docUid = rdoc.docUid;

		for (size_t i = 0; i < rdoc.words.size(); ++i)
		{
			if(rdoc.words[i] == tomoto::non_vocab_id) continue;
			if (insert_into_empty) doc.words.emplace_back(rdoc.words[i]);
			else doc.words.emplace_back(corpus->getVocabDict().mapToNewDict(rdoc.words[i], self->inst->getVocabDict()));
			if (!doc.rawStr.empty())
			{
				doc.origWordPos.emplace_back(rdoc.origWordPos[i]);
				doc.origWordLen.emplace_back(rdoc.origWordLen[i]);
			}
		}

		if (doc.words.empty())
		{
			fprintf(stderr, "[warn] Adding empty document was ignored.\n");
			continue;
		}

		if (!doc.rawStr.empty()) char2Byte(doc.rawStr, doc.origWordPos, doc.origWordLen);
		auto miscConverter = ((TopicModelTypeObject*)self->ob_base.ob_type)->miscConverter;
		if (!miscConverter) doc.misc.clear();
		else doc.misc = miscConverter(transformMisc(rdoc.misc, transform));
		ret.emplace_back(self->inst->addDoc(doc));
	}
	return ret;
}

CorpusObject* makeCorpus(TopicModelObject* self, PyObject* _corpus, PyObject* transform)
{
	if (!_corpus || _corpus == Py_None) return nullptr;
	if (!PyObject_TypeCheck(_corpus, &UtilsCorpus_type)) throw runtime_error{ "`corpus` must be an instance of `tomotopy.utils.Corpus`" };
	auto corpus = (CorpusObject*)_corpus;
	py::UniqueObj _corpusMade{ PyObject_CallFunctionObjArgs((PyObject*)&UtilsCorpus_type, (PyObject*)self, nullptr) };
	CorpusObject* corpusMade = (CorpusObject*)_corpusMade.get();
	corpusMade->made = true;
	for (auto& rdoc : corpus->docs)
	{
		tomoto::RawDoc doc;
		doc.rawStr = rdoc.rawStr;
		doc.weight = rdoc.weight;
		doc.docUid = rdoc.docUid;

		for (size_t i = 0; i < rdoc.words.size(); ++i)
		{
			if (rdoc.words[i] == tomoto::non_vocab_id) continue;
			tomoto::Vid w = corpus->getVocabDict().mapToNewDict(rdoc.words[i], self->inst->getVocabDict());
			if (w == tomoto::non_vocab_id) continue;
			doc.words.emplace_back(w);

			if (!doc.rawStr.empty())
			{
				doc.origWordPos.emplace_back(rdoc.origWordPos[i]);
				doc.origWordLen.emplace_back(rdoc.origWordLen[i]);
			}
		}

		if (doc.words.empty())
		{
			fprintf(stderr, "[warn] Adding empty document was ignored.\n");
			continue;
		}

		if (!doc.rawStr.empty()) char2Byte(doc.rawStr, doc.origWordPos, doc.origWordLen);
		auto miscConverter = ((TopicModelTypeObject*)self->ob_base.ob_type)->miscConverter;
		if (!miscConverter) doc.misc.clear();
		else doc.misc = miscConverter(transformMisc(rdoc.misc, transform));
		corpusMade->docsMade.emplace_back(self->inst->makeDoc(doc));
	}
	_corpusMade.release();
	return corpusMade;
}

void addUtilsTypes(PyObject* gModule)
{
	if (PyType_Ready(&UtilsCorpus_type) < 0) throw runtime_error{ "UtilsCorpus_type is not ready." };
	Py_INCREF(&UtilsCorpus_type);
	PyModule_AddObject(gModule, "_UtilsCorpus", (PyObject*)&UtilsCorpus_type);

	if (PyType_Ready(&UtilsCorpusIter_type) < 0) throw runtime_error{ "UtilsCorpusIter_type is not ready." };
	Py_INCREF(&UtilsCorpusIter_type);
	PyModule_AddObject(gModule, "_UtilsCorpusIter", (PyObject*)&UtilsCorpusIter_type);

	if (PyType_Ready(&UtilsDocument_type) < 0) throw runtime_error{ "UtilsDocument_type is not ready." };
	Py_INCREF(&UtilsDocument_type);
	PyModule_AddObject(gModule, "Document", (PyObject*)&UtilsDocument_type);

	if (PyType_Ready(&UtilsVocab_type) < 0) throw runtime_error{ "UtilsVocab_type is not ready." };
	Py_INCREF(&UtilsVocab_type);
	PyModule_AddObject(gModule, "_UtilsVocabDict", (PyObject*)&UtilsVocab_type);
}
