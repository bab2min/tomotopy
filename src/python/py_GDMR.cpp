#include "../TopicModel/GDMR.h"

#include "module.h"

using namespace std;

static int GDMR_init(TopicModelObject *self, PyObject *args, PyObject *kwargs)
{
	size_t tw = 0, minCnt = 0, minDf = 0, rmTop = 0;
	size_t K = 1;
	float alpha = 0.1, eta = 0.01, sigma = 1, sigma0 = 3, alphaEpsilon = 1e-10;
	size_t seed = random_device{}();
	PyObject* objCorpus = nullptr, *objTransform = nullptr, 
		*objDegrees = nullptr, *objRange = nullptr;
	static const char* kwlist[] = { "tw", "min_cf", "min_df", "rm_top", "k", 
		"degrees", "alpha", "eta", "sigma", "sigma0", "alpha_epsilon", 
		"metadata_range", "seed", "corpus", "transform", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnnnnOfffffOnOO", (char**)kwlist, 
		&tw, &minCnt, &minDf, &rmTop,&K, 
		&objDegrees, &alpha, &eta, &sigma, &sigma0, &alphaEpsilon, 
		&objRange, &seed, &objCorpus, &objTransform)) return -1;
	try
	{
		if (objCorpus && !PyObject_HasAttrString(objCorpus, corpus_feeder_name))
		{
			throw runtime_error{ "`corpus` must be `tomotopy.utils.Corpus` type." };
		}

		vector<uint64_t> degrees;
		if (objDegrees)
		{
			py::UniqueObj degreeIter = PyObject_GetIter(objDegrees);
			if (!degreeIter)
			{
				throw runtime_error{ "`degrees` must be an iterable of int." };
			}

			degrees = py::makeIterToVector<uint64_t>(degreeIter);
		}

		tomoto::IGDMRModel* inst = tomoto::IGDMRModel::create((tomoto::TermWeight)tw, K,
			degrees, alpha, sigma, sigma0, eta, alphaEpsilon, tomoto::RandGen{ seed });
		if (!inst) throw runtime_error{ "unknown tw value" };
		self->inst = inst;
		self->isPrepared = false;
		self->minWordCnt = minCnt;
		self->minWordDf = minDf;
		self->removeTopWord = rmTop;

		if (objRange && objRange != Py_None)
		{
			vector<float> vMin, vMax;
			py::UniqueObj rangeIter = PyObject_GetIter(objRange), item;
			if(!rangeIter) throw runtime_error{ "`metadata_range` must be a list of pairs." };
			while (item = PyIter_Next(rangeIter))
			{
				item = PyObject_GetIter(item);
				auto r = py::makeIterToVector<float>(item);
				if (r.size() != 2) throw runtime_error{ "`metadata_range` must be a list of pairs." };
				vMin.emplace_back(r[0]);
				vMax.emplace_back(r[1]);
			}
			if(vMin.size() != degrees.size()) throw runtime_error{ "`len(metadata_range)` must be equal to `len(degrees)`" };

			inst->setMdRange(vMin, vMax);
		}

		if (objCorpus)
		{
			py::UniqueObj feeder = PyObject_GetAttrString(objCorpus, corpus_feeder_name),
				param = Py_BuildValue("(OO)", self, objTransform ? objTransform : Py_None);
			py::UniqueObj ret = PyObject_CallObject(feeder, param);
			if (!ret) return -1;
		}
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return -1;
	}
	return 0;
}

static PyObject* GDMR_addDoc(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords, *argMetadata = nullptr;
	static const char* kwlist[] = { "words", "metadata", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", (char**)kwlist, &argWords, &argMetadata)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		if (self->isPrepared) throw runtime_error{ "cannot add_doc() after train()" };
		auto* inst = static_cast<tomoto::IGDMRModel*>(self->inst);
		if (PyUnicode_Check(argWords)) PRINT_WARN("[warn] 'words' should be an iterable of str.");
		py::UniqueObj iter, iterMetadata;
		if (!(iter = PyObject_GetIter(argWords)))
		{
			throw runtime_error{ "words must be an iterable of str." };
		}

		if (!argMetadata || !(iterMetadata = PyObject_GetIter(argMetadata)))
		{
			throw runtime_error{ "`metadata` must be an iterable of float." };
		}

		auto ret = inst->addDoc(py::makeIterToVector<string>(iter), py::makeIterToStringVector<float>(iterMetadata));
		return py::buildPyValue(ret);
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

static PyObject* GDMR_addDoc_(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords, *argStartPos = nullptr, *argLength = nullptr, *argMetadata = nullptr;
	const char* argRaw = nullptr;
	static const char* kwlist[] = { "words", "raw", "start_pos", "length", "metadata", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|sOOO", (char**)kwlist,
		&argWords, &argRaw, &argStartPos, &argLength, &argMetadata)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IGDMRModel*>(self->inst);
		string raw;
		if (argRaw) raw = argRaw;
		if (argRaw && (!argStartPos || !argLength))
		{
			throw runtime_error{ "`start_pos` and `length` must be given when `raw` is given." };
		}

		vector<tomoto::Vid> words;
		vector<uint32_t> startPos;
		vector<uint16_t> length;

		py::UniqueObj iter = PyObject_GetIter(argWords), iterMetadata;
		words = py::makeIterToVector<tomoto::Vid>(iter);
		if (argStartPos)
		{
			iter = PyObject_GetIter(argStartPos);
			startPos = py::makeIterToVector<uint32_t>(iter);
			iter = PyObject_GetIter(argLength);
			length = py::makeIterToVector<uint16_t>(iter);
			char2Byte(raw, startPos, length);
		}

		if (!argMetadata || !(iterMetadata = PyObject_GetIter(argMetadata)))
		{
			throw runtime_error{ "`metadata` must be an iterable of float." };
		}

		auto ret = inst->addDoc(raw, words, startPos, length, py::makeIterToStringVector<float>(iterMetadata));
		return py::buildPyValue(ret);
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

static PyObject* GDMR_makeDoc(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords, *argMetadata = nullptr;
	static const char* kwlist[] = { "words", "metadata", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", (char**)kwlist, &argWords, &argMetadata)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IDMRModel*>(self->inst);
		if (PyUnicode_Check(argWords)) PRINT_WARN("[warn] 'words' should be an iterable of str.");
		py::UniqueObj iter, iterMetadata;
		if (!(iter = PyObject_GetIter(argWords)))
		{
			throw runtime_error{ "words must be an iterable of str." };
		}

		if (!argMetadata || !(iterMetadata = PyObject_GetIter(argMetadata)))
		{
			throw runtime_error{ "`metadata` must be an iterable of float." };
		}

		auto ret = inst->makeDoc(py::makeIterToVector<string>(iter), py::makeIterToStringVector<float>(iterMetadata));
		py::UniqueObj args = Py_BuildValue("(Onn)", self, ret.release(), 1);
		return PyObject_CallObject((PyObject*)&Document_type, args);
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

static PyObject* GDMR_tdf(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argMetadata = nullptr;
	int normalize = 1;
	static const char* kwlist[] = { "metadata", "normalize", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p", (char**)kwlist, &argMetadata, &normalize)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IGDMRModel*>(self->inst);
		py::UniqueObj iterMetadata;

		if (!(iterMetadata = PyObject_GetIter(argMetadata)))
		{
			throw runtime_error{ "`metadata` must be an iterable of float." };
		}
		auto v = py::makeIterToVector<float>(iterMetadata);
		if (v.size() != inst->getFs().size()) throw runtime_error{ "`len(metadata)` must be equal to `len(degree).`" };
		return py::buildPyValue(inst->getTDF(v.data(), !!normalize));
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

static PyObject* GDMR_tdfLinspace(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argMetadataStart = nullptr, *argMetadataStop = nullptr, *argNum = nullptr;
	size_t endpoint = 1, normalize = 1;
	static const char* kwlist[] = { "metadata_start", "metadata_stop", "num", "endpoint", "normalize", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|pp", (char**)kwlist, 
		&argMetadataStart, &argMetadataStop, &argNum, &endpoint, &normalize)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IGDMRModel*>(self->inst);
		py::UniqueObj iterMetadataStart, iterMetadataStop, iterNum;

		if (!(iterMetadataStart = PyObject_GetIter(argMetadataStart)))
		{
			throw runtime_error{ "`metadata_start` must be an iterable of float." };
		}
		auto start = py::makeIterToVector<float>(iterMetadataStart);
		if (start.size() != inst->getFs().size()) throw runtime_error{ "`len(metadata_start)` must be equal to `len(degree).`" };

		if (!(iterMetadataStop = PyObject_GetIter(argMetadataStop)))
		{
			throw runtime_error{ "`metadata_stop` must be an iterable of float." };
		}
		auto stop = py::makeIterToVector<float>(iterMetadataStop);
		if (stop.size() != inst->getFs().size()) throw runtime_error{ "`len(metadata_stop)` must be equal to `len(degree).`" };

		if (!(iterNum = PyObject_GetIter(argNum)))
		{
			throw runtime_error{ "`num` must be an iterable of float." };
		}
		auto num = py::makeIterToVector<npy_intp>(iterNum);
		if (num.size() != inst->getFs().size()) throw runtime_error{ "`len(num)` must be equal to `len(degree).`" };

		ssize_t tot = 1;
		for (auto& v : num)
		{
			if (v <= 0) v = 1;
			tot *= v;
		}

		Eigen::MatrixXf mds{ (Eigen::Index)num.size(), (Eigen::Index)tot };
		vector<npy_intp> idcs(num.size());
		for (size_t i = 0; i < tot; ++i)
		{
			for (size_t j = 0; j < num.size(); ++j)
			{
				mds(j, i) = start[j] + (stop[j] - start[j]) * idcs[j] / (endpoint ? max(num[j] - 1, (npy_intp)1) : num[j]);
			}

			idcs.back()++;
			for (int j = idcs.size() - 1; j >= 0; --j)
			{
				if (idcs[j] >= num[j])
				{
					idcs[j] = 0;
					if(j) idcs[j - 1]++;
				}
				else break;
			}
		}

		py::UniqueObj obj = py::buildPyValue(inst->getTDFBatch(mds.data(), num.size(), tot, !!normalize));
		PyArray_Dims dims;
		num.emplace_back(inst->getK());
		dims.ptr = num.data();
		dims.len = num.size();
		return PyArray_Newshape((PyArrayObject*)obj.get(), &dims, NPY_CORDER);
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

static PyObject* GDMR_getMetadataRange(TopicModelObject* self, void* closure)
{
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IGDMRModel*>(self->inst);
		vector<float> vMin, vMax;
		inst->getMdRange(vMin, vMax);
		vector<pair<float, float>> ret;
		for (size_t i = 0; i < vMin.size(); ++i)
		{
			ret.emplace_back(vMin[i], vMax[i]);
		}
		return py::buildPyValue(ret);
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

DEFINE_GETTER(tomoto::IGDMRModel, GDMR, getSigma0);
DEFINE_GETTER(tomoto::IGDMRModel, GDMR, getFs);

DEFINE_DOCUMENT_GETTER(tomoto::DocumentGDMR, GDMR_metadata, metadataOrg);

DEFINE_LOADER(GDMR, GDMR_type);

static PyMethodDef GDMR_methods[] =
{
	{ "add_doc", (PyCFunction)GDMR_addDoc, METH_VARARGS | METH_KEYWORDS, GDMR_add_doc__doc__ },
	{ "_add_doc", (PyCFunction)GDMR_addDoc_, METH_VARARGS | METH_KEYWORDS, "" },
	{ "make_doc", (PyCFunction)GDMR_makeDoc, METH_VARARGS | METH_KEYWORDS, GDMR_make_doc__doc__ },
	{ "load", (PyCFunction)GDMR_load, METH_STATIC | METH_VARARGS | METH_KEYWORDS, LDA_load__doc__ },
	{ "tdf", (PyCFunction)GDMR_tdf, METH_VARARGS | METH_KEYWORDS, GDMR_tdf__doc__ },
	{ "tdf_linspace", (PyCFunction)GDMR_tdfLinspace, METH_VARARGS | METH_KEYWORDS, GDMR_tdf_linspace__doc__ },
	{ nullptr }
};

static PyGetSetDef GDMR_getseters[] = {
	{ (char*)"degrees", (getter)GDMR_getFs, nullptr, GDMR_degrees__doc__, nullptr },
	{ (char*)"sigma0", (getter)GDMR_getSigma0, nullptr, GDMR_sigma0__doc__, nullptr },
	{ (char*)"metadata_range", (getter)GDMR_getMetadataRange, nullptr, GDMR_metadata_range__doc__, nullptr },
	{ nullptr },
};


PyTypeObject GDMR_type = {
	PyVarObject_HEAD_INIT(nullptr, 0)
	"tomotopy.GDMRModel",             /* tp_name */
	sizeof(TopicModelObject), /* tp_basicsize */
	0,                         /* tp_itemsize */
	(destructor)TopicModelObject::dealloc, /* tp_dealloc */
	0,                         /* tp_print */
	0,                         /* tp_getattr */
	0,                         /* tp_setattr */
	0,                         /* tp_reserved */
	0,                         /* tp_repr */
	0,                         /* tp_as_number */
	0,                         /* tp_as_sequence */
	0,                         /* tp_as_mapping */
	0,                         /* tp_hash  */
	0,                         /* tp_call */
	0,                         /* tp_str */
	0,                         /* tp_getattro */
	0,                         /* tp_setattro */
	0,                         /* tp_as_buffer */
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
	GDMR___init____doc__,           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	0,                         /* tp_iter */
	0,                         /* tp_iternext */
	GDMR_methods,             /* tp_methods */
	0,						 /* tp_members */
	GDMR_getseters,                         /* tp_getset */
	&DMR_type,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)GDMR_init,      /* tp_init */
	PyType_GenericAlloc,
	PyType_GenericNew,
};
