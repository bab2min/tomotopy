#include "../TopicModel/GDMR.h"

#include "module.h"
#include "utils.h"

using namespace std;

tomoto::RawDoc::MiscType GDMR_misc_args(TopicModelObject* self, const tomoto::RawDoc::MiscType& o)
{
	tomoto::RawDoc::MiscType ret;
	ret["metadata"] = getValueFromMiscDefault<string>("metadata", o, 
		"Since version 0.11.0, `GDMRModel` requires a `metadata` value in `str` type. You can store numerical metadata to a `numeric_metadata` argument."
	);
	ret["numeric_metadata"] = getValueFromMiscDefault<vector<tomoto::Float>>("numeric_metadata", o, 
		"`GDMRModel` requires a `numeric_metadata` value in `Iterable[float]` type."
	);
	return ret;
}

static int GDMR_init(TopicModelObject *self, PyObject *args, PyObject *kwargs)
{
	size_t tw = 0, minCnt = 0, minDf = 0, rmTop = 0;
	tomoto::GDMRArgs margs;
	PyObject* objCorpus = nullptr, *objTransform = nullptr, 
		*objDegrees = nullptr, *objRange = nullptr;
	PyObject* objAlpha = nullptr, *objSeed = nullptr;
	static const char* kwlist[] = { "tw", "min_cf", "min_df", "rm_top", "k", 
		"degrees", "alpha", "eta", "sigma", "sigma0", "alpha_epsilon", 
		"decay", "metadata_range", "seed", "corpus", "transform", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnnnnOOfffffOOOO", (char**)kwlist, 
		&tw, &minCnt, &minDf, &rmTop, &margs.k, 
		&objDegrees, &objAlpha, &margs.eta, &margs.sigma, &margs.sigma0, &margs.alphaEps, 
		&margs.orderDecay, &objRange, &objSeed, &objCorpus, &objTransform)) return -1;
	return py::handleExc([&]()
	{
		if (objAlpha) margs.alpha = broadcastObj<tomoto::Float>(objAlpha, margs.k,
			[=]() { return "`alpha` must be an instance of `float` or `List[float]` with length `k` (given " + py::repr(objAlpha) + ")"; }
		);
		if (objSeed) margs.seed = py::toCpp<size_t>(objSeed, "`seed` must be an integer or None.");

		if (objDegrees)
		{
			margs.degrees = py::toCpp<vector<uint64_t>>(objDegrees, "`degrees` must be an iterable of int.");
		}

		tomoto::IGDMRModel* inst = tomoto::IGDMRModel::create((tomoto::TermWeight)tw, margs);
		if (!inst) throw py::ValueError{ "unknown `tw` value" };
		self->inst = inst;
		self->isPrepared = false;
		self->seedGiven = !!objSeed;
		self->minWordCnt = minCnt;
		self->minWordDf = minDf;
		self->removeTopWord = rmTop;
		self->initParams = py::buildPyDict(kwlist,
			tw, minCnt, minDf, rmTop,
			margs.k, margs.degrees, margs.alpha, margs.eta, margs.sigma, margs.sigma0, margs.alphaEps,
			margs.orderDecay
		);
		py::setPyDictItem(self->initParams, "version", getVersion());

		if (objRange && objRange != Py_None)
		{
			vector<tomoto::Float> vMin, vMax;
			py::UniqueObj rangeIter{ PyObject_GetIter(objRange) }, item;
			if (!rangeIter) throw py::ValueError{ "`metadata_range` must be a list of pairs." };
			while (item = py::UniqueObj{ PyIter_Next(rangeIter) })
			{
				auto r = py::toCpp<vector<tomoto::Float>>(item, "`metadata_range` must be a list of pairs.");
				if (r.size() != 2) throw py::ValueError{ "`metadata_range` must be a list of pairs." };
				vMin.emplace_back(r[0]);
				vMax.emplace_back(r[1]);
			}
			if (vMin.size() != margs.degrees.size()) throw py::ValueError{ "`len(metadata_range)` must be equal to `len(degrees)`" };

			inst->setMdRange(vMin, vMax);
		}

		insertCorpus(self, objCorpus, objTransform);
		return 0;
	});
}

static PyObject* GDMR_addDoc(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject* argWords, *argNumMetadata = nullptr;
	const char* metadata = nullptr;
	size_t ignoreEmptyWords = 1;
	static const char* kwlist[] = { "words", "numeric_metadata", "metadata", "ignore_empty_words", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Ozp", (char**)kwlist, &argWords, &argNumMetadata, &metadata, &ignoreEmptyWords)) return nullptr;
	return py::handleExc([&]() -> PyObject*
	{
		if (!self->inst) throw py::RuntimeError{ "inst is null" };
		if (self->isPrepared) throw py::RuntimeError{ "cannot add_doc() after train()" };
		auto* inst = static_cast<tomoto::IGDMRModel*>(self->inst);
		if (PyUnicode_Check(argWords))
		{
			if (PyErr_WarnEx(PyExc_RuntimeWarning, "`words` should be an iterable of str.", 1)) return nullptr;
		}

		if (!metadata) metadata = "";

		tomoto::RawDoc raw = buildRawDoc(argWords);
		raw.misc["metadata"] = metadata;

		auto nmd = py::toCpp<vector<tomoto::Float>>(argNumMetadata, "`numeric_metadata` must be an iterable of float.");
		for (auto x : nmd)
		{
			if (!isfinite(x)) throw py::ValueError{ "`numeric_metadata` has non-finite value (" + py::reprFromCpp(nmd) + ")." };
		}
		raw.misc["numeric_metadata"] = move(nmd);
		try
		{
			auto ret = inst->addDoc(raw);
			return py::buildPyValue(ret);
		}
		catch (const tomoto::exc::EmptyWordArgument&)
		{
			if (ignoreEmptyWords)
			{
				Py_INCREF(Py_None);
				return Py_None;
			}
			else
			{
				throw;
			}
		}
	});
}

static DocumentObject* GDMR_makeDoc(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject* argWords, * argNumMetadata = nullptr;
	const char* metadata = "";
	static const char* kwlist[] = { "words", "numeric_metadata", "metadata", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Oz", (char**)kwlist, &argWords, &argNumMetadata, &metadata)) return nullptr;
	return py::handleExc([&]() -> DocumentObject*
	{
		if (!self->inst) throw py::RuntimeError{ "inst is null" };
		if (!self->isPrepared) throw py::RuntimeError{ "`train()` should be called before `make_doc()`." };
		auto* inst = static_cast<tomoto::IDMRModel*>(self->inst);
		if (PyUnicode_Check(argWords))
		{
			if (PyErr_WarnEx(PyExc_RuntimeWarning, "`words` should be an iterable of str.", 1)) return nullptr;
		}

		if (!metadata) metadata = "";

		tomoto::RawDoc raw = buildRawDoc(argWords);
		raw.misc["metadata"] = metadata;

		auto nmd = py::toCpp<vector<tomoto::Float>>(argNumMetadata, "`numeric_metadata` must be an iterable of float.");
		for (auto x : nmd)
		{
			if (!isfinite(x)) throw py::ValueError{ "`numeric_metadata` has non-finite value (" + py::reprFromCpp(nmd) + ")." };
		}
		raw.misc["numeric_metadata"] = move(nmd);

		auto doc = inst->makeDoc(raw);
		py::UniqueObj corpus{ PyObject_CallFunctionObjArgs((PyObject*)&UtilsCorpus_type, (PyObject*)self, nullptr) };
		auto* ret = (DocumentObject*)PyObject_CallFunctionObjArgs((PyObject*)&UtilsDocument_type, corpus.get(), nullptr);
		ret->doc = doc.release();
		ret->owner = true;
		return ret;
	});
}

static PyObject* GDMR_tdf(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argNumMetadata = nullptr;
	PyObject* multiMetadata = nullptr;
	const char* metadata = "";
	int normalize = 1;
	static const char* kwlist[] = { "numeric_metadata", "metadata", "multi_metadata", "normalize", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|zOp", (char**)kwlist, &argNumMetadata, &metadata, &multiMetadata, &normalize)) return nullptr;
	return py::handleExc([&]() -> PyObject*
	{
		if (!self->inst) throw py::RuntimeError{ "inst is null" };
		auto* inst = static_cast<tomoto::IGDMRModel*>(self->inst);

		auto v = py::toCpp<vector<tomoto::Float>>(argNumMetadata, "`numeric_metadata` must be an iterable of float.");
		if (v.size() != inst->getFs().size()) throw py::ValueError{ "`len(numeric_metadata)` must be equal to `len(degree).`" };
		
		try
		{
			return py::buildPyValue(inst->getTDF(v.data(), metadata, {}, !!normalize));
		}
		catch (const tomoto::exc::InvalidArgument& e)
		{
			throw py::ValueError{ e.what() };
		}
	});
}

static PyObject* GDMR_tdfLinspace(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argMetadataStart = nullptr, *argMetadataStop = nullptr, *argNum = nullptr;
	PyObject* multiMetadata = nullptr;
	const char* metadata = "";
	size_t endpoint = 1, normalize = 1;
	static const char* kwlist[] = { "numeric_metadata_start", "numeric_metadata_stop", "num", "metadata", "multi_metadata", "endpoint", "normalize", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|zOpp", (char**)kwlist, 
		&argMetadataStart, &argMetadataStop, &argNum, &metadata, &multiMetadata, &endpoint, &normalize)) return nullptr;
	return py::handleExc([&]() -> PyObject*
	{
		if (!self->inst) throw py::RuntimeError{ "inst is null" };
		auto* inst = static_cast<tomoto::IGDMRModel*>(self->inst);
		auto start = py::toCpp<vector<tomoto::Float>>(argMetadataStart, "`metadata_start` must be an iterable of float.");
		if (start.size() != inst->getFs().size()) throw py::ValueError{ "`len(metadata_start)` must be equal to `len(degree).`" };

		auto stop = py::toCpp<vector<tomoto::Float>>(argMetadataStop, "`metadata_stop` must be an iterable of float.");
		if (stop.size() != inst->getFs().size()) throw py::ValueError{ "`len(metadata_stop)` must be equal to `len(degree).`" };

		auto num = py::toCpp<vector<npy_intp>>(argNum, "`num` must be an iterable of float.");
		if (num.size() != inst->getFs().size()) throw py::ValueError{ "`len(num)` must be equal to `len(degree).`" };

		std::ptrdiff_t tot = 1;
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
					if (j) idcs[j - 1]++;
				}
				else break;
			}
		}

		try
		{
			py::UniqueObj obj{ py::buildPyValue(inst->getTDFBatch(mds.data(), metadata, {}, num.size(), tot, !!normalize)) };
			PyArray_Dims dims;
			num.emplace_back(inst->getK());
			dims.ptr = num.data();
			dims.len = num.size();
			return PyArray_Newshape((PyArrayObject*)obj.get(), &dims, NPY_CORDER);
		}
		catch (const tomoto::exc::InvalidArgument& e)
		{
			throw py::ValueError{ e.what() };
		}
	});
}

static PyObject* GDMR_getMetadataRange(TopicModelObject* self, void* closure)
{
	return py::handleExc([&]()
	{
		if (!self->inst) throw py::RuntimeError{ "inst is null" };
		auto* inst = static_cast<tomoto::IGDMRModel*>(self->inst);
		vector<float> vMin, vMax;
		inst->getMdRange(vMin, vMax);
		vector<pair<float, float>> ret;
		for (size_t i = 0; i < vMin.size(); ++i)
		{
			ret.emplace_back(vMin[i], vMax[i]);
		}
		return py::buildPyValue(ret);
	});
}


static PyObject* GDMR_getTopicPrior(TopicModelObject* self, PyObject* args, PyObject* kwargs)
{
	return py::handleExc([&]() -> PyObject*
	{
		throw py::RuntimeError{ "GDMRModel doesn't support get_topic_prior(). Use tdf() instead." };
	});
}


DEFINE_GETTER(tomoto::IGDMRModel, GDMR, getSigma0);
DEFINE_GETTER(tomoto::IGDMRModel, GDMR, getOrderDecay);
DEFINE_GETTER(tomoto::IGDMRModel, GDMR, getFs);

DEFINE_DOCUMENT_GETTER(tomoto::DocumentGDMR, numericMetadata, metadataOrg);

DEFINE_LOADER(GDMR, GDMR_type);

static PyMethodDef GDMR_methods[] =
{
	{ "add_doc", (PyCFunction)GDMR_addDoc, METH_VARARGS | METH_KEYWORDS, GDMR_add_doc__doc__ },
	{ "make_doc", (PyCFunction)GDMR_makeDoc, METH_VARARGS | METH_KEYWORDS, GDMR_make_doc__doc__ },
	{ "load", (PyCFunction)GDMR_load, METH_STATIC | METH_VARARGS | METH_KEYWORDS, LDA_load__doc__ },
	{ "loads", (PyCFunction)GDMR_loads, METH_STATIC | METH_VARARGS | METH_KEYWORDS, LDA_loads__doc__ },
	{ "get_topic_prior", (PyCFunction)GDMR_getTopicPrior, METH_VARARGS | METH_KEYWORDS, DMR_get_topic_prior__doc__ },
	{ "tdf", (PyCFunction)GDMR_tdf, METH_VARARGS | METH_KEYWORDS, GDMR_tdf__doc__ },
	{ "tdf_linspace", (PyCFunction)GDMR_tdfLinspace, METH_VARARGS | METH_KEYWORDS, GDMR_tdf_linspace__doc__ },
	{ nullptr }
};

static PyGetSetDef GDMR_getseters[] = {
	{ (char*)"degrees", (getter)GDMR_getFs, nullptr, GDMR_degrees__doc__, nullptr },
	{ (char*)"sigma0", (getter)GDMR_getSigma0, nullptr, GDMR_sigma0__doc__, nullptr },
	{ (char*)"decay", (getter)GDMR_getOrderDecay, nullptr, GDMR_decay__doc__, nullptr },
	{ (char*)"metadata_range", (getter)GDMR_getMetadataRange, nullptr, GDMR_metadata_range__doc__, nullptr },
	{ nullptr },
};

TopicModelTypeObject GDMR_type = { {
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
}, GDMR_misc_args };
