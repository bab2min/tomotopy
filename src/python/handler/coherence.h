#pragma once

#include "module.h"
#include "utils.h"
#include "../Coherence/CoherenceModel.hpp"

struct CoherenceObject
{
	using ProbEstimation = tomoto::coherence::ProbEstimation;
	using Segmentation = tomoto::coherence::Segmentation;
	using ConfirmMeasure = tomoto::coherence::ConfirmMeasure;
	using IndirectMeasure = tomoto::coherence::IndirectMeasure;

	PyObject_HEAD;
	CorpusObject* corpus;
	Segmentation seg;
	union { tomoto::coherence::CoherenceModel model; };
	union { tomoto::coherence::AnyConfirmMeasurer cm; };
	static int init(CoherenceObject* self, PyObject* args, PyObject* kwargs);
	static PyObject* repr(CoherenceObject* self);
	static void dealloc(CoherenceObject* self);

	static PyObject* getScore(CoherenceObject* self, PyObject* args, PyObject* kwargs);
};

void addCoherenceTypes(PyObject* gModule);
