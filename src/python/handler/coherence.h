#pragma once

#include "module.h"
#include "utils.h"
#include "../../Coherence/CoherenceModel.hpp"

struct CoherenceObject : public py::CObject<CoherenceObject>
{
	using ProbEstimation = tomoto::coherence::ProbEstimation;
	using Segmentation = tomoto::coherence::Segmentation;
	using ConfirmMeasure = tomoto::coherence::ConfirmMeasure;
	using IndirectMeasure = tomoto::coherence::IndirectMeasure;

	py::UniqueCObj<CorpusObject> corpus;
	Segmentation seg;
	tomoto::coherence::CoherenceModel model;
	tomoto::coherence::AnyConfirmMeasurer cm;

	CoherenceObject() = default;
	CoherenceObject(PyObject* corpus, 
		ProbEstimation pe, Segmentation seg, ConfirmMeasure cm, IndirectMeasure im, 
		size_t windowSize, double eps, double gamma, PyObject* targets);

	double getScore(PyObject* words) const;
};

void addCoherenceTypes(py::Module& module);
