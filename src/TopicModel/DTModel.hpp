#pragma once
#include "LDAModel.hpp"
#include "DTM.h"

/*
Implementation of Dynamic Topic Model using Gibbs sampling by bab2min

* Blei, D. M., & Lafferty, J. D. (2006, June). Dynamic topic models. In Proceedings of the 23rd international conference on Machine learning (pp. 113-120).
* Bhadury, A., Chen, J., Zhu, J., & Liu, S. (2016, April). Scaling up dynamic topic models. In Proceedings of the 25th International Conference on World Wide Web (pp. 381-390).

*/

namespace tomoto
{
	template<TermWeight _tw>
	struct ModelStateDTM
	{
		using WeightType = typename std::conditional<_tw == TermWeight::one, int32_t, float>::type;

		Eigen::Matrix<Float, -1, 1> zLikelihood;
		Eigen::Matrix<WeightType, -1, -1> numByTopic; // Dim: (Topic, T)
		Eigen::Matrix<WeightType, -1, -1> numByTopicWord; // Dim: (Topic, Vocabs * T)
		DEFINE_SERIALIZER(numByTopic, numByTopicWord);
	};

	template<TermWeight _tw, size_t _Flags = flags::partitioned_multisampling,
		typename _Interface = IDTModel,
		typename _Derived = void,
		typename _DocType = DocumentDTM<_tw>,
		typename _ModelState = ModelStateDTM<_tw>>
		class DTModel : public LDAModel<_tw, _Flags, _Interface,
		typename std::conditional<std::is_same<_Derived, void>::value, DTModel<_tw, _Flags>, _Derived>::type,
		_DocType, _ModelState>
	{
	protected:
		using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, DTModel<_tw>, _Derived>::type;
		using BaseClass = LDAModel<_tw, _Flags, _Interface, DerivedClass, _DocType, _ModelState>;
		friend BaseClass;
		friend typename BaseClass::BaseClass;
		using WeightType = typename BaseClass::WeightType;

		const char *TMID = "DTM\0";

		size_t T;

		Eigen::Matrix<Float, -1, -1> alphas; // Dim: (Topic, Time)
		Eigen::Matrix<Float, -1, -1> dtm_eta; // Dim: (Docs, Topic)
		Eigen::Matrix<Float, -1, -1> dtm_phi; // Dim: (Topic, Word * Time)
	};
}