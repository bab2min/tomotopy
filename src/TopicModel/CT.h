#pragma once
#include "LDA.h"

namespace tomoto
{
    template<TermWeight _TW, size_t _Flags = 0>
	struct DocumentCTM : public DocumentLDA<_TW, _Flags>
	{
		using DocumentLDA<_TW, _Flags>::DocumentLDA;
		Eigen::Matrix<FLOAT, -1, -1> beta; // Dim: (K, betaSample)
		Eigen::Matrix<FLOAT, -1, 1> smBeta; // Dim: K
		DEFINE_SERIALIZER_AFTER_BASE2(DocumentLDA<_TW, _Flags>, smBeta);
	};

	class ICTModel : public ILDAModel
	{
	public:
		using DefaultDocType = DocumentCTM<TermWeight::one>;
		static ICTModel* create(TermWeight _weight, size_t _K = 1,
			FLOAT smoothingAlpha = 0.1,  FLOAT _eta = 0.01,
			const RandGen& _rg = RandGen{ std::random_device{}() });

		virtual void setNumBetaSample(size_t numSample) = 0;
		virtual size_t getNumBetaSample() const = 0;
		virtual void setNumTMNSample(size_t numSample) = 0;
		virtual size_t getNumTMNSample() const = 0;
		virtual void setNumDocBetaSample(size_t numSample) = 0;
		virtual size_t getNumDocBetaSample() const = 0;
		virtual std::vector<FLOAT> getPriorMean() const = 0;
		virtual std::vector<FLOAT> getPriorCov() const = 0;
		virtual std::vector<FLOAT> getCorrelationTopic(TID k) const = 0;
	};
}
