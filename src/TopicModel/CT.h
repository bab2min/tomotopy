#pragma once
#include "LDA.h"

namespace tomoto
{
    template<TermWeight _tw, size_t _Flags = 0>
	struct DocumentCTM : public DocumentLDA<_tw, _Flags>
	{
		using BaseDocument = DocumentLDA<_tw, _Flags>;
		using DocumentLDA<_tw, _Flags>::DocumentLDA;
		Eigen::Matrix<Float, -1, -1> beta; // Dim: (K, betaSample)
		Eigen::Matrix<Float, -1, 1> smBeta; // Dim: K
		DEFINE_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseDocument, 0, smBeta);
		DEFINE_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseDocument, 1, 0x00010001, smBeta);
	};

	class ICTModel : public ILDAModel
	{
	public:
		using DefaultDocType = DocumentCTM<TermWeight::one>;
		static ICTModel* create(TermWeight _weight, size_t _K = 1,
			Float smoothingAlpha = 0.1,  Float _eta = 0.01,
			const RandGen& _rg = RandGen{ std::random_device{}() });

		virtual void setNumBetaSample(size_t numSample) = 0;
		virtual size_t getNumBetaSample() const = 0;
		virtual void setNumTMNSample(size_t numSample) = 0;
		virtual size_t getNumTMNSample() const = 0;
		virtual void setNumDocBetaSample(size_t numSample) = 0;
		virtual size_t getNumDocBetaSample() const = 0;
		virtual std::vector<Float> getPriorMean() const = 0;
		virtual std::vector<Float> getPriorCov() const = 0;
		virtual std::vector<Float> getCorrelationTopic(Tid k) const = 0;
	};
}
