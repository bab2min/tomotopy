#pragma once
#include "LDA.h"

namespace tomoto
{
    template<TermWeight _tw>
	struct DocumentMGLDA : public DocumentLDA<_tw>
	{
		using BaseDocument = DocumentLDA<_tw>;
		using DocumentLDA<_tw>::DocumentLDA;
		using WeightType = typename DocumentLDA<_tw>::WeightType;

		std::vector<uint16_t> sents; // sentence id of each word (const)
		std::vector<WeightType> numBySent; // number of words in the sentence (const)

		//std::vector<Tid> Zs; // gl./loc. and topic assignment
		std::vector<uint8_t> Vs; // window assignment
		WeightType numGl = 0; // number of words assigned as gl.
		//std::vector<uint32_t> numByTopic; // len = K + KL
		Eigen::Matrix<WeightType, -1, -1> numBySentWin; // len = S * T
		Eigen::Matrix<WeightType, -1, 1> numByWinL; // number of words assigned as loc. in the window (len = S + T - 1)
		Eigen::Matrix<WeightType, -1, 1> numByWin; // number of words in the window (len = S + T - 1)
		Eigen::Matrix<WeightType, -1, -1> numByWinTopicL; // number of words in the loc. topic in the window (len = KL * (S + T - 1))

		DEFINE_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseDocument, 0, sents, Vs, numGl, numBySentWin, numByWinL, numByWin, numByWinTopicL);
		DEFINE_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseDocument, 1, 0x00010001, sents, Vs, numGl, numBySentWin, numByWinL, numByWin, numByWinTopicL);

		template<typename _TopicModel> void update(WeightType* ptr, const _TopicModel& mdl);
	};

	class IMGLDAModel : public ILDAModel
	{
	public:
		using DefaultDocType = DocumentMGLDA<TermWeight::one>;
		static IMGLDAModel* create(TermWeight _weight, size_t _KG = 1, size_t _KL = 1, size_t _T = 3,
			Float _alphaG = 0.1, Float _alphaL = 0.1, Float _alphaMG = 0.1, Float _alphaML = 0.1,
			Float _etaG = 0.01, Float _etaL = 0.01, Float _gamma = 0.1, const RandGen& _rg = RandGen{ std::random_device{}() });

		virtual size_t addDoc(const std::vector<std::string>& words, const std::string& delimiter) = 0;
		virtual std::unique_ptr<DocumentBase> makeDoc(const std::vector<std::string>& words, const std::string& delimiter) const = 0;

		virtual size_t addDoc(const std::string& rawStr, const RawDocTokenizer::Factory& tokenizer,
			const std::string& delimiter) = 0;
		virtual std::unique_ptr<DocumentBase> makeDoc(const std::string& rawStr, const RawDocTokenizer::Factory& tokenizer,
			const std::string& delimiter) const = 0;

		virtual size_t addDoc(const std::string& rawStr, const std::vector<Vid>& words,
			const std::vector<uint32_t>& pos, const std::vector<uint16_t>& len,
			const std::string& delimiter) = 0;
		virtual std::unique_ptr<DocumentBase> makeDoc(const std::string& rawStr, const std::vector<Vid>& words,
			const std::vector<uint32_t>& pos, const std::vector<uint16_t>& len,
			const std::string& delimiter) const = 0;

		virtual size_t getKL() const = 0;
		virtual size_t getT() const = 0;
		virtual Float getAlphaL() const = 0;
		virtual Float getEtaL() const = 0;
		virtual Float getGamma() const = 0;
		virtual Float getAlphaM() const = 0;
		virtual Float getAlphaML() const = 0;
	};
}