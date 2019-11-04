#pragma once
#include "LDA.h"

namespace tomoto
{
    template<TermWeight _TW>
	struct DocumentMGLDA : public DocumentLDA<_TW>
	{
		using DocumentLDA<_TW>::DocumentLDA;
		using WeightType = typename DocumentLDA<_TW>::WeightType;

		std::vector<uint16_t> sents; // sentence id of each word (const)
		std::vector<WeightType> numBySent; // number of words in the sentence (const)

		//std::vector<TID> Zs; // gl./loc. and topic assignment
		std::vector<uint8_t> Vs; // window assignment
		WeightType numGl = 0; // number of words assigned as gl.
		//std::vector<uint32_t> numByTopic; // len = K + KL
		Eigen::Matrix<WeightType, -1, -1> numBySentWin; // len = S * T
		Eigen::Matrix<WeightType, -1, 1> numByWinL; // number of words assigned as loc. in the window (len = S + T - 1)
		Eigen::Matrix<WeightType, -1, 1> numByWin; // number of words in the window (len = S + T - 1)
		Eigen::Matrix<WeightType, -1, -1> numByWinTopicL; // number of words in the loc. topic in the window (len = KL * (S + T - 1))

		DEFINE_SERIALIZER_AFTER_BASE(DocumentLDA<_TW>, sents, Vs, numGl, numBySentWin, numByWinL, numByWin, numByWinTopicL);

		template<typename _TopicModel> void update(WeightType* ptr, const _TopicModel& mdl);
	};

	class IMGLDAModel : public ILDAModel
	{
	public:
		using DefaultDocType = DocumentMGLDA<TermWeight::one>;
		static IMGLDAModel* create(TermWeight _weight, size_t _KG = 1, size_t _KL = 1, size_t _T = 3,
			FLOAT _alphaG = 0.1, FLOAT _alphaL = 0.1, FLOAT _alphaMG = 0.1, FLOAT _alphaML = 0.1,
			FLOAT _etaG = 0.01, FLOAT _etaL = 0.01, FLOAT _gamma = 0.1, const RandGen& _rg = RandGen{ std::random_device{}() });

		virtual size_t addDoc(const std::vector<std::string>& words, const std::string& delimiter) = 0;
		virtual std::unique_ptr<DocumentBase> makeDoc(const std::vector<std::string>& words, const std::string& delimiter) const = 0;

		virtual size_t getKL() const = 0;
		virtual size_t getT() const = 0;
		virtual FLOAT getAlphaL() const = 0;
		virtual FLOAT getEtaL() const = 0;
		virtual FLOAT getGamma() const = 0;
		virtual FLOAT getAlphaM() const = 0;
		virtual FLOAT getAlphaML() const = 0;
	};
}