#pragma once
#include "TopicModel.hpp"

namespace tomoto
{
    enum class TermWeight { one, idf, pmi, size };

	template<typename _Scalar, Eigen::Index _rows, Eigen::Index _cols>
	struct ShareableMatrix : Eigen::Map<Eigen::Matrix<_Scalar, _rows, _cols>>
	{
		using BaseType = Eigen::Map<Eigen::Matrix<_Scalar, _rows, _cols>>;
		Eigen::Matrix<_Scalar, _rows, _cols> ownData;

		ShareableMatrix(_Scalar* ptr = nullptr, Eigen::Index rows = 0, Eigen::Index cols = 0) 
			: BaseType(nullptr, _rows != -1 ? _rows : 0, _cols != -1 ? _cols : 0)
		{
			init(ptr, rows, cols);
		}

		ShareableMatrix(const ShareableMatrix& o)
			: BaseType(nullptr, _rows != -1 ? _rows : 0, _cols != -1 ? _cols : 0), ownData{ o.ownData }
		{
			if (o.ownData.data())
			{
				new (this) BaseType(ownData.data(), ownData.rows(), ownData.cols());
			}
			else
			{
				new (this) BaseType((_Scalar*)o.data(), o.rows(), o.cols());
			}
		}

		ShareableMatrix(ShareableMatrix&& o) = default;

		ShareableMatrix& operator=(const ShareableMatrix& o)
		{
			if (o.ownData.data())
			{
				ownData = o.ownData;
				new (this) BaseType(ownData.data(), ownData.rows(), ownData.cols());
			}
			else
			{
				new (this) BaseType((_Scalar*)o.data(), o.rows(), o.cols());
			}
			return *this;
		}

		ShareableMatrix& operator=(ShareableMatrix&& o) = default;

		void init(_Scalar* ptr, Eigen::Index rows, Eigen::Index cols)
		{
			if (!ptr && rows && cols)
			{
				ownData = Eigen::Matrix<_Scalar, _rows, _cols>::Zero(_rows != -1 ? _rows : rows, _cols != -1 ? _cols : cols);
				ptr = ownData.data();
			}
			else
			{
				ownData = Eigen::Matrix<_Scalar, _rows, _cols>{};
			}
			new (this) BaseType(ptr, _rows != -1 ? _rows : rows, _cols != -1 ? _cols : cols);
		}

		void conservativeResize(size_t newRows, size_t newCols)
		{
			ownData.conservativeResize(_rows != -1 ? _rows : newRows, _cols != -1 ? _cols : newCols);
			new (this) BaseType(ownData.data(), ownData.rows(), ownData.cols());
		}

		void becomeOwner()
		{
			if (ownData.data() != this->m_data)
			{
				ownData = *this;
				new (this) BaseType(ownData.data(), ownData.rows(), ownData.cols());
			}
		}

		void serializerRead(std::istream& istr)
		{
			uint32_t rows = serializer::readFromStream<uint32_t>(istr);
			uint32_t cols = serializer::readFromStream<uint32_t>(istr);
			init(nullptr, rows, cols);
			if (!istr.read((char*)this->data(), sizeof(_Scalar) * this->size()))
				throw std::ios_base::failure(std::string("reading type '") + typeid(_Scalar).name() + std::string("' is failed"));
		}

		void serializerWrite(std::ostream& ostr) const
		{
			serializer::writeToStream<uint32_t>(ostr, (uint32_t)this->rows());
			serializer::writeToStream<uint32_t>(ostr, (uint32_t)this->cols());
			if (!ostr.write((const char*)this->data(), sizeof(_Scalar) * this->size()))
				throw std::ios_base::failure(std::string("writing type '") + typeid(_Scalar).name() + std::string("' is failed"));
		}
	};

	template<typename _Base, TermWeight _tw>
	struct SumWordWeight
	{
		Float sumWordWeight = 0;
		Float getSumWordWeight() const
		{
			return sumWordWeight;
		}

		void updateSumWordWeight(size_t realV)
		{
			sumWordWeight = std::accumulate(static_cast<_Base*>(this)->wordWeights.begin(), static_cast<_Base*>(this)->wordWeights.end(), 0.f);
		}
	};

	template<typename _Base>
	struct SumWordWeight<_Base, TermWeight::one>
	{
		int32_t sumWordWeight = 0;
		int32_t getSumWordWeight() const
		{
			return sumWordWeight;
		}

		void updateSumWordWeight(size_t realV)
		{
			sumWordWeight = (int32_t)std::count_if(static_cast<_Base*>(this)->words.begin(), static_cast<_Base*>(this)->words.end(), [realV](Vid w)
			{
				return w < realV;
			});
		}
	};

	template<TermWeight _tw>
	struct DocumentLDA : public DocumentBase, SumWordWeight<DocumentLDA<_tw>, _tw>
	{
	public:
		using DocumentBase::DocumentBase;
		using WeightType = typename std::conditional<_tw == TermWeight::one, int32_t, float>::type;

		tvector<Tid> Zs;
		tvector<Float> wordWeights;
		ShareableMatrix<WeightType, -1, 1> numByTopic;

		DEFINE_SERIALIZER_AFTER_BASE_WITH_VERSION(DocumentBase, 0, Zs, wordWeights);
		DEFINE_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(DocumentBase, 1, 0x00010001, Zs, wordWeights);

		template<typename _TopicModel> void update(WeightType* ptr, const _TopicModel& mdl);
		
		WeightType getWordWeight(size_t idx) const
		{
			return _tw == TermWeight::one ? 1 : wordWeights[idx];
		}

		std::vector<Float> getCountVector(size_t V) const
		{
			std::vector<Float> vs(V);
			for (size_t i = 0; i < words.size(); ++i)
			{
				if (words[i] >= V) continue;
				vs[words[i]] += wordWeights.empty() ? 1.f : wordWeights[i];
			}
			return vs;
		}
	};

	struct LDAArgs
	{
		size_t k = 1;
		std::vector<Float> alpha = { (Float)0.1 };
		Float eta = (Float)0.01;
		size_t seed = std::random_device{}();
	};

    class ILDAModel : public ITopicModel
	{
	public:
		using DefaultDocType = DocumentLDA<TermWeight::one>;
		static ILDAModel* create(TermWeight _weight, const LDAArgs& args,
			bool scalarRng = false);

		virtual TermWeight getTermWeight() const = 0;
		virtual size_t getOptimInterval() const = 0;
		virtual void setOptimInterval(size_t) = 0;
		virtual size_t getBurnInIteration() const = 0;
		virtual void setBurnInIteration(size_t) = 0;
		virtual std::vector<uint64_t> getCountByTopic() const = 0;
		virtual Float getAlpha() const = 0;
		virtual Float getAlpha(size_t k) const = 0;
		virtual Float getEta() const = 0;

		virtual std::vector<Float> getWordPrior(const std::string& word) const = 0;
		virtual void setWordPrior(const std::string& word, const std::vector<Float>& priors) = 0;
	};
}
