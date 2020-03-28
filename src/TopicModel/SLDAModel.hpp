#pragma once
#include "LDAModel.hpp"
#include "../Utils/PolyaGamma.hpp"
#include "SLDA.h"

/*
Implementation of sLDA using Gibbs sampling by bab2min
* Mcauliffe, J. D., & Blei, D. M. (2008). Supervised topic models. In Advances in neural information processing systems (pp. 121-128).
* Python version implementation using Gibbs sampling : https://github.com/Savvysherpa/slda
*/

namespace tomoto
{
	namespace detail
	{
		template<typename _WeightType>
		struct GLMFunctor
		{
			Eigen::Matrix<Float, -1, 1> regressionCoef; // Dim : (K)

			GLMFunctor(size_t K = 0, Float mu = 0) : regressionCoef(Eigen::Matrix<Float, -1, 1>::Constant(K, mu))
			{
			}

			virtual ISLDAModel::GLM getType() const = 0;

			virtual void updateZLL(
				Eigen::Matrix<Float, -1, 1>& zLikelihood,
				Float y, const Eigen::Matrix<_WeightType, -1, 1>& numByTopic, size_t docId, Float docSize) const = 0;

			virtual void optimizeCoef(
				const Eigen::Matrix<Float, -1, -1>& normZ,
				Float mu, Float nuSq,
				Eigen::Block<Eigen::Matrix<Float, -1, -1>, -1, 1, true> ys
			) = 0;

			virtual double getLL(Float y, const Eigen::Matrix<_WeightType, -1, 1>& numByTopic,
				Float docSize) const = 0;

			virtual Float estimate(const Eigen::Matrix<_WeightType, -1, 1>& numByTopic,
				Float docSize) const = 0;

			virtual ~GLMFunctor() {};

			DEFINE_SERIALIZER_VIRTUAL(regressionCoef);

			static void serializerWrite(const std::unique_ptr<GLMFunctor>& p, std::ostream& ostr)
			{
				if (!p) serializer::writeToStream<uint32_t>(ostr, 0);
				else
				{
					serializer::writeToStream<uint32_t>(ostr, (uint32_t)p->getType() + 1);
					p->serializerWrite(ostr);
				}
			}

			static void serializerRead(std::unique_ptr<GLMFunctor>& p, std::istream& istr);
		};

		template<typename _WeightType>
		struct LinearFunctor : public GLMFunctor<_WeightType>
		{
			Float sigmaSq = 1;

			LinearFunctor(size_t K = 0, Float mu = 0, Float _sigmaSq = 1)
				: GLMFunctor<_WeightType>(K, mu), sigmaSq(_sigmaSq)
			{
			}

			ISLDAModel::GLM getType() const override { return ISLDAModel::GLM::linear; }

			void updateZLL(
				Eigen::Matrix<Float, -1, 1>& zLikelihood,
				Float y, const Eigen::Matrix<_WeightType, -1, 1>& numByTopic, size_t docId, Float docSize) const override
			{
				Float yErr = y -
					(this->regressionCoef.array() * numByTopic.array().template cast<Float>()).sum()
					/ docSize;
				zLikelihood.array() *= (this->regressionCoef.array() / docSize / 2 / sigmaSq *
					(2 * yErr - this->regressionCoef.array() / docSize)).exp();
			}

			void optimizeCoef(
				const Eigen::Matrix<Float, -1, -1>& normZ,
				Float mu, Float nuSq,
				Eigen::Block<Eigen::Matrix<Float, -1, -1>, -1, 1, true> ys
			) override
			{
				Eigen::Matrix<Float, -1, -1> selectedNormZ = normZ.array().rowwise() * (!ys.array().transpose().isNaN()).template cast<Float>();
				Eigen::Matrix<Float, -1, -1> normZZT = selectedNormZ * selectedNormZ.transpose();
				normZZT += Eigen::Matrix<Float, -1, -1>::Identity(normZZT.cols(), normZZT.cols()) / nuSq;
				this->regressionCoef = normZZT.colPivHouseholderQr().solve(selectedNormZ * ys.array().isNaN().select(0, ys).matrix());
			}

			double getLL(Float y, const Eigen::Matrix<_WeightType, -1, 1>& numByTopic,
				Float docSize) const override
			{
				Float estimatedY = estimate(numByTopic, docSize);
				return -pow(estimatedY - y, 2) / 2 / sigmaSq;
			}

			Float estimate(const Eigen::Matrix<_WeightType, -1, 1>& numByTopic,
				Float docSize) const override
			{
				return (this->regressionCoef.array() * numByTopic.array().template cast<Float>()).sum()
					/ std::max(docSize, 0.01f);
			}

			DEFINE_SERIALIZER_AFTER_BASE(GLMFunctor<_WeightType>, sigmaSq);
		};

		template<typename _WeightType>
		struct BinaryLogisticFunctor : public GLMFunctor<_WeightType>
		{
			Float b = 1;
			Eigen::Matrix<Float, -1, 1> omega;

			BinaryLogisticFunctor(size_t K = 0, Float mu = 0, Float _b = 1, size_t numDocs = 0)
				: GLMFunctor<_WeightType>(K, mu), b(_b), omega{ Eigen::Matrix<Float, -1, 1>::Ones(numDocs) }
			{
			}

			ISLDAModel::GLM getType() const override { return ISLDAModel::GLM::binary_logistic; }

			void updateZLL(
				Eigen::Matrix<Float, -1, 1>& zLikelihood,
				Float y, const Eigen::Matrix<_WeightType, -1, 1>& numByTopic, size_t docId, Float docSize) const override
			{
				Float yErr = b * (y - 0.5f) -
					(this->regressionCoef.array() * numByTopic.array().template cast<Float>()).sum()
					/ docSize * omega[docId];
				zLikelihood.array() *= (this->regressionCoef.array() / docSize *
					(yErr - omega[docId] / 2 * this->regressionCoef.array() / docSize)).exp();
			}

			void optimizeCoef(
				const Eigen::Matrix<Float, -1, -1>& normZ,
				Float mu, Float nuSq,
				Eigen::Block<Eigen::Matrix<Float, -1, -1>, -1, 1, true> ys
			) override
			{
				Eigen::Matrix<Float, -1, -1> selectedNormZ = normZ.array().rowwise() * (!ys.array().transpose().isNaN()).template cast<Float>();
				Eigen::Matrix<Float, -1, -1> normZZT = selectedNormZ * Eigen::DiagonalMatrix<Float, -1>{ omega } * selectedNormZ.transpose();
				normZZT += Eigen::Matrix<Float, -1, -1>::Identity(normZZT.cols(), normZZT.cols()) / nuSq;

				this->regressionCoef = normZZT
					.colPivHouseholderQr().solve(selectedNormZ * ys.array().isNaN().select(0, b * (ys.array() - 0.5f)).matrix()
						+ Eigen::Matrix<Float, -1, 1>::Constant(selectedNormZ.rows(), mu / nuSq));

				RandGen rng;
				for (size_t i = 0; i < omega.size(); ++i)
				{
					if (std::isnan(ys[i])) continue;
					omega[i] = math::drawPolyaGamma(b, (this->regressionCoef.array() * normZ.col(i).array()).sum(), rng);
				}
			}

			double getLL(Float y, const Eigen::Matrix<_WeightType, -1, 1>& numByTopic,
				Float docSize) const override
			{
				Float z = (this->regressionCoef.array() * numByTopic.array().template cast<Float>()).sum()
					/ std::max(docSize, 0.01f);
				return b * (y * z - log(1 + exp(z)));
			}

			Float estimate(const Eigen::Matrix<_WeightType, -1, 1>& numByTopic,
				Float docSize) const override
			{
				Float z = (this->regressionCoef.array() * numByTopic.array().template cast<Float>()).sum()
					/ std::max(docSize, 0.01f);
				return 1 / (1 + exp(-z));
			}

			DEFINE_SERIALIZER_AFTER_BASE(GLMFunctor<_WeightType>, b, omega);
		};
	}

	template<TermWeight _tw, size_t _Flags = flags::partitioned_multisampling,
		typename _Interface = ISLDAModel,
		typename _Derived = void,
		typename _DocType = DocumentSLDA<_tw>,
		typename _ModelState = ModelStateLDA<_tw>>
	class SLDAModel : public LDAModel<_tw, _Flags, _Interface,
		typename std::conditional<std::is_same<_Derived, void>::value, SLDAModel<_tw, _Flags>, _Derived>::type,
		_DocType, _ModelState>
	{
	protected:
		using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, SLDAModel<_tw>, _Derived>::type;
		using BaseClass = LDAModel<_tw, _Flags, _Interface, DerivedClass, _DocType, _ModelState>;
		friend BaseClass;
		friend typename BaseClass::BaseClass;
		using WeightType = typename BaseClass::WeightType;

		static constexpr char TMID[] = "SLDA";

		size_t F; // number of response variables
		std::vector<ISLDAModel::GLM> varTypes;
		std::vector<Float> glmParam;

		Eigen::Matrix<Float, -1, 1> mu; // Mean of regression coefficients, Dim : (F)
		Eigen::Matrix<Float, -1, 1> nuSq; // Variance of regression coefficients, Dim : (F)

		std::vector<std::unique_ptr<detail::GLMFunctor<WeightType>>> responseVars;
		Eigen::Matrix<Float, -1, -1> normZ; // topic proportions for all docs, Dim : (K, D)
		Eigen::Matrix<Float, -1, -1> Ys; // response variables, Dim : (D, F)

		template<bool _asymEta>
		Float* getZLikelihoods(_ModelState& ld, const _DocType& doc, size_t docId, size_t vid) const
		{
			const size_t V = this->realV;
			assert(vid < V);
			auto etaHelper = this->template getEtaHelper<_asymEta>();
			auto& zLikelihood = ld.zLikelihood;
			zLikelihood = (doc.numByTopic.array().template cast<Float>() + this->alphas.array())
				* (ld.numByTopicWord.col(vid).array().template cast<Float>() + etaHelper.getEta(vid))
				/ (ld.numByTopic.array().template cast<Float>() + etaHelper.getEtaSum());

			for (size_t f = 0; f < F; ++f)
			{
				if (std::isnan(doc.y[f])) continue;
				responseVars[f]->updateZLL(zLikelihood, doc.y[f], doc.numByTopic,
					docId, doc.getSumWordWeight());
			}
			sample::prefixSum(zLikelihood.data(), this->K);
			return &zLikelihood[0];
		}

		void optimizeRegressionCoef()
		{
			for (size_t i = 0; i < this->docs.size(); ++i)
			{
				normZ.col(i) = this->docs[i].numByTopic.array().template cast<Float>() / 
					std::max((Float)this->docs[i].getSumWordWeight(), 0.01f);
			}

			for (size_t f = 0; f < F; ++f)
			{
				responseVars[f]->optimizeCoef(normZ, mu[f], nuSq[f], Ys.col(f));
			}
		}

		void optimizeParameters(ThreadPool& pool, _ModelState* localData, RandGen* rgs)
		{
			BaseClass::optimizeParameters(pool, localData, rgs);
		}

		void updateGlobalInfo(ThreadPool& pool, _ModelState* localData)
		{
			optimizeRegressionCoef();
		}

		template<typename _DocIter>
		double getLLDocs(_DocIter _first, _DocIter _last) const
		{
			const auto K = this->K;

			double ll = 0;
			for (; _first != _last; ++_first)
			{
				auto& doc = *_first;
				ll -= math::lgammaT(doc.getSumWordWeight() + this->alphas.sum()) - math::lgammaT(this->alphas.sum());
				for (size_t f = 0; f < F; ++f)
				{
					if (std::isnan(doc.y[f])) continue;
					ll += responseVars[f]->getLL(doc.y[f], doc.numByTopic, doc.getSumWordWeight());
				}
				for (Tid k = 0; k < K; ++k)
				{
					ll += math::lgammaT(doc.numByTopic[k] + this->alphas[k]) - math::lgammaT(this->alphas[k]);
				}
			}
			return ll;
		}

		double getLLRest(const _ModelState& ld) const
		{
			double ll = BaseClass::getLLRest(ld);
			for (size_t f = 0; f < F; ++f)
			{
				ll -= (responseVars[f]->regressionCoef.array() - mu[f]).pow(2).sum() / 2 / nuSq[f];
			}
			return ll;
		}

		void prepareDoc(_DocType& doc, WeightType* topicDocPtr, size_t wordSize) const
		{
			BaseClass::prepareDoc(doc, topicDocPtr, wordSize);
		}

		void initGlobalState(bool initDocs)
		{
			BaseClass::initGlobalState(initDocs);
			if (initDocs)
			{
				for (size_t f = 0; f < F; ++f)
				{
					std::unique_ptr<detail::GLMFunctor<WeightType>> v;
					switch (varTypes[f])
					{
					case ISLDAModel::GLM::linear:
						v = make_unique<detail::LinearFunctor<WeightType>>(this->K, mu[f], 
							f < glmParam.size() ? glmParam[f] : 1.f);
						break;
					case ISLDAModel::GLM::binary_logistic:
						v = make_unique<detail::BinaryLogisticFunctor<WeightType>>(this->K, mu[f],
							f < glmParam.size() ? glmParam[f] : 1.f, this->docs.size());
						break;
					}
					responseVars.emplace_back(std::move(v));
				}
			}
			Ys.resize(this->docs.size(), F);
			normZ.resize(this->K, this->docs.size());
			for (size_t i = 0; i < this->docs.size(); ++i)
			{
				Ys.row(i) = Eigen::Map<Eigen::Matrix<Float, 1, -1>>(this->docs[i].y.data(), F);
			}
		}

	public:
		DEFINE_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseClass, 0, F, responseVars, mu, nuSq);
		DEFINE_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseClass, 1, 0x00010001, F, responseVars, mu, nuSq);

		SLDAModel(size_t _K = 1, const std::vector<ISLDAModel::GLM>& vars = {}, 
			Float _alpha = 0.1, Float _eta = 0.01, 
			const std::vector<Float>& _mu = {}, const std::vector<Float>& _nuSq = {},
			const std::vector<Float>& _glmParam = {},
			const RandGen& _rg = RandGen{ std::random_device{}() })
			: BaseClass(_K, _alpha, _eta, _rg), F(vars.size()), varTypes(vars), 
			glmParam(_glmParam)
		{
			for (auto t : varTypes)
			{
				if (t != ISLDAModel::GLM::linear && t != ISLDAModel::GLM::binary_logistic) THROW_ERROR_WITH_INFO(std::runtime_error, "unknown var GLM type in 'vars'");
			}
			mu = decltype(mu)::Zero(F);
			std::copy(_mu.begin(), _mu.end(), mu.data());
			nuSq = decltype(nuSq)::Ones(F);
			std::copy(_nuSq.begin(), _nuSq.end(), nuSq.data());
		}

		std::vector<Float> getRegressionCoef(size_t f) const override
		{
			return { responseVars[f]->regressionCoef.data(), responseVars[f]->regressionCoef.data() + this->K };
		}

		GETTER(F, size_t, F);

		ISLDAModel::GLM getTypeOfVar(size_t f) const override
		{
			return responseVars[f]->getType();
		}

		template<bool _const = false>
		_DocType& _updateDoc(_DocType& doc, const std::vector<Float>& y)
		{
			if (_const)
			{
				if (y.size() > F) throw std::runtime_error{ text::format(
					"size of 'y' is greater than the number of vars.\n"
					"size of 'y' : %zd, number of vars: %zd", y.size(), F) };
				doc.y = y;
				while (doc.y.size() < F)
				{
					doc.y.emplace_back(NAN);
				}
			}
			else
			{
				if (y.size() != F) throw std::runtime_error{ text::format(
					"size of 'y' must be equal to the number of vars.\n"
					"size of 'y' : %zd, number of vars: %zd", y.size(), F) };
				doc.y = y;
			}
			return doc;
		}

		size_t addDoc(const std::vector<std::string>& words, const std::vector<Float>& y) override
		{
			auto doc = this->_makeDoc(words);
			return this->_addDoc(_updateDoc(doc, y));
		}

		std::unique_ptr<DocumentBase> makeDoc(const std::vector<std::string>& words, const std::vector<Float>& y) const override
		{
			auto doc = as_mutable(this)->template _makeDoc<true>(words);
			return make_unique<_DocType>(as_mutable(this)->template _updateDoc<true>(doc, y));
		}

		size_t addDoc(const std::string& rawStr, const RawDocTokenizer::Factory& tokenizer,
			const std::vector<Float>& y) override
		{
			auto doc = this->template _makeRawDoc<false>(rawStr, tokenizer);
			return this->_addDoc(_updateDoc(doc, y));
		}

		std::unique_ptr<DocumentBase> makeDoc(const std::string& rawStr, const RawDocTokenizer::Factory& tokenizer,
			const std::vector<Float>& y) const override
		{
			auto doc = as_mutable(this)->template _makeRawDoc<true>(rawStr, tokenizer);
			return make_unique<_DocType>(as_mutable(this)->template _updateDoc<true>(doc, y));
		}

		size_t addDoc(const std::string& rawStr, const std::vector<Vid>& words,
			const std::vector<uint32_t>& pos, const std::vector<uint16_t>& len,
			const std::vector<Float>& y) override
		{
			auto doc = this->_makeRawDoc(rawStr, words, pos, len);
			return this->_addDoc(_updateDoc(doc, y));
		}

		std::unique_ptr<DocumentBase> makeDoc(const std::string& rawStr, const std::vector<Vid>& words,
			const std::vector<uint32_t>& pos, const std::vector<uint16_t>& len,
			const std::vector<Float>& y) const override
		{
			auto doc = this->_makeRawDoc(rawStr, words, pos, len);
			return make_unique<_DocType>(as_mutable(this)->template _updateDoc<true>(doc, y));
		}

		std::vector<Float> estimateVars(const DocumentBase* doc) const override
		{
			std::vector<Float> ret;
			auto pdoc = dynamic_cast<const _DocType*>(doc);
			if (!pdoc) return ret;
			for (auto& f : responseVars)
			{
				ret.emplace_back(f->estimate(pdoc->numByTopic, pdoc->getSumWordWeight()));
			}
			return ret;
		}
	};

	template<typename _WeightType>
	void detail::GLMFunctor<_WeightType>::serializerRead(
		std::unique_ptr<detail::GLMFunctor<_WeightType>>& p, std::istream& istr)
	{
		uint32_t t = serializer::readFromStream<uint32_t>(istr);
		if (!t) p.reset();
		else
		{
			switch ((ISLDAModel::GLM)(t - 1))
			{
			case ISLDAModel::GLM::linear:
				p = make_unique<LinearFunctor<_WeightType>>();
				break;
			case ISLDAModel::GLM::binary_logistic:
				p = make_unique<BinaryLogisticFunctor<_WeightType>>();
				break;
			default:
				throw std::ios_base::failure(text::format("wrong GLMFunctor type id %d", (t - 1)));
			}
			p->serializerRead(istr);
		}
	}
}
