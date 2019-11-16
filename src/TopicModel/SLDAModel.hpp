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
			Eigen::Matrix<FLOAT, -1, 1> regressionCoef; // Dim : (K)

			GLMFunctor(size_t K = 0, FLOAT mu = 0) : regressionCoef(Eigen::Matrix<FLOAT, -1, 1>::Constant(K, mu))
			{
			}

			virtual ISLDAModel::GLM getType() const = 0;

			virtual void updateZLL(
				Eigen::Matrix<FLOAT, -1, 1>& zLikelihood,
				FLOAT y, const Eigen::Matrix<_WeightType, -1, 1>& numByTopic, size_t docId, FLOAT docSize) const = 0;

			virtual void optimizeCoef(
				const Eigen::Matrix<FLOAT, -1, -1>& normZ,
				const Eigen::Matrix<FLOAT, -1, -1>& normZZT,
				FLOAT mu, FLOAT nuSq,
				Eigen::Block<Eigen::Matrix<FLOAT, -1, -1>, -1, 1, true> ys
			) = 0;

			virtual double getLL(FLOAT y, const Eigen::Matrix<_WeightType, -1, 1>& numByTopic,
				FLOAT docSize) const = 0;

			virtual FLOAT estimate(const Eigen::Matrix<_WeightType, -1, 1>& numByTopic,
				FLOAT docSize) const = 0;

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
			FLOAT sigmaSq = 1;

			LinearFunctor(size_t K = 0, FLOAT mu = 0, FLOAT _sigmaSq = 1)
				: GLMFunctor<_WeightType>(K, mu), sigmaSq(_sigmaSq)
			{
			}

			ISLDAModel::GLM getType() const { return ISLDAModel::GLM::linear; }

			void updateZLL(
				Eigen::Matrix<FLOAT, -1, 1>& zLikelihood,
				FLOAT y, const Eigen::Matrix<_WeightType, -1, 1>& numByTopic, size_t docId, FLOAT docSize) const override
			{
				FLOAT yErr = y -
					(this->regressionCoef.array() * numByTopic.array().template cast<FLOAT>()).sum()
					/ docSize;
				zLikelihood.array() *= (this->regressionCoef.array() / docSize / 2 / sigmaSq *
					(2 * yErr - this->regressionCoef.array() / docSize)).exp();
			}

			void optimizeCoef(
				const Eigen::Matrix<FLOAT, -1, -1>& normZ,
				const Eigen::Matrix<FLOAT, -1, -1>& normZZT,
				FLOAT mu, FLOAT nuSq,
				Eigen::Block<Eigen::Matrix<FLOAT, -1, -1>, -1, 1, true> ys
			) override
			{
				this->regressionCoef = (normZZT + Eigen::Matrix<FLOAT, -1, -1>::Identity(normZZT.cols(), normZZT.cols()) / nuSq)
					.colPivHouseholderQr().solve(normZ * ys);
			}

			double getLL(FLOAT y, const Eigen::Matrix<_WeightType, -1, 1>& numByTopic,
				FLOAT docSize) const override
			{
				FLOAT estimatedY = estimate(numByTopic, docSize);
				return -pow(estimatedY - y, 2) / 2 / sigmaSq;
			}

			FLOAT estimate(const Eigen::Matrix<_WeightType, -1, 1>& numByTopic,
				FLOAT docSize) const
			{
				return (this->regressionCoef.array() * numByTopic.array().template cast<FLOAT>()).sum()
					/ std::max(docSize, 0.01f);
			}

			DEFINE_SERIALIZER_AFTER_BASE(GLMFunctor<_WeightType>, sigmaSq);
		};

		template<typename _WeightType>
		struct BinaryLogisticFunctor : public GLMFunctor<_WeightType>
		{
			FLOAT b = 1;
			Eigen::Matrix<FLOAT, -1, 1> omega;

			BinaryLogisticFunctor(size_t K = 0, FLOAT mu = 0, FLOAT _b = 1, size_t numDocs = 0)
				: GLMFunctor<_WeightType>(K, mu), b(_b), omega{ Eigen::Matrix<FLOAT, -1, 1>::Ones(numDocs) }
			{
			}

			ISLDAModel::GLM getType() const { return ISLDAModel::GLM::binary_logistic; }

			void updateZLL(
				Eigen::Matrix<FLOAT, -1, 1>& zLikelihood,
				FLOAT y, const Eigen::Matrix<_WeightType, -1, 1>& numByTopic, size_t docId, FLOAT docSize) const override
			{
				FLOAT yErr = b * (y - 0.5f) -
					(this->regressionCoef.array() * numByTopic.array().template cast<FLOAT>()).sum()
					/ docSize * omega[docId];
				zLikelihood.array() *= (this->regressionCoef.array() / docSize *
					(yErr - omega[docId] / 2 * this->regressionCoef.array() / docSize)).exp();
			}

			void optimizeCoef(
				const Eigen::Matrix<FLOAT, -1, -1>& normZ,
				const Eigen::Matrix<FLOAT, -1, -1>& normZZT,
				FLOAT mu, FLOAT nuSq,
				Eigen::Block<Eigen::Matrix<FLOAT, -1, -1>, -1, 1, true> ys
			) override
			{
				this->regressionCoef = ((normZ * Eigen::DiagonalMatrix<FLOAT, -1>{ omega }) * normZ.transpose()
					+ Eigen::Matrix<FLOAT, -1, -1>::Identity(normZZT.cols(), normZZT.cols()) / nuSq)
					.colPivHouseholderQr().solve(normZ * (b * (ys - decltype(ys)::Constant(ys.size(), 0.5f)))
						+ Eigen::Matrix<FLOAT, -1, 1>::Constant(normZ.rows(), mu / nuSq));

				RandGen rng;
				for (size_t i = 0; i < omega.size(); ++i)
				{
					omega[i] = math::drawPolyaGamma(b, (this->regressionCoef.array() * normZ.col(i).array()).sum(), rng);
				}
			}

			double getLL(FLOAT y, const Eigen::Matrix<_WeightType, -1, 1>& numByTopic,
				FLOAT docSize) const override
			{
				FLOAT z = (this->regressionCoef.array() * numByTopic.array().template cast<FLOAT>()).sum()
					/ std::max(docSize, 0.01f);
				return b * (y * z - log(1 + exp(z)));
			}

			FLOAT estimate(const Eigen::Matrix<_WeightType, -1, 1>& numByTopic,
				FLOAT docSize) const
			{
				FLOAT z = (this->regressionCoef.array() * numByTopic.array().template cast<FLOAT>()).sum()
					/ std::max(docSize, 0.01f);
				return 1 / (1 + exp(-z));
			}

			DEFINE_SERIALIZER_AFTER_BASE(GLMFunctor<_WeightType>, b, omega);
		};
	}

	template<TermWeight _TW, size_t _Flags = 0,
		typename _Interface = ISLDAModel,
		typename _Derived = void,
		typename _DocType = DocumentSLDA<_TW>,
		typename _ModelState = ModelStateLDA<_TW>>
	class SLDAModel : public LDAModel<_TW, _Flags, _Interface,
		typename std::conditional<std::is_same<_Derived, void>::value, SLDAModel<_TW, _Flags>, _Derived>::type,
		_DocType, _ModelState>
	{
		static constexpr const char* TMID = "SLDA";
	protected:
		using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, SLDAModel<_TW>, _Derived>::type;
		using BaseClass = LDAModel<_TW, _Flags, _Interface, DerivedClass, _DocType, _ModelState>;
		friend BaseClass;
		friend typename BaseClass::BaseClass;
		using WeightType = typename BaseClass::WeightType;

		size_t F; // number of response variables
		std::vector<ISLDAModel::GLM> varTypes;
		std::vector<FLOAT> glmParam;

		Eigen::Matrix<FLOAT, -1, 1> mu; // Mean of regression coefficients, Dim : (F)
		Eigen::Matrix<FLOAT, -1, 1> nuSq; // Variance of regression coefficients, Dim : (F)

		std::vector<std::unique_ptr<detail::GLMFunctor<WeightType>>> responseVars;
		Eigen::Matrix<FLOAT, -1, -1> normZ; // topic proportions for all docs, Dim : (K, D)
		Eigen::Matrix<FLOAT, -1, -1> Ys; // response variables, Dim : (D, F)

		FLOAT* getZLikelihoods(_ModelState& ld, const _DocType& doc, size_t docId, size_t vid) const
		{
			const size_t V = this->realV;
			assert(vid < V);
			auto& zLikelihood = ld.zLikelihood;
			zLikelihood = (doc.numByTopic.array().template cast<FLOAT>() + this->alphas.array())
				* (ld.numByTopicWord.col(vid).array().template cast<FLOAT>() + this->eta)
				/ (ld.numByTopic.array().template cast<FLOAT>() + V * this->eta);
			if (docId != (size_t)-1)
			{
				for (size_t f = 0; f < F; ++f)
				{
					responseVars[f]->updateZLL(zLikelihood, doc.y[f], doc.numByTopic,
						docId, doc.getSumWordWeight());
				}
			}
			sample::prefixSum(zLikelihood.data(), this->K);
			return &zLikelihood[0];
		}

		void optimizeRegressionCoef()
		{
			for (size_t i = 0; i < this->docs.size(); ++i)
			{
				normZ.col(i) = this->docs[i].numByTopic.array().template cast<FLOAT>() / 
					std::max((FLOAT)this->docs[i].getSumWordWeight(), 0.01f);
			}
			Eigen::Matrix<FLOAT, -1, -1> normZZT = normZ * normZ.transpose();
			for (size_t f = 0; f < F; ++f)
			{
				responseVars[f]->optimizeCoef(normZ, normZZT, mu[f], nuSq[f], Ys.col(f));
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
					ll += responseVars[f]->getLL(doc.y[f], doc.numByTopic, doc.getSumWordWeight());
				}
				for (TID k = 0; k < K; ++k)
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
				Ys.row(i) = Eigen::Map<Eigen::Matrix<FLOAT, 1, -1>>(this->docs[i].y.data(), F);
			}
		}

		DEFINE_SERIALIZER_AFTER_BASE(BaseClass, F, responseVars, mu, nuSq);

	public:
		SLDAModel(size_t _K = 1, const std::vector<ISLDAModel::GLM>& vars = {}, 
			FLOAT _alpha = 0.1, FLOAT _eta = 0.01, 
			const std::vector<FLOAT>& _mu = {}, const std::vector<FLOAT>& _nuSq = {},
			const std::vector<FLOAT>& _glmParam = {},
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

		std::vector<FLOAT> getRegressionCoef(size_t f) const
		{
			return { responseVars[f]->regressionCoef.data(), responseVars[f]->regressionCoef.data() + this->K };
		}

		GETTER(F, size_t, F);

		ISLDAModel::GLM getTypeOfVar(size_t f) const
		{
			return responseVars[f]->getType();
		}

		size_t addDoc(const std::vector<std::string>& words, const std::vector<FLOAT>& y) override
		{
			if (y.size() != F) throw std::runtime_error{ text::format(
				"size of 'y' must be equal to the number of vars.\n"
				"size of 'y' : %zd, number of vars: %zd", y.size(), F) };
			auto doc = this->_makeDoc(words);
			doc.y = y;
			return this->_addDoc(doc);
		}

		std::unique_ptr<DocumentBase> makeDoc(const std::vector<std::string>& words, const std::vector<FLOAT>& y) const override
		{
			auto doc = this->_makeDocWithinVocab(words);
			doc.y = y;
			return make_unique<_DocType>(doc);
		}

		std::vector<FLOAT> estimateVars(const DocumentBase* doc) const
		{
			std::vector<FLOAT> ret;
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
