#pragma once

#include <cmath>
#include "Common.h"

namespace tomoto
{
	namespace coherence
	{
		template<ConfirmMeasure _cm>
		class ConfirmMeasurer;

		template<>
		class ConfirmMeasurer<ConfirmMeasure::difference>
		{
			const double eps;
		public:
			ConfirmMeasurer(double _eps = 1e-12) : eps{ _eps } {}

			double operator()(
				const IProbEstimator* pe, Vid word1, Vid word2) const
			{
				if (word1 == word2) return -pe->getProb(word1);
				return pe->getProb(word1, word2) / (pe->getProb(word2) + eps) - pe->getProb(word1);
			}

			double operator()(
				const IProbEstimator* pe, Vid word1, const std::vector<Vid>& word2) const
			{
				return pe->getProb(word1, word2) / (pe->getProb(word2) + eps) - pe->getProb(word1);
			}
		};

		template<>
		class ConfirmMeasurer<ConfirmMeasure::ratio>
		{
			const double eps;
		public:
			ConfirmMeasurer(double _eps = 1e-12) : eps{ _eps } {}

			double operator()(
				const IProbEstimator* pe, Vid word1, Vid word2) const
			{
				if (word1 == word2) return 1 / (pe->getProb(word1) + eps);
				return pe->getProb(word1, word2) / (pe->getProb(word1) * pe->getProb(word2) + eps);
			}

			double operator()(
				const IProbEstimator* pe, Vid word1, const std::vector<Vid>& word2) const
			{
				return pe->getProb(word1, word2) / (pe->getProb(word1) * pe->getProb(word2) + eps);
			}
		};

		template<>
		class ConfirmMeasurer<ConfirmMeasure::pmi>
		{
			const double eps;
		public:
			ConfirmMeasurer(double _eps = 1e-12) : eps{ _eps } {}

			double operator()(
				const IProbEstimator* pe, Vid word1, Vid word2) const
			{
				if (word1 == word2) return -std::log(pe->getProb(word1) + eps);
				return std::log((pe->getProb(word1, word2) + eps) / (pe->getProb(word1) * pe->getProb(word2) + eps));
			}

			double operator()(
				const IProbEstimator* pe, Vid word1, const std::vector<Vid>& word2) const
			{
				return std::log((pe->getProb(word1, word2) + eps) / (pe->getProb(word1) * pe->getProb(word2) + eps));
			}
		};

		template<>
		class ConfirmMeasurer<ConfirmMeasure::npmi>
		{
			const double eps;
		public:
			ConfirmMeasurer(double _eps = 1e-12) : eps{ _eps } {}

			double operator()(
				const IProbEstimator* pe, Vid word1, Vid word2) const
			{
				if (word1 == word2) return 1;
				return (std::log((pe->getProb(word1, word2) + eps) / (pe->getProb(word1) * pe->getProb(word2) + eps)))
					/ -std::log(pe->getProb(word1, word2) + eps);
			}

			double operator()(
				const IProbEstimator* pe, Vid word1, const std::vector<Vid>& word2) const
			{
				return (std::log((pe->getProb(word1, word2) + eps) / (pe->getProb(word1) * pe->getProb(word2) + eps)))
					/ -std::log(pe->getProb(word1, word2) + eps);
			}
		};

		template<>
		class ConfirmMeasurer<ConfirmMeasure::likelihood>
		{
			const double eps;
		public:
			ConfirmMeasurer(double _eps = 1e-12) : eps{ _eps } {}

			double operator()(
				const IProbEstimator* pe, Vid word1, Vid word2) const
			{
				return pe->getProb(word1, word2) / (pe->getProb(word2) + eps)
					/ (pe->getJointNotProb(word1, word2) + eps) * (1 - pe->getProb(word2));
			}

			double operator()(
				const IProbEstimator* pe, Vid word1, const std::vector<Vid>& word2) const
			{
				return pe->getProb(word1, word2) / (pe->getProb(word2) + eps)
					/ (pe->getJointNotProb(word1, word2) + eps) * (1 - pe->getProb(word2));
			}
		};

		template<>
		class ConfirmMeasurer<ConfirmMeasure::loglikelihood>
		{
			const double eps;
		public:
			ConfirmMeasurer(double _eps = 1e-12) : eps{ _eps } {}

			double operator()(
				const IProbEstimator* pe, Vid word1, Vid word2) const
			{
				return std::log(pe->getProb(word1, word2) / (pe->getProb(word2) + eps)
					/ (pe->getJointNotProb(word1, word2) + eps) * (1 - pe->getProb(word2)) + eps);
			}

			double operator()(
				const IProbEstimator* pe, Vid word1, const std::vector<Vid>& word2) const
			{
				return std::log(pe->getProb(word1, word2) / (pe->getProb(word2) + eps)
					/ (pe->getJointNotProb(word1, word2) + eps) * (1 - pe->getProb(word2)) + eps);
			}
		};

		template<>
		class ConfirmMeasurer<ConfirmMeasure::logcond>
		{
			const double eps;
		public:
			ConfirmMeasurer(double _eps = 1e-12) : eps{ _eps } {}

			double operator()(
				const IProbEstimator* pe, Vid word1, Vid word2) const
			{
				if (word1 == word2) return 0;
				return std::log(pe->getProb(word1, word2) / (pe->getProb(word2) + eps) + eps);
			}

			double operator()(
				const IProbEstimator* pe, Vid word1, const std::vector<Vid>& word2) const
			{
				return std::log(pe->getProb(word1, word2) / (pe->getProb(word2) + eps) + eps);
			}
		};

		template<typename _CMFunc, IndirectMeasure _im>
		class IndirectMeasurer
		{
			_CMFunc cm;
			float gamma;
			std::vector<Vid> targets;
			mutable std::unordered_map<Vid, Eigen::ArrayXf> vectorCache;

			const Eigen::ArrayXf& getVector(const IProbEstimator* pe, Vid word) const
			{
				auto it = vectorCache.find(word);
				if (it != vectorCache.end()) return it->second;

				Eigen::ArrayXf v(targets.size());
				for (size_t i = 0; i < targets.size(); ++i)
				{
					v[i] = cm(pe, word, targets[i]);
				}
				v = v.pow(gamma);
				return vectorCache.emplace(word, std::move(v)).first->second;
			}

			double calcMeasure(const Eigen::ArrayXf& v1, const Eigen::ArrayXf& v2) const
			{
				switch (_im)
				{
				case IndirectMeasure::cosine:
					return (v1 * v2).sum() / std::sqrt(v1.square().sum() * v2.square().sum());
				case IndirectMeasure::dice:
					return v1.min(v2).sum() / (v1.sum() + v2.sum());
				case IndirectMeasure::jaccard:
					return v1.min(v2).sum() / v1.max(v2).sum();
				}
				return 0;
			}

		public:
			template<typename _TargetIter>
			IndirectMeasurer(const _CMFunc& _cm, double _gamma, _TargetIter targetFirst, _TargetIter targetLast)
				: cm{ _cm }, gamma{ (float)_gamma }
			{
				std::set<Vid> uniqTargets{ targetFirst, targetLast };
				targets.insert(targets.end(), uniqTargets.begin(), uniqTargets.end());
			}

			double operator()(
				const IProbEstimator* pe, Vid word1, Vid word2) const
			{
				const Eigen::ArrayXf& v1 = getVector(pe, word1);
				const Eigen::ArrayXf& v2 = getVector(pe, word2);
				return calcMeasure(v1, v2);
			}

			double operator()(
				const IProbEstimator* pe, Vid word1, const std::vector<Vid>& word2) const
			{
				const Eigen::ArrayXf& v1 = getVector(pe, word1);
				Eigen::ArrayXf v2 = getVector(pe, word2[0]);
				for (size_t i = 1; i < word2.size(); ++i)
				{
					v2 += getVector(pe, word2[i]);
				}
				return calcMeasure(v1, v2);
			}
		};

		template<typename _CMFunc>
		class IndirectMeasurer<_CMFunc, IndirectMeasure::none> : public _CMFunc
		{
		public:
			template<typename _TargetIter>
			IndirectMeasurer(const _CMFunc& _cm, double _gamma, _TargetIter targetFirst, _TargetIter targetLast)
				: _CMFunc{ _cm }
			{
			}
		};

		class AnyConfirmMeasurer
		{
			struct Concept
			{
				virtual ~Concept() {}
				virtual double operator()(
					const IProbEstimator* pe, Vid word1, Vid word2) const = 0;
				virtual double operator()(
					const IProbEstimator* pe, Vid word1, const std::vector<Vid>& word2) const = 0;
			};

			template<typename T>
			struct Model : Concept
			{
			private:
				T object;
			public:
				Model(const T& t) : object(t) {}

				double operator()(
					const IProbEstimator* pe, Vid word1, Vid word2) const override
				{
					return object(pe, word1, word2);
				}

				double operator()(
					const IProbEstimator* pe, Vid word1, const std::vector<Vid>& word2) const override
				{
					return object(pe, word1, word2);
				}
			};

			std::shared_ptr<const Concept> object;

		public:
			AnyConfirmMeasurer() = default;

			template<typename T>
			AnyConfirmMeasurer(const T& obj)
				: object{ std::make_shared<Model<T>>(std::move(obj)) }
			{
			}

			double operator()(
				const IProbEstimator* pe, Vid word1, Vid word2) const
			{
				return (*object)(pe, word1, word2);
			}

			double operator()(
				const IProbEstimator* pe, Vid word1, const std::vector<Vid>& word2) const
			{
				return (*object)(pe, word1, word2);
			}

			operator bool() const
			{
				return (bool)object;
			}

			template<IndirectMeasure _im, typename _TargetIter>
			static AnyConfirmMeasurer makeIM(ConfirmMeasure cm, double eps, double gamma, _TargetIter targetFirst, _TargetIter targetLast)
			{
				switch (cm)
				{
				case ConfirmMeasure::difference:
					return { IndirectMeasurer<ConfirmMeasurer<ConfirmMeasure::difference>, _im>{ 
						ConfirmMeasurer<ConfirmMeasure::difference>{ eps }, gamma, targetFirst, targetLast 
					} };
				case ConfirmMeasure::ratio:
					return { IndirectMeasurer<ConfirmMeasurer<ConfirmMeasure::ratio>, _im>{
						ConfirmMeasurer<ConfirmMeasure::ratio>{ eps }, gamma, targetFirst, targetLast
					} };
				case ConfirmMeasure::pmi:
					return { IndirectMeasurer<ConfirmMeasurer<ConfirmMeasure::pmi>, _im>{
						ConfirmMeasurer<ConfirmMeasure::pmi>{ eps }, gamma, targetFirst, targetLast
					} };
				case ConfirmMeasure::npmi:
					return { IndirectMeasurer<ConfirmMeasurer<ConfirmMeasure::npmi>, _im>{
						ConfirmMeasurer<ConfirmMeasure::npmi>{ eps }, gamma, targetFirst, targetLast
					} };
				case ConfirmMeasure::likelihood:
					return { IndirectMeasurer<ConfirmMeasurer<ConfirmMeasure::likelihood>, _im>{
						ConfirmMeasurer<ConfirmMeasure::likelihood>{ eps }, gamma, targetFirst, targetLast
					} };
				case ConfirmMeasure::loglikelihood:
					return { IndirectMeasurer<ConfirmMeasurer<ConfirmMeasure::loglikelihood>, _im>{
						ConfirmMeasurer<ConfirmMeasure::loglikelihood>{ eps }, gamma, targetFirst, targetLast
					} };
				case ConfirmMeasure::logcond:
					return { IndirectMeasurer<ConfirmMeasurer<ConfirmMeasure::logcond>, _im>{
						ConfirmMeasurer<ConfirmMeasure::logcond>{ eps }, gamma, targetFirst, targetLast
					} };
				default:
					throw std::invalid_argument{ "invalid ConfirmMeasure `cm`" };
				}
			}

			template<typename _TargetIter>
			static AnyConfirmMeasurer getInstance(ConfirmMeasure cm, IndirectMeasure im, 
				_TargetIter targetFirst, _TargetIter targetLast,
				double eps = 1e-12, double gamma = 1)
			{
				switch (im)
				{
				case IndirectMeasure::none:
					return makeIM<IndirectMeasure::none>(cm, eps, gamma, targetFirst, targetLast);
				case IndirectMeasure::cosine:
					return makeIM<IndirectMeasure::cosine>(cm, eps, gamma, targetFirst, targetLast);
				case IndirectMeasure::dice:
					return makeIM<IndirectMeasure::dice>(cm, eps, gamma, targetFirst, targetLast);
				case IndirectMeasure::jaccard:
					return makeIM<IndirectMeasure::jaccard>(cm, eps, gamma, targetFirst, targetLast);
				default:
					throw std::invalid_argument{ "invalid IndirectMeasure `im`" };
				}
			}
		};

	}
}
