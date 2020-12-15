#pragma once
#include "../TopicModel/TopicModel.hpp"

namespace tomoto
{
	namespace coherence
	{
		enum class Segmentation
		{
			none = 0,
			one_one,
			one_pre,
			one_suc,
			one_all,
			one_set,
		};

		enum class ProbEstimation
		{
			none = 0,
			document,
			sliding_windows,
		};

		class IProbEstimator
		{
		public:
			virtual double getProb(Vid word) const = 0;
			virtual double getProb(Vid word1, Vid word2) const = 0;
			virtual double getProb(const std::vector<Vid>& words) const = 0;
			virtual double getJointNotProb(Vid word1, Vid word2) const = 0;
			virtual double getJointNotProb(Vid word1, const std::vector<Vid>& word2) const = 0;
			virtual ~IProbEstimator() {}

			double getProb(Vid word1, const std::vector<Vid>& word2) const
			{
				auto words = word2;
				if(std::find(words.begin(), words.end(), word1) == words.end()) words.emplace_back(word1);
				return getProb(words);
			}
		};

		enum class ConfirmMeasure
		{
			none = 0,
			difference,
			ratio,
			likelihood,
			loglikelihood,
			pmi,
			npmi,
			logcond,
		};

		enum class IndirectMeasure
		{
			none = 0,
			cosine,
			dice,
			jaccard,
		};

		/*enum class Aggregation
		{
			none = 0,
			amean,
			median,
			gmean,
			hmean,
			qmean,
		};*/
	}
}
