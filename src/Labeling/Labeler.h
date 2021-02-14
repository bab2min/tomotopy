#pragma once
#include <vector>
#include <string>
#include "../TopicModel/TopicModel.hpp"

namespace tomoto
{
	namespace label
	{
		struct Candidate
		{
			float score = 0;
			size_t cf = 0, df = 0;
			std::vector<Vid> w;
			std::string name;

			Candidate()
			{
			}

			Candidate(float _score, Vid w1)
				: score{ _score }, w{ w1 }
			{
			}

			Candidate(float _score, Vid w1, Vid w2)
				: score{ _score }, w{ w1, w2 }
			{
			}

			Candidate(float _score, const std::vector<Vid>& _w)
				: score{ _score }, w{ _w }
			{
			}
		};

		class IExtractor
		{
		public:
			
			virtual std::vector<Candidate> extract(const ITopicModel* tm) const = 0;
			virtual ~IExtractor() {}
		};

		class ILabeler
		{
		public:
			virtual std::vector<std::pair<std::string, float>> getLabels(Tid tid, size_t topK = 10) const = 0;
			virtual ~ILabeler() {}
		};
	}
}