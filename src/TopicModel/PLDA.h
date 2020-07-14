#pragma once
#include "LLDA.h"

namespace tomoto
{

	class IPLDAModel : public ILLDAModel
	{
	public:
		using DefaultDocType = DocumentLLDA<TermWeight::one>;
		static IPLDAModel* create(TermWeight _weight, size_t _numLatentTopics = 0, size_t _numTopicsPerLabel = 1,
			Float alpha = 0.1, Float eta = 0.01, size_t seed = std::random_device{}(),
			bool scalarRng = false);

		virtual size_t getNumLatentTopics() const = 0;
	};
}