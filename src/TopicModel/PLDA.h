#pragma once
#include "LLDA.h"

namespace tomoto
{

	class IPLDAModel : public ILLDAModel
	{
	public:
		using DefaultDocType = DocumentLLDA<TermWeight::one>;
		static IPLDAModel* create(TermWeight _weight, size_t _numLatentTopics = 0, size_t _numTopicsPerLabel = 1,
			Float alpha = 0.1, Float eta = 0.01, const RandGen& _rg = RandGen{ std::random_device{}() });

		virtual size_t getNumLatentTopics() const = 0;
	};
}