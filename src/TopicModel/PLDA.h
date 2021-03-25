#pragma once
#include "LLDA.h"

namespace tomoto
{
	struct PLDAArgs : public LDAArgs
	{
		size_t numLatentTopics = 0;
		size_t numTopicsPerLabel = 1;

		PLDAArgs setK(size_t _k = 1) const
		{
			PLDAArgs ret = *this;
			ret.k = _k;
			return ret;
		}
	};

	class IPLDAModel : public ILLDAModel
	{
	public:
		using DefaultDocType = DocumentLLDA<TermWeight::one>;
		static IPLDAModel* create(TermWeight _weight, const PLDAArgs& args,
			bool scalarRng = false);

		virtual size_t getNumLatentTopics() const = 0;
	};
}