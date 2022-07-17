#pragma once
#include "LDA.h"

namespace tomoto
{
    template<TermWeight _tw>
	struct DocumentHDP : public DocumentLDA<_tw>
	{
		/* 
		For DocumentHDP, the topic in numByTopic, Zs indicates 'table id', not 'topic id'.
		To get real 'topic id', check the topic field of numTopicByTable.
		*/
		using BaseDocument = DocumentLDA<_tw>;
		using DocumentLDA<_tw>::DocumentLDA;
		using WeightType = typename DocumentLDA<_tw>::WeightType;
		struct TableTopicInfo
		{
			WeightType num;
			Tid topic;

			TableTopicInfo(WeightType _num = 0, Tid _topic = 0) : num(_num), topic(_topic)
			{
			}

			operator const bool() const
			{
				return num > (WeightType)1e-2;
			}

			void serializerWrite(std::ostream& writer) const
			{
				serializer::writeMany(writer, topic);
			}

			void serializerRead(std::istream& reader)
			{
				serializer::readMany(reader, topic);
			}
		};
		std::vector<TableTopicInfo> numTopicByTable;

		DEFINE_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseDocument, 0, numTopicByTable);
		DEFINE_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseDocument, 1, 0x00010001, numTopicByTable);

		size_t getNumTable() const
		{
			return std::count_if(numTopicByTable.begin(), numTopicByTable.end(), [](const TableTopicInfo& e) { return (bool)e; });
		}

		// add a new table into doc and return the new table's idx
		size_t addNewTable(Tid tid)
		{
			return insertIntoEmpty(numTopicByTable, TableTopicInfo( 0, tid ));
		}

		template<typename _TopicModel> void update(WeightType* ptr, const _TopicModel& mdl);
	};

	struct HDPArgs : public LDAArgs
	{
		Float gamma = 0.1;
		
		HDPArgs()
		{
			k = 2;
		}
	};

    class IHDPModel : public ILDAModel
	{
	public:
		using DefaultDocType = DocumentHDP<TermWeight::one>;
		static IHDPModel* create(TermWeight _weight, const HDPArgs& args,
			bool scalarRng = false);

		virtual Float getGamma() const = 0;
		virtual size_t getTotalTables() const = 0;
		virtual size_t getLiveK() const = 0;
		virtual bool isLiveTopic(Tid tid) const = 0;

		virtual std::unique_ptr<ILDAModel> convertToLDA(float topicThreshold, std::vector<Tid>& newK) const = 0;
		virtual std::vector<Tid> purgeDeadTopics() = 0;
	};
}