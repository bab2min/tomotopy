#pragma once
#include "LDA.h"

namespace tomoto
{
    template<TermWeight _TW>
	struct DocumentHDP : public DocumentLDA<_TW>
	{
		/* 
		For DocumentHDP, the topic in numByTopic, Zs indicates 'table id', not 'topic id'.
		To get real 'topic id', check the topic field of numTopicByTable.
		*/
		using DocumentLDA<_TW>::DocumentLDA;
		using WeightType = typename DocumentLDA<_TW>::WeightType;
		struct TableTopicInfo
		{
			WeightType num;
			TID topic;

			TableTopicInfo(WeightType _num = 0, TID _topic = 0) : num(_num), topic(_topic)
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

		DEFINE_SERIALIZER_AFTER_BASE(DocumentLDA<_TW>, numTopicByTable);

		size_t getNumTable() const
		{
			return std::count_if(numTopicByTable.begin(), numTopicByTable.end(), [](const TableTopicInfo& e) { return (bool)e; });
		}
		size_t addNewTable(TID tid)
		{
			return insertIntoEmpty(numTopicByTable, TableTopicInfo( 0, tid ));
		}

		template<typename _TopicModel> void update(WeightType* ptr, const _TopicModel& mdl);
	};

    class IHDPModel : public ILDAModel
	{
	public:
		using DefaultDocType = DocumentHDP<TermWeight::one>;
		static IHDPModel* create(TermWeight _weight, size_t _K = 1, FLOAT alpha = 0.1, FLOAT eta = 0.01, FLOAT gamma = 0.1, const RandGen& _rg = RandGen{ std::random_device{}() });

		virtual FLOAT getGamma() const = 0;
		virtual size_t getTotalTables() const = 0;
		virtual size_t getLiveK() const = 0;
		virtual bool isLiveTopic(TID tid) const = 0;
	};
}