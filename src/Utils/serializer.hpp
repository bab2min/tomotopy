#pragma once
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <Eigen/Dense>
#include <vector>
#include "tvector.hpp"
#include "text.hpp"

namespace tomoto
{
	namespace serializer
	{
		template<typename _Ty> inline void writeToStream(std::ostream& ostr, const _Ty& v);
		template<typename _Ty> inline void readFromStream(std::istream& istr, _Ty& v);
		template<typename _Ty> inline _Ty readFromStream(std::istream& istr);

		class UnfitException : public std::ios_base::failure
		{
			using std::ios_base::failure::failure;
		};

		struct MagicConstant
		{
			const char* m;
			MagicConstant(const char* _m) : m(_m)
			{}
		};

		inline void writeMany(std::ostream& ostr)
		{
			// do nothing
		}

		template<typename _FirstTy, typename ... _RestTy>
		inline typename std::enable_if<!std::is_same<_FirstTy, MagicConstant>::value>::type writeMany(std::ostream& ostr, const _FirstTy& first, _RestTy&&... rest)
		{
			writeToStream(ostr, first);
			writeMany(ostr, std::forward<_RestTy>(rest)...);
		}

		template<typename ... _RestTy>
		inline void writeMany(std::ostream& ostr, const MagicConstant& first, _RestTy&&... rest)
		{
			writeToStream(ostr, *(uint32_t*)first.m);
			writeMany(ostr, std::forward<_RestTy>(rest)...);
		}

		inline void readMany(std::istream& istr)
		{
			// do nothing
		}

		template<typename _FirstTy, typename ... _RestTy>
		inline typename std::enable_if<!std::is_same<_FirstTy, MagicConstant>::value>::type readMany(std::istream& istr, _FirstTy& first, _RestTy&&... rest)
		{
			readFromStream(istr, first);
			readMany(istr, std::forward<_RestTy>(rest)...);
		}

		template<typename ... _RestTy>
		inline void readMany(std::istream& istr, MagicConstant&& first, _RestTy&&... rest)
		{
			char m[5] = {0, };
			readFromStream(istr, *(uint32_t*)m);
			if (*(uint32_t*)m != *(uint32_t*)first.m)
			{
				throw UnfitException(std::string("'") + first.m + std::string("' is needed but '") + m + std::string("'"));
			}
			readMany(istr, std::forward<_RestTy>(rest)...);
		}

		namespace detail
		{
			template<typename> struct sfinae_true : std::true_type {};
			template<typename _Ty>
			static auto testSave(int)->sfinae_true<decltype(&_Ty::serializerWrite)> ;
			template<typename _Ty>
			static auto testSave(long)->std::false_type;

			template<typename _Ty>
			static auto testLoad(int)->sfinae_true<decltype(&_Ty::serializerRead)>;
			template<typename _Ty>
			static auto testLoad(long)->std::false_type;
		}
		template<typename _Ty>
		struct hasSave : decltype(detail::testSave<_Ty>(0)){};

		template<typename _Ty>
		struct hasLoad : decltype(detail::testLoad<_Ty>(0)){};

		template<class _Ty>
		inline typename std::enable_if<std::is_fundamental<_Ty>::value>::type writeToBinStreamImpl(std::ostream& ostr, const _Ty& v)
		{
			if (!ostr.write((const char*)&v, sizeof(_Ty)))
				throw std::ios_base::failure(std::string("writing type '") + typeid(_Ty).name() + std::string("' is failed") );
		}

		template<class _Ty>
		inline typename std::enable_if<std::is_fundamental<_Ty>::value>::type readFromBinStreamImpl(std::istream& istr, _Ty& v)
		{
			if (!istr.read((char*)&v, sizeof(_Ty)))
				throw std::ios_base::failure(std::string("reading type '") + typeid(_Ty).name() + std::string("' is failed") );
		}

		template<class _Ty>
		inline typename std::enable_if<hasSave<_Ty>::value>::type writeToBinStreamImpl(std::ostream& ostr, const _Ty& v)
		{
			v.serializerWrite(ostr);
		}

		template<class _Ty>
		inline typename std::enable_if<hasLoad<_Ty>::value>::type readFromBinStreamImpl(std::istream& istr, _Ty& v)
		{
			v.serializerRead(istr);
		}


		template<class _Ty>
		inline void writeToBinStreamImpl(std::ostream& ostr, const Eigen::Matrix<_Ty, -1, -1>& v)
		{
			writeToStream<uint32_t>(ostr, v.rows());
			writeToStream<uint32_t>(ostr, v.cols());
			if (!ostr.write((const char*)v.data(), sizeof(_Ty) * v.size()))
				throw std::ios_base::failure( std::string("writing type '") + typeid(_Ty).name() + std::string("' is failed") );
		}

		template<class _Ty>
		inline void readFromBinStreamImpl(std::istream& istr, Eigen::Matrix<_Ty, -1, -1>& v)
		{
			uint32_t rows = readFromStream<uint32_t>(istr);
			uint32_t cols = readFromStream<uint32_t>(istr);
			v = Eigen::Matrix<_Ty, -1, -1>::Zero(rows, cols);
			if (!istr.read((char*)v.data(), sizeof(_Ty) * rows * cols))
				throw std::ios_base::failure( std::string("reading type '") + typeid(_Ty).name() + std::string("' is failed") );
		}

		template<class _Ty>
		inline void writeToBinStreamImpl(std::ostream& ostr, const Eigen::Matrix<_Ty, -1, 1>& v)
		{
			writeToStream<uint32_t>(ostr, v.rows());
			writeToStream<uint32_t>(ostr, v.cols());
			if (!ostr.write((const char*)v.data(), sizeof(_Ty) * v.size()))
				throw std::ios_base::failure( std::string("writing type '") + typeid(_Ty).name() + std::string("' is failed") );
		}

		template<class _Ty>
		inline void readFromBinStreamImpl(std::istream& istr, Eigen::Matrix<_Ty, -1, 1>& v)
		{
			uint32_t rows = readFromStream<uint32_t>(istr);
			uint32_t cols = readFromStream<uint32_t>(istr);
			if (cols != 1) throw std::ios_base::failure( "matrix cols != 1'" );
			v = Eigen::Matrix<_Ty, -1, 1>::Zero(rows);
			if (!istr.read((char*)v.data(), sizeof(_Ty) * rows * cols))
				throw std::ios_base::failure( std::string("reading type '") + typeid(_Ty).name() + std::string("' is failed") );
		}

		template<class _Ty>
		inline void writeToBinStreamImpl(std::ostream& ostr, const std::vector<_Ty>& v)
		{
			writeToStream<uint32_t>(ostr, v.size());
			for (auto& e : v) writeToStream(ostr, e);
		}

		template<class _Ty>
		inline void readFromBinStreamImpl(std::istream& istr, std::vector<_Ty>& v)
		{
			uint32_t size = readFromStream<uint32_t>(istr);
			v.resize(size);
			for (auto& e : v) readFromStream(istr, e);
		}

		template<class _Ty, size_t _N>
		inline void writeToBinStreamImpl(std::ostream& ostr, const std::array<_Ty, _N>& v)
		{
			writeToStream<uint32_t>(ostr, v.size());
			for (auto& e : v) writeToStream(ostr, e);
		}

		template<class _Ty, size_t _N>
		inline void readFromBinStreamImpl(std::istream& istr, std::array<_Ty, _N>& v)
		{
			uint32_t size = readFromStream<uint32_t>(istr);
			if (_N != size) throw std::ios_base::failure( text::format("the size of array must be %zd, not %zd", _N, size) );
			for (auto& e : v) readFromStream(istr, e);
		}

		template<class _Ty>
		inline void writeToBinStreamImpl(std::ostream& ostr, const tvector<_Ty>& v)
		{
			writeToStream<uint32_t>(ostr, (uint32_t)v.size());
			for (auto& e : v) writeToStream(ostr, e);
		}

		template<class _Ty>
		inline void readFromBinStreamImpl(std::istream& istr, tvector<_Ty>& v)
		{
			uint32_t size = readFromStream<uint32_t>(istr);
			v.resize(size);
			for (auto& e : v) readFromStream(istr, e);
		}

		template<class _Ty>
		inline void writeToBinStreamImpl(std::ostream& ostr, const std::basic_string<_Ty>& v)
		{
			writeToStream<uint32_t>(ostr, (uint32_t)v.size());
			if (!ostr.write((const char*)v.data(), sizeof(_Ty) * v.size()))
				throw std::ios_base::failure( std::string("writing type '") + typeid(_Ty).name() + std::string("' is failed") );
		}

		template<class _Ty>
		inline void readFromBinStreamImpl(std::istream& istr, std::basic_string<_Ty>& v)
		{
			uint32_t size = readFromStream<uint32_t>(istr);
			v.resize(size);
			if (!istr.read((char*)v.data(), sizeof(_Ty) * v.size()))
				throw std::ios_base::failure( std::string("reading type '") + typeid(_Ty).name() + std::string("' is failed") );
		}

		template<class _Ty>
		inline typename std::enable_if<std::is_abstract<_Ty>::value>::type writeToBinStreamImpl(std::ostream& ostr, const std::unique_ptr<_Ty>& v)
		{
			_Ty::serializerWrite(v, ostr);
		}

		template<class _Ty>
		inline typename std::enable_if<std::is_abstract<_Ty>::value>::type readFromBinStreamImpl(std::istream& istr, std::unique_ptr<_Ty>& v)
		{
			_Ty::serializerRead(v, istr);
		}

		template<typename _Ty> 
		inline void writeToStream(std::ostream& ostr, const _Ty& v)
		{
			return writeToBinStreamImpl(ostr, v);
		}

		template<typename _Ty> 
		inline void readFromStream(std::istream& istr, _Ty& v)
		{
			return readFromBinStreamImpl(istr, v);
		}

		template<typename _Ty>
		inline _Ty readFromStream(std::istream& istr)
		{
			_Ty v;
			readFromBinStreamImpl(istr, v);
			return v;
		}
	}
}

#define DEFINE_SERIALIZER(...) void serializerRead(std::istream& istr)\
{\
	tomoto::serializer::readMany(istr, __VA_ARGS__);\
}\
void serializerWrite(std::ostream& ostr) const\
{\
	tomoto::serializer::writeMany(ostr, __VA_ARGS__);\
}


#define DEFINE_SERIALIZER_CALLBACK(onRead, ...) void serializerRead(std::istream& istr)\
{\
	tomoto::serializer::readMany(istr, __VA_ARGS__);\
	this->onRead();\
}\
void serializerWrite(std::ostream& ostr) const\
{\
	tomoto::serializer::writeMany(ostr, __VA_ARGS__);\
}


#define DEFINE_SERIALIZER_AFTER_BASE(base, ...) void serializerRead(std::istream& istr)\
{\
	base::serializerRead(istr);\
	tomoto::serializer::readMany(istr, __VA_ARGS__);\
}\
void serializerWrite(std::ostream& ostr) const\
{\
	base::serializerWrite(ostr);\
	tomoto::serializer::writeMany(ostr, __VA_ARGS__);\
}

#define DEFINE_SERIALIZER_AFTER_BASE2(base1, base2, ...) void serializerRead(std::istream& istr)\
{\
	base1, base2::serializerRead(istr);\
	tomoto::serializer::readMany(istr, __VA_ARGS__);\
}\
void serializerWrite(std::ostream& ostr) const\
{\
	base1, base2::serializerWrite(ostr);\
	tomoto::serializer::writeMany(ostr, __VA_ARGS__);\
}

#define DEFINE_SERIALIZER_AFTER_BASE_CALLBACK(base, onRead, ...) void serializerRead(std::istream& istr)\
{\
	base::serializerRead(istr);\
	tomoto::serializer::readMany(istr, __VA_ARGS__);\
	this->onRead();\
}\
void serializerWrite(std::ostream& ostr) const\
{\
	base::serializerWrite(ostr);\
	tomoto::serializer::writeMany(ostr, __VA_ARGS__);\
}

#define DEFINE_SERIALIZER_AFTER_BASE2_CALLBACK(base1, base2, onRead, ...) void serializerRead(std::istream& istr)\
{\
	base1, base2::serializerRead(istr);\
	tomoto::serializer::readMany(istr, __VA_ARGS__);\
	this->onRead();\
}\
void serializerWrite(std::ostream& ostr) const\
{\
	base1, base2::serializerWrite(ostr);\
	tomoto::serializer::writeMany(ostr, __VA_ARGS__);\
}

#define DEFINE_SERIALIZER_VIRTUAL(...) virtual void serializerRead(std::istream& istr)\
{\
	tomoto::serializer::readMany(istr, __VA_ARGS__);\
}\
virtual void serializerWrite(std::ostream& ostr) const\
{\
	tomoto::serializer::writeMany(ostr, __VA_ARGS__);\
}
