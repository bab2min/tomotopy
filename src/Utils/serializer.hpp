#pragma once
#include <iostream>
#include <memory>
#include <string>
#include <array>
#include <type_traits>
#include <Eigen/Dense>
#include <vector>
#include "tvector.hpp"
#include "text.hpp"

/*

A new serializer format for tomotopy 0.6.0

struct TaggedData
{
	char[4] magic_id;
	uint16_t major_version, minor_version;
	uint64_t tot_size;
	uint32_t key_size, trailing_data_cnt;
	char[key_size] key;
	char[...] data;
}

*/

namespace tomoto
{
	namespace serializer
	{
		namespace detail
		{
			template<class _T> using Invoke = typename _T::type;

			template<size_t...> struct seq { using type = seq; };

			template<class _S1, class _S2> struct concat;

			template<size_t... _i1, size_t... _i2>
			struct concat<seq<_i1...>, seq<_i2...>>
				: seq<_i1..., (sizeof...(_i1) + _i2)...> {};

			template<class _S1, class _S2>
			using Concat = Invoke<concat<_S1, _S2>>;

			template<size_t _n> struct gen_seq;
			template<size_t _n> using GenSeq = Invoke<gen_seq<_n>>;

			template<size_t _n>
			struct gen_seq : Concat<GenSeq<_n / 2>, GenSeq<_n - _n / 2>> {};

			template<> struct gen_seq<0> : seq<> {};
			template<> struct gen_seq<1> : seq<0> {};

			template <size_t _n, size_t ... _is>
			std::array<char, _n - 1> to_array(const char(&a)[_n], seq<_is...>)
			{
				return { {a[_is]...} };
			}

			template <size_t _n>
			constexpr std::array<char, _n - 1> to_array(const char(&a)[_n])
			{
				return to_array(a, GenSeq<_n - 1>{});
			}

			template <size_t _n, size_t ... _is>
			std::array<char, _n> to_arrayz(const char(&a)[_n], seq<_is...>)
			{
				return { {a[_is]..., 0} };
			}

			template <size_t _n>
			constexpr std::array<char, _n> to_arrayz(const char(&a)[_n])
			{
				return to_arrayz(a, GenSeq<_n - 1>{});
			}
		}

		template<typename _Ty> inline void writeToStream(std::ostream& ostr, const _Ty& v);
		template<typename _Ty> inline void readFromStream(std::istream& istr, _Ty& v);
		template<typename _Ty> inline _Ty readFromStream(std::istream& istr);

		class UnfitException : public std::ios_base::failure
		{
			using std::ios_base::failure::failure;
		};

		template<size_t _len>
		struct Key
		{
			std::array<char, _len> m;

			std::string str() const
			{
				return std::string{ m.begin(), m.end() };
			}

			Key(const std::array<char, _len>& _m) : m(_m)
			{
			}

			Key(std::array<char, _len>&& _m) : m(_m)
			{
			}

			Key(const char(&a)[_len + 1]) : Key{ detail::to_array(a) }
			{
			}
		};

		template<typename _Ty>
		struct is_key : public std::false_type
		{
		};

		template<size_t _len>
		struct is_key<Key<_len>> : public std::true_type
		{
		};

		template<size_t _n>
		constexpr Key<_n - 1> to_key(const char(&a)[_n])
		{
			return Key<_n - 1>{detail::to_array(a)};
		}

		template<size_t _n>
		constexpr Key<_n> to_keyz(const char(&a)[_n])
		{
			return Key<_n>{detail::to_arrayz(a)};
		}

		inline void writeMany(std::ostream& ostr)
		{
			// do nothing
		}

		template<typename _FirstTy, typename ... _RestTy>
		inline typename std::enable_if<!is_key<_FirstTy>::value>::type writeMany(std::ostream& ostr, const _FirstTy& first, _RestTy&&... rest)
		{
			writeToStream(ostr, first);
			writeMany(ostr, std::forward<_RestTy>(rest)...);
		}

		template<size_t _len, typename ... _RestTy>
		inline void writeMany(std::ostream& ostr, const Key<_len>& first, _RestTy&&... rest)
		{
			ostr.write(first.m.data(), first.m.size());
			writeMany(ostr, std::forward<_RestTy>(rest)...);
		}

		inline void readMany(std::istream& istr)
		{
			// do nothing
		}

		template<typename _FirstTy, typename ... _RestTy>
		inline typename std::enable_if<!is_key<_FirstTy>::value>::type readMany(std::istream& istr, _FirstTy& first, _RestTy&&... rest)
		{
			readFromStream(istr, first);
			readMany(istr, std::forward<_RestTy>(rest)...);
		}

		template<size_t _len, typename ... _RestTy>
		inline void readMany(std::istream& istr, const Key<_len>& first, _RestTy&&... rest)
		{
			std::array<char, _len> m;
			istr.read(m.data(), m.size());
			if (m != first.m)
			{
				throw UnfitException(std::string("'") + first.str() + std::string("' is needed but '") + std::string{ m.begin(), m.end() } +std::string("'"));
			}
			readMany(istr, std::forward<_RestTy>(rest)...);
		}

		template<size_t _len>
		inline bool readTest(std::istream& istr, const Key<_len>& first)
		{
			std::array<char, _len> m;
			istr.read(m.data(), m.size());
			return m == first.m;
		}

		template<size_t>
		struct version_holder {};

		namespace detail
		{
			template<typename _Class, typename _RetTy, typename ..._Args>
			_RetTy test_mf(_RetTy(_Class::*mf)(_Args...))
			{
				return _RetTy{};
			}

			template<typename _Class, typename _RetTy, typename ..._Args>
			_RetTy test_mf_c(_RetTy(_Class::*mf)(_Args...) const)
			{
				return _RetTy{};
			}

			template<typename> struct sfinae_true : std::true_type {};
			template<typename _Ty>
			static auto testSave(int)->sfinae_true<decltype(test_mf_c<_Ty, void, std::ostream&>(&_Ty::serializerWrite))>;
			template<typename _Ty>
			static auto testSave(long)->std::false_type;

			template<typename _Ty>
			static auto testLoad(int)->sfinae_true<decltype(test_mf<_Ty, void, std::istream&>(&_Ty::serializerRead))>;
			template<typename _Ty>
			static auto testLoad(long)->std::false_type;

			template<typename _Ty, size_t _version>
			static auto testSaveV(int)->sfinae_true<decltype(test_mf_c<_Ty, void, version_holder<_version>, std::ostream&>(&_Ty::serializerWrite))>;
			template<typename _Ty, size_t _version>
			static auto testSaveV(long)->std::false_type;

			template<typename _Ty, size_t _version>
			static auto testLoadV(int)->sfinae_true<decltype(test_mf<_Ty, void, version_holder<_version>, std::istream&>(&_Ty::serializerRead))>;
			template<typename _Ty, size_t _version>
			static auto testLoadV(long)->std::false_type;
		}
		template<typename _Ty>
		struct hasSave : decltype(detail::testSave<_Ty>(0)){};

		template<typename _Ty>
		struct hasLoad : decltype(detail::testLoad<_Ty>(0)){};

		template<typename _Ty, size_t _version>
		struct hasSaveV : decltype(detail::testSaveV<_Ty, _version>(0)){};

		template<typename _Ty, size_t _version>
		struct hasLoadV : decltype(detail::testLoadV<_Ty, _version>(0)){};

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

		template<class _Ty, size_t _version = 0>
		inline typename std::enable_if<
			hasSaveV<_Ty, _version>::value && !hasSaveV<_Ty, _version + 1>::value
		>::type writeToBinStreamImpl(std::ostream& ostr, const _Ty& v)
		{
			v.serializerWrite(version_holder<_version>{}, ostr);
		}

		template<class _Ty, size_t _version = 0>
		inline typename std::enable_if<
			hasSaveV<_Ty, _version>::value && hasSaveV<_Ty, _version + 1>::value
		>::type writeToBinStreamImpl(std::ostream& ostr, const _Ty& v)
		{
			return writeToBinStreamImpl<_Ty, _version + 1>(ostr, v);
		}

		template<class _Ty>
		inline typename std::enable_if<hasLoad<_Ty>::value>::type readFromBinStreamImpl(std::istream& istr, _Ty& v)
		{
			v.serializerRead(istr);
		}

		template<class _Ty, size_t _version = 0>
		inline typename std::enable_if<
			hasLoadV<_Ty, _version>::value && !hasLoadV<_Ty, _version + 1>::value
		>::type readFromBinStreamImpl(std::istream& istr, _Ty& v)
		{
			v.serializerRead(version_holder<_version>{}, istr);
		}

		template<class _Ty, size_t _version = 0>
		inline typename std::enable_if<
			hasLoadV<_Ty, _version>::value && hasLoadV<_Ty, _version + 1>::value
		>::type readFromBinStreamImpl(std::istream& istr, _Ty& v)
		{
			auto pos = istr.tellg();
			try
			{
				// try higher version first
				return readFromBinStreamImpl<_Ty, _version + 1>(istr, v);
			}
			catch (const std::ios_base::failure&)
			{
				istr.seekg(pos);
				// try current version if fails
				v.serializerRead(version_holder<_version>{}, istr);
			}
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

		template<class _Ty1, class _Ty2>
		inline void writeToBinStreamImpl(std::ostream& ostr, const std::pair<_Ty1, _Ty2>& v)
		{
			writeToStream(ostr, v.first);
			writeToStream(ostr, v.second);
		}

		template<class _Ty1, class _Ty2>
		inline void readFromBinStreamImpl(std::istream& istr, std::pair<_Ty1, _Ty2>& v)
		{
			readFromStream(istr, v.first);
			readFromStream(istr, v.second);
		}

		template<class _KeyTy, class _ValTy>
		inline void writeToBinStreamImpl(std::ostream& ostr, const std::unordered_map<_KeyTy, _ValTy>& v)
		{
			writeToStream<uint32_t>(ostr, v.size());
			for (auto& e : v) writeToStream(ostr, e);
		}

		template<class _KeyTy, class _ValTy>
		inline void readFromBinStreamImpl(std::istream& istr, std::unordered_map<_KeyTy, _ValTy>& v)
		{
			uint32_t size = readFromStream<uint32_t>(istr);
			v.clear();
			for (size_t i = 0; i < size; ++i)
			{
				v.emplace(readFromStream<std::pair<_KeyTy, _ValTy>>(istr));
			}
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

		static auto taggedDataKey = to_key("TPTK");

		template<size_t _len, typename _Ty>
		inline void writeTaggedData(std::ostream& ostr, uint32_t version, uint32_t trailing_cnt, const Key<_len>& key, const _Ty& data)
		{
			uint16_t major = version >> 16, minor = version & 0xFFFF;
			writeMany(ostr, taggedDataKey, version);
			std::streampos totsize_pos = ostr.tellp();
			writeMany(ostr, (uint64_t)0, (uint32_t)_len, trailing_cnt, key, data);
			std::streampos end_pos = ostr.tellp();
			ostr.seekp(totsize_pos);
			writeMany(ostr, (uint64_t)(end_pos - totsize_pos));
			ostr.seekp(end_pos);
		}

		template<size_t _len, typename _Ty>
		inline std::pair<bool, std::streampos> readTaggedData(std::istream& istr, uint32_t version, uint32_t& trailing_cnt, const Key<_len>& key, _Ty& data)
		{
			uint16_t major = version >> 16, minor = version & 0xFFFF;
			uint64_t totsize;
			uint32_t keysize;
			std::streampos start_pos = istr.tellg();
			readMany(istr, taggedDataKey, version);
			std::streampos totsize_pos = istr.tellg();
			readMany(istr, totsize, keysize, trailing_cnt);
			std::streampos end_pos = totsize_pos + (std::streamoff)totsize;
			if (_len != keysize)
			{
				istr.seekg(start_pos);
				return std::make_pair(false, end_pos);
			}
			
			if (!readTest(istr, key))
			{
				istr.seekg(start_pos);
				return std::make_pair(false, end_pos);
			}
			
			readMany(istr, data);
			if (end_pos != istr.tellg())
			{
				istr.seekg(start_pos);
				return std::make_pair(false, end_pos);
			}
			return std::make_pair(true, end_pos);
		}

		inline void readTaggedMany(std::istream& istr, uint32_t version)
		{
			// seek to the end of tagged data list
			uint32_t trailing_cnt;
			do
			{
				uint64_t totsize;
				uint32_t keysize;
				readMany(istr, taggedDataKey, version);
				std::streampos totsize_pos = istr.tellg();
				readMany(istr, totsize, keysize, trailing_cnt);
				istr.seekg(totsize_pos + (std::streamoff)totsize);

			} while (trailing_cnt);
		}

		template<size_t _len, typename _Ty, typename ... _Rest>
		inline void readTaggedMany(std::istream& istr, uint32_t version, const Key<_len>& key, _Ty& data, _Rest&&... rest)
		{
			auto start_pos = istr.tellg();
			uint32_t trailing_cnt;
			do
			{
				std::pair<bool, std::streampos> p = readTaggedData(istr, version, trailing_cnt, key, data);
				if (p.first)
				{
					break;
				}
				else
				{
					istr.seekg(p.second);
				}
			} while (trailing_cnt);

			istr.seekg(start_pos);
			readTaggedMany(istr, version, std::forward<_Rest>(rest)...);
		}

		inline void writeTaggedMany(std::ostream& ostr, uint32_t version)
		{
			// do nothing
		}

		template<size_t _len, typename _Ty, typename ... _Rest>
		inline void writeTaggedMany(std::ostream& ostr, uint32_t version, const Key<_len>& key, const _Ty& data, _Rest&&... rest)
		{
			writeTaggedData(ostr, version, sizeof...(_Rest) / 2, key, data);
			writeTaggedMany(ostr, version, std::forward<_Rest>(rest)...);
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

#define DEFINE_SERIALIZER_WITH_VERSION(v,...) void serializerRead(tomoto::serializer::version_holder<v>, std::istream& istr)\
{\
	tomoto::serializer::readMany(istr, __VA_ARGS__);\
}\
void serializerWrite(tomoto::serializer::version_holder<v>, std::ostream& ostr) const\
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

#define DEFINE_SERIALIZER_AFTER_BASE_WITH_VERSION(base, v, ...) void serializerRead(tomoto::serializer::version_holder<v> _v, std::istream& istr)\
{\
	base::serializerRead(_v, istr);\
	tomoto::serializer::readMany(istr, __VA_ARGS__);\
}\
void serializerWrite(tomoto::serializer::version_holder<v> _v, std::ostream& ostr) const\
{\
	base::serializerWrite(_v, ostr);\
	tomoto::serializer::writeMany(ostr, __VA_ARGS__);\
}

#define DEFINE_SERIALIZER_BASE_WITH_VERSION(base, v) void serializerRead(tomoto::serializer::version_holder<v> _v, std::istream& istr)\
{\
	base::serializerRead(_v, istr);\
}\
void serializerWrite(tomoto::serializer::version_holder<v> _v, std::ostream& ostr) const\
{\
	base::serializerWrite(_v, ostr);\
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

#define _TO_KEY_VALUE_0()
#define _TO_KEY_VALUE_1(a) tomoto::serializer::to_keyz(#a), a
#define _TO_KEY_VALUE_2(a, b) _TO_KEY_VALUE_1(a), _TO_KEY_VALUE_1(b)
#define _TO_KEY_VALUE_3(a, b, c) _TO_KEY_VALUE_2(a, b), _TO_KEY_VALUE_1(c)
#define _TO_KEY_VALUE_4(a, b, c, d) _TO_KEY_VALUE_2(a, b), _TO_KEY_VALUE_2(c, d)
#define _TO_KEY_VALUE_5(a, b, c, d, e) _TO_KEY_VALUE_3(a, b, c), _TO_KEY_VALUE_2(d, e)
#define _TO_KEY_VALUE_6(a, b, c, d, e, f) _TO_KEY_VALUE_3(a, b, c), _TO_KEY_VALUE_3(d, e, f)
#define _TO_KEY_VALUE_7(a, b, c, d, e, f, g) _TO_KEY_VALUE_4(a, b, c, d), _TO_KEY_VALUE_3(e, f, g)
#define _TO_KEY_VALUE_8(a, b, c, d, e, f, g, h) _TO_KEY_VALUE_4(a, b, c, d), _TO_KEY_VALUE_4(e, f, g, h)

#define _EXPAND(x) x
#define _TO_KEY_VALUE_K(_1, _2, _3, _4, _5, _6, _7, _8, N, ...) _TO_KEY_VALUE_ ## N
#define _TO_KEY_VALUE(...) _EXPAND( _TO_KEY_VALUE_K(__VA_ARGS__, 8, 7, 6, 5, 4, 3, 2, 1, 0)( __VA_ARGS__ ) )

#define DEFINE_TAGGED_SERIALIZER_WITH_VERSION(v, t,...) void serializerRead(tomoto::serializer::version_holder<v>, std::istream& istr)\
{\
	tomoto::serializer::readTaggedMany(istr, t, _TO_KEY_VALUE(__VA_ARGS__));\
}\
void serializerWrite(tomoto::serializer::version_holder<v>, std::ostream& ostr) const\
{\
	tomoto::serializer::writeTaggedMany(ostr, t, _TO_KEY_VALUE(__VA_ARGS__));\
}

#define DEFINE_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(base, v, t,...) void serializerRead(tomoto::serializer::version_holder<v> _v, std::istream& istr)\
{\
	base::serializerRead(_v, istr);\
	tomoto::serializer::readTaggedMany(istr, t, _TO_KEY_VALUE(__VA_ARGS__));\
}\
void serializerWrite(tomoto::serializer::version_holder<v> _v, std::ostream& ostr) const\
{\
	base::serializerWrite(_v, ostr);\
	tomoto::serializer::writeTaggedMany(ostr, t, _TO_KEY_VALUE(__VA_ARGS__));\
}
