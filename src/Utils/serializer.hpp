#pragma once
#include <iostream>
#include <memory>
#include <string>
#include <array>
#include <type_traits>
#include <vector>
#include <map>
#include <unordered_map>
#include <Eigen/Dense>
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
		struct membuf : std::streambuf 
		{
			membuf(char* base, std::ptrdiff_t n) 
			{
				this->setg(base, base, base + n);
			}

			pos_type seekpos(pos_type sp, std::ios_base::openmode which) override {
				return seekoff(sp - pos_type(off_type(0)), std::ios_base::beg, which);
			}

			pos_type seekoff(off_type off,
				std::ios_base::seekdir dir,
				std::ios_base::openmode which = std::ios_base::in) override {
				if (dir == std::ios_base::cur)
					gbump(off);
				else if (dir == std::ios_base::end)
					setg(eback(), egptr() + off, egptr());
				else if (dir == std::ios_base::beg)
					setg(eback(), eback() + off, egptr());
				return gptr() - eback();
			}
		};

		class imstream : public std::istream
		{
			membuf buf;
		public:
			imstream(const char* base, std::ptrdiff_t n)
				: std::istream(&buf), buf((char*)base, n)
			{
			}
		};

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
			constexpr std::array<char, sizeof... (_is)> to_array(const char(&a)[_n], seq<_is...>)
			{
				return { {a[_is]...} };
			}

			template <size_t _n>
			constexpr std::array<char, _n - 1> to_array(const char(&a)[_n])
			{
				return to_array(a, GenSeq<_n - 1>{});
			}

			template <size_t _n, size_t ... _is>
			constexpr std::array<char, sizeof... (_is) + 1> to_arrayz(const char(&a)[_n], seq<_is...>)
			{
				return { {a[_is]..., 0} };
			}

			template <size_t _n, size_t ... _is>
			constexpr std::array<char, sizeof... (_is) + 1> to_arrayz(const std::array<char, _n>& a, seq<_is...>)
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

			constexpr Key(const std::array<char, _len>& _m) : m(_m)
			{
			}

			constexpr Key(std::array<char, _len>&& _m) : m(_m)
			{
			}

			constexpr Key(const char(&a)[_len + 1]) : Key{ detail::to_array(a) }
			{
			}

			constexpr char operator[](size_t n) const
			{
				return n < _len ? m[n] : throw std::out_of_range("");
			}

			constexpr size_t size() const { return _len; }
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
		constexpr Key<_n> to_key(const Key<_n>& key)
		{
			return key;
		}

		template<size_t _n>
		constexpr Key<_n> to_keyz(const char(&a)[_n])
		{
			return Key<_n>{detail::to_arrayz(a)};
		}

		template<size_t _n>
		constexpr Key<_n + 1> to_keyz(const Key<_n>& key)
		{
			return Key<_n + 1>{detail::to_arrayz(key.m, detail::GenSeq<_n>{})};
		}

		template<typename _Ty, typename = void>
		struct Serializer;

		template<typename _Ty, size_t _version = 0, typename = void>
		struct SerializerV;

		template<typename _Ty>
		inline void writeToStream(std::ostream& ostr, const _Ty& v)
		{
			Serializer<
				typename std::remove_const<typename std::remove_reference<_Ty>::type>::type
			>{}.write(ostr, v);
		}

		template<typename _Ty>
		inline void readFromStream(std::istream& istr, _Ty& v)
		{
			Serializer<
				typename std::remove_const<typename std::remove_reference<_Ty>::type>::type
			>{}.read(istr, v);
		}

		template<typename _Ty>
		inline _Ty readFromStream(std::istream& istr)
		{
			_Ty v;
			Serializer<
				typename std::remove_const<typename std::remove_reference<_Ty>::type>::type
			>{}.read(istr, v);
			return v;
		}

		inline void writeMany(std::ostream& ostr)
		{
			// do nothing
		}

		template<typename _FirstTy, typename ... _RestTy>
		inline typename std::enable_if<
			!is_key<typename std::remove_reference<_FirstTy>::type>::value
		>::type writeMany(std::ostream& ostr, _FirstTy&& first, _RestTy&&... rest)
		{
			writeToStream(ostr, std::forward<_FirstTy>(first));
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
		inline typename std::enable_if<
			!is_key<typename std::remove_reference<_FirstTy>::type>::value
		>::type readMany(std::istream& istr, _FirstTy&& first, _RestTy&&... rest)
		{
			readFromStream(istr, std::forward<_FirstTy>(first));
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

		template<typename _Ty>
		struct Serializer<_Ty, typename std::enable_if<std::is_fundamental<_Ty>::value>::type>
		{
			void write(std::ostream& ostr, const _Ty& v)
			{
				if (!ostr.write((const char*)&v, sizeof(_Ty)))
					throw std::ios_base::failure(std::string("writing type '") + typeid(_Ty).name() + std::string("' is failed"));
			}

			void read(std::istream& istr, _Ty& v)
			{
				if (!istr.read((char*)&v, sizeof(_Ty)))
					throw std::ios_base::failure(std::string("reading type '") + typeid(_Ty).name() + std::string("' is failed"));
			}
		};

		template<typename _Ty>
		struct Serializer<_Ty, typename std::enable_if<hasSave<_Ty>::value>::type>
		{
			void write(std::ostream& ostr, const _Ty& v)
			{
				v.serializerWrite(ostr);
			}

			void read(std::istream& istr, _Ty& v)
			{
				v.serializerRead(istr);
			}
		};

		template<typename _Ty>
		struct Serializer<_Ty, typename std::enable_if<hasSaveV<_Ty, 0>::value>::type>
		{
			void write(std::ostream& ostr, const _Ty& v)
			{
				SerializerV<_Ty>{}.write(ostr, v);
			}

			void read(std::istream& istr, _Ty& v)
			{
				SerializerV<_Ty>{}.read(istr, v);
			}
		};
		
		template<typename _Ty, size_t _version>
		struct SerializerV<_Ty, _version, typename std::enable_if<
			hasSaveV<_Ty, _version>::value && !hasSaveV<_Ty, _version + 1>::value
		>::type>
		{
			void write(std::ostream& ostr, const _Ty& v)
			{
				v.serializerWrite(version_holder<_version>{}, ostr);
			}

			void read(std::istream& istr, _Ty& v)
			{
				v.serializerRead(version_holder<_version>{}, istr);
			}
		};

		template<typename _Ty, size_t _version>
		struct SerializerV<_Ty, _version, typename std::enable_if<
			hasSaveV<_Ty, _version>::value && hasSaveV<_Ty, _version + 1>::value
		>::type>
		{
			void write(std::ostream& ostr, const _Ty& v)
			{
				SerializerV<_Ty, _version + 1>{}.write(ostr, v);
			}

			void read(std::istream& istr, _Ty& v)
			{
				auto pos = istr.tellg();
				try
				{
					// try higher version first
					return SerializerV<_Ty, _version + 1>{}.read(istr, v);
				}
				catch (const std::ios_base::failure&)
				{
					istr.seekg(pos);
					// try current version if fails
					v.serializerRead(version_holder<_version>{}, istr);
				}
			}
		};

		template<typename _Ty>
		struct Serializer<Eigen::Matrix<_Ty, -1, -1>>
		{
			using VTy = Eigen::Matrix<_Ty, -1, -1>;
			void write(std::ostream& ostr, const VTy& v)
			{
				writeMany(ostr, (uint32_t)v.rows(), (uint32_t)v.cols());
				if (!ostr.write((const char*)v.data(), sizeof(_Ty) * v.size()))
					throw std::ios_base::failure(std::string("writing type '") + typeid(_Ty).name() + std::string("' is failed"));
			}

			void read(std::istream& istr, VTy& v)
			{
				uint32_t rows, cols;
				readMany(istr, rows, cols);
				v = Eigen::Matrix<_Ty, -1, -1>::Zero(rows, cols);
				if (!istr.read((char*)v.data(), sizeof(_Ty) * rows * cols))
					throw std::ios_base::failure(std::string("reading type '") + typeid(_Ty).name() + std::string("' is failed"));
			}
		};

		template<typename _Ty>
		struct Serializer<Eigen::Matrix<_Ty, -1, 1>>
		{
			using VTy = Eigen::Matrix<_Ty, -1, 1>;
			void write(std::ostream& ostr, const VTy& v)
			{
				writeMany(ostr, (uint32_t)v.rows(), (uint32_t)v.cols());
				if (!ostr.write((const char*)v.data(), sizeof(_Ty) * v.size()))
					throw std::ios_base::failure(std::string("writing type '") + typeid(_Ty).name() + std::string("' is failed"));
			}

			void read(std::istream& istr, VTy& v)
			{
				uint32_t rows, cols;
				readMany(istr, rows, cols);
				if (cols != 1) throw std::ios_base::failure("matrix cols != 1");
				v = Eigen::Matrix<_Ty, -1, -1>::Zero(rows, cols);
				if (!istr.read((char*)v.data(), sizeof(_Ty) * rows * cols))
					throw std::ios_base::failure(std::string("reading type '") + typeid(_Ty).name() + std::string("' is failed"));
			}
		};

		template<typename _Ty>
		struct Serializer<PreventCopy<_Ty>> : public Serializer<_Ty>
		{
		};

		template<typename _Ty, typename _Ty2>
		struct Serializer<DelegateCopy<_Ty, _Ty2>> : public Serializer<_Ty>
		{
		};

		template<typename _Ty>
		struct Serializer<std::vector<_Ty>, typename std::enable_if<std::is_fundamental<_Ty>::value>::type>
		{
			using VTy = std::vector<_Ty>;
			void write(std::ostream& ostr, const VTy& v)
			{
				writeToStream(ostr, (uint32_t)v.size());
				if (!ostr.write((const char*)v.data(), sizeof(_Ty) * v.size()))
					throw std::ios_base::failure(std::string("writing type '") + typeid(_Ty).name() + std::string("' is failed"));
			}

			void read(std::istream& istr, VTy& v)
			{
				auto size = readFromStream<uint32_t>(istr);
				v.resize(size);
				if (!istr.read((char*)v.data(), sizeof(_Ty) * size))
					throw std::ios_base::failure(std::string("reading type '") + typeid(_Ty).name() + std::string("' is failed"));
			}
		};

		template<typename _Ty>
		struct Serializer<std::vector<_Ty>, typename std::enable_if<!std::is_fundamental<_Ty>::value>::type>
		{
			using VTy = std::vector<_Ty>;
			void write(std::ostream& ostr, const VTy& v)
			{
				writeToStream(ostr, (uint32_t)v.size());
				for (auto& e : v) Serializer<_Ty>{}.write(ostr, e);
			}

			void read(std::istream& istr, VTy& v)
			{
				auto size = readFromStream<uint32_t>(istr);
				v.resize(size);
				for (auto& e : v) Serializer<_Ty>{}.read(istr, e);
			}
		};

		template<typename _Ty, size_t n>
		struct Serializer<std::array<_Ty, n>, typename std::enable_if<std::is_fundamental<_Ty>::value>::type>
		{
			using VTy = std::array<_Ty, n>;
			void write(std::ostream& ostr, const VTy& v)
			{
				writeToStream(ostr, (uint32_t)v.size());
				if (!ostr.write((const char*)v.data(), sizeof(_Ty) * v.size()))
					throw std::ios_base::failure(std::string("writing type '") + typeid(_Ty).name() + std::string("' is failed"));
			}

			void read(std::istream& istr, VTy& v)
			{
				auto size = readFromStream<uint32_t>(istr);
				if (n != size) throw std::ios_base::failure(text::format("the size of array must be %zd, not %zd", n, size));
				if (!istr.read((char*)v.data(), sizeof(_Ty) * size))
					throw std::ios_base::failure(std::string("reading type '") + typeid(_Ty).name() + std::string("' is failed"));
			}
		};

		template<typename _Ty, size_t n>
		struct Serializer<std::array<_Ty, n>, typename std::enable_if<!std::is_fundamental<_Ty>::value>::type>
		{
			using VTy = std::array<_Ty, n>;
			void write(std::ostream& ostr, const VTy& v)
			{
				writeToStream(ostr, (uint32_t)v.size());
				for (auto& e : v) Serializer<_Ty>{}.write(ostr, e);
			}

			void read(std::istream& istr, VTy& v)
			{
				auto size = readFromStream<uint32_t>(istr);
				if (n != size) throw std::ios_base::failure(text::format("the size of array must be %zd, not %zd", n, size));
				for (auto& e : v) Serializer<_Ty>{}.read(istr, e);
			}
		};

		template<typename _Ty>
		struct Serializer<std::basic_string<_Ty>>
		{
			using VTy = std::basic_string<_Ty>;
			void write(std::ostream& ostr, const VTy& v)
			{
				writeToStream(ostr, (uint32_t)v.size());
				if (!ostr.write((const char*)v.data(), sizeof(_Ty) * v.size()))
					throw std::ios_base::failure(std::string("writing type '") + typeid(_Ty).name() + std::string("' is failed"));
			}

			void read(std::istream& istr, VTy& v)
			{
				auto size = readFromStream<uint32_t>(istr);
				v.resize(size);
				if (!istr.read((char*)v.data(), sizeof(_Ty) * size))
					throw std::ios_base::failure(std::string("reading type '") + typeid(_Ty).name() + std::string("' is failed"));
			}
		};

		template<typename _Ty1, typename _Ty2>
		struct Serializer<std::pair<_Ty1, _Ty2>>
		{
			using VTy = std::pair<_Ty1, _Ty2>;
			void write(std::ostream& ostr, const VTy& v)
			{
				writeMany(ostr, v.first, v.second);
			}

			void read(std::istream& istr, VTy& v)
			{
				readMany(istr, v.first, v.second);
			}
		};

		template<typename _Ty1, typename _Ty2>
		struct Serializer<std::unordered_map<_Ty1, _Ty2>>
		{
			using VTy = std::unordered_map<_Ty1, _Ty2>;
			void write(std::ostream& ostr, const VTy& v)
			{
				writeToStream(ostr, (uint32_t)v.size());
				for (auto& e : v) writeToStream(ostr, e);
			}

			void read(std::istream& istr, VTy& v)
			{
				auto size = readFromStream<uint32_t>(istr);
				v.clear();
				for (size_t i = 0; i < size; ++i)
				{
					v.emplace(readFromStream<std::pair<_Ty1, _Ty2>>(istr));
				}
			}
		};

		template<typename _Ty1, typename _Ty2>
		struct Serializer<std::map<_Ty1, _Ty2>>
		{
			using VTy = std::map<_Ty1, _Ty2>;
			void write(std::ostream& ostr, const VTy& v)
			{
				writeToStream(ostr, (uint32_t)v.size());
				for (auto& e : v) writeToStream(ostr, e);
			}

			void read(std::istream& istr, VTy& v)
			{
				auto size = readFromStream<uint32_t>(istr);
				v.clear();
				for (size_t i = 0; i < size; ++i)
				{
					v.emplace(readFromStream<std::pair<_Ty1, _Ty2>>(istr));
				}
			}
		};

		template<typename _Ty>
		struct Serializer<std::unique_ptr<_Ty>, typename std::enable_if<std::is_abstract<_Ty>::value>::type>
		{
			using VTy = std::unique_ptr<_Ty>;
			void write(std::ostream& ostr, const VTy& v)
			{
				_Ty::serializerWrite(v, ostr);
			}

			void read(std::istream& istr, VTy& v)
			{
				_Ty::serializerRead(v, istr);
			}
		};

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
#define _TO_KEY_VALUE_9(a, b, c, d, e, f, g, h, i) _TO_KEY_VALUE_5(a, b, c, d, e), _TO_KEY_VALUE_4(f, g, h, i)
#define _TO_KEY_VALUE_10(a, b, c, d, e, f, g, h, i, j) _TO_KEY_VALUE_5(a, b, c, d, e), _TO_KEY_VALUE_5(f, g, h, i, j)

#define _EXPAND(x) x
#define _TO_KEY_VALUE_K(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) _TO_KEY_VALUE_ ## N
#define _TO_KEY_VALUE(...) _EXPAND( _TO_KEY_VALUE_K(__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)( __VA_ARGS__ ) )

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
