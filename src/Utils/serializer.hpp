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
#include "Utils.hpp"

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
		struct membuf : public std::streambuf
		{
			membuf(bool read, bool write, char* base, std::ptrdiff_t n);
			~membuf();

			std::streampos seekpos(pos_type sp, std::ios_base::openmode which) override;

			std::streampos seekoff(off_type off,
				std::ios_base::seekdir dir,
				std::ios_base::openmode which = std::ios_base::in
			) override;

			const char* curptr() const
			{
				return this->gptr();
			}
		};

		class imstream : public std::istream
		{
			membuf buf;
		public:
			imstream(const char* base, std::ptrdiff_t n);
			~imstream();

			template<class Ty>
			imstream(const Ty& m) : imstream(m.get(), m.size())
			{
			}

			const char* curptr() const
			{
				return buf.curptr();
			}
		};

		class omstream : public std::ostream
		{
			membuf buf;
		public:
			omstream(char* base, std::ptrdiff_t n);
			~omstream();

			template<class Ty>
			omstream(const Ty& m) : omstream(m.get(), m.size())
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

		template<typename _Ty>
		inline uint64_t computeHash(uint64_t seed, const _Ty& v)
		{
			return Serializer<
				typename std::remove_const<typename std::remove_reference<_Ty>::type>::type
			>{}.hash(seed, v);
		}

		uint64_t computeFastHash(const void* data, size_t size, uint64_t seed = 0);

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

		inline uint64_t computeHashMany(uint64_t seed)
		{
			return seed;
		}

		template<typename _FirstTy, typename ... _RestTy>
		inline typename std::enable_if<
			!is_key<typename std::remove_reference<_FirstTy>::type>::value,
			uint64_t
		>::type computeHashMany(uint64_t seed, _FirstTy&& first, _RestTy&&... rest)
		{
			seed = computeHash(seed, std::forward<_FirstTy>(first));
			seed = computeHashMany(seed, std::forward<_RestTy>(rest)...);
			return seed;
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

			uint64_t hash(uint64_t seed, const _Ty& v)
			{
				return computeFastHash(&v, sizeof(_Ty), seed);
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

			uint64_t hash(uint64_t seed, const _Ty& v)
			{
				return v.computeHash(seed);
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

			uint64_t hash(uint64_t seed, const VTy& v)
			{
				seed = computeHash(seed, (uint32_t)v.rows());
				seed = computeHash(seed, (uint32_t)v.cols());
				return computeFastHash(v.data(), sizeof(_Ty) * v.size(), seed);
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

			uint64_t hash(uint64_t seed, const VTy& v)
			{
				seed = computeHash(seed, (uint32_t)v.rows());
				seed = computeHash(seed, (uint32_t)v.cols());
				return computeFastHash(v.data(), sizeof(_Ty) * v.size(), seed);
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

			uint64_t hash(uint64_t seed, const VTy& v)
			{
				seed = computeHash(seed, (uint32_t)v.size());
				return computeFastHash(v.data(), sizeof(_Ty) * v.size(), seed);
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

			uint64_t hash(uint64_t seed, const VTy& v)
			{
				seed = computeHash(seed, (uint32_t)v.size());
				for (auto& e : v)
				{
					seed = computeHash(seed, e);
				}
				return seed;
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

			uint64_t hash(uint64_t seed, const VTy& v)
			{
				seed = computeHash(seed, (uint32_t)v.size());
				return computeFastHash(v.data(), sizeof(_Ty) * v.size(), seed);
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

			uint64_t hash(uint64_t seed, const VTy& v)
			{
				seed = computeHash(seed, (uint32_t)v.size());
				for (auto& e : v)
				{
					seed = computeHash(seed, e);
				}
				return seed;
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

			uint64_t hash(uint64_t seed, const VTy& v)
			{
				seed = computeHash(seed, (uint32_t)v.size());
				return computeFastHash(v.data(), sizeof(_Ty) * v.size(), seed);
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

			uint64_t hash(uint64_t seed, const VTy& v)
			{
				seed = computeHash(seed, v.first);
				seed = computeHash(seed, v.second);
				return seed;
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

			// not support hash
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

			uint64_t hash(uint64_t seed, const VTy& v)
			{
				seed = computeHash(seed, (uint32_t)v.size());
				for (auto& e : v)
				{
					seed = computeHash(seed, e);
				}
				return seed;
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

			uint64_t hash(uint64_t seed, const VTy& v)
			{
				return _Ty::serializerHash(v, seed);
			}
		};

		class BlockStreamBuffer : public std::basic_streambuf<char>
		{
			std::vector<std::unique_ptr<uint8_t[]>> buffers;
			size_t block_size = 0;
		public:
			BlockStreamBuffer(size_t _block_size = 4096);
			~BlockStreamBuffer();

			int overflow(int c) override;

			std::streamsize xsputn(const char* s, std::streamsize n) override;

			size_t totalSize() const;

			template<class Fn>
			void iterateBuffers(Fn fn) const
			{
				for (size_t i = 0; i < buffers.size() - 1; ++i)
				{
					fn(buffers[i].get(), block_size);
				}
				fn(buffers.back().get(), this->pptr() - this->pbase());
			}
		};

		static constexpr uint32_t taggedDataKeyUint = 0x4b545054; // "TPTK"

		struct TaggedDataHeader
		{
			uint32_t key;
			uint32_t version;
			uint64_t totsize;
			uint32_t keysize;
			uint32_t trailing_cnt;
		};

		template<size_t _len, typename _Ty>
		inline void writeTaggedData(std::ostream& ostr, uint32_t version, uint32_t trailing_cnt, const Key<_len>& key, const _Ty& data)
		{
			BlockStreamBuffer buf;
			std::ostream serialized_data(&buf);
			writeMany(serialized_data, key, data);
			const auto key_data_size = buf.totalSize();

			TaggedDataHeader h;
			h.key = taggedDataKeyUint;
			h.version = version;
			h.totsize = key_data_size + 16;
			h.keysize = key.size();
			h.trailing_cnt = trailing_cnt;

			ostr.write((const char*)&h, sizeof(h));
			buf.iterateBuffers([&](const void* data, size_t size)
			{
				ostr.write((const char*)data, size);
			});
		}

		using TaggedDataMap = std::unordered_map<std::string, std::pair<std::streampos, std::streampos>>;

		TaggedDataMap readTaggedDataMap(std::istream& istr, uint32_t version);

		inline void readTaggedMany(std::istream& istr, const TaggedDataMap& data_map, uint32_t version)
		{
			// seek to the end of tagged data list
			istr.seekg(data_map.find("")->second.second);
		}

		template<size_t _len, typename _Ty, typename ... _Rest>
		inline void readTaggedMany(std::istream& istr, const TaggedDataMap& data_map, uint32_t version, const Key<_len>& key, _Ty& data, _Rest&&... rest)
		{
			auto it = data_map.find(key.str());
			if (it != data_map.end())
			{
				istr.seekg(it->second.first);
				readMany(istr, data);
			}
			readTaggedMany(istr, data_map, version, std::forward<_Rest>(rest)...);
		}

		template<typename ... _Rest>
		inline void readTaggedMany(std::istream& istr, uint32_t version, _Rest&&... rest)
		{
			const auto data_map = readTaggedDataMap(istr, version);
			readTaggedMany(istr, data_map, version, std::forward<_Rest>(rest)...);
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

#define DEFINE_HASHER(...) uint64_t computeHash(uint64_t seed) const\
{\
	return tomoto::serializer::computeHashMany(seed, __VA_ARGS__);\
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

#define DEFINE_HASHER_AFTER_BASE(base, ...) uint64_t computeHash(uint64_t seed) const\
{\
	seed = base::computeHash(seed);\
	return tomoto::serializer::computeHashMany(seed, __VA_ARGS__);\
}\

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

#define DEFINE_OUT_SERIALIZER_AFTER_BASE_WITH_VERSION(cls, base, v, ...) template<TermWeight _tw> void cls<_tw>::serializerRead(tomoto::serializer::version_holder<v> _v, std::istream& istr)\
{\
	base::serializerRead(_v, istr);\
	tomoto::serializer::readMany(istr, __VA_ARGS__);\
}\
template<TermWeight _tw> void cls<_tw>::serializerWrite(tomoto::serializer::version_holder<v> _v, std::ostream& ostr) const\
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

#define DEFINE_OUT_SERIALIZER_BASE_WITH_VERSION(cls, base, v) template<TermWeight _tw> void cls<_tw>::serializerRead(tomoto::serializer::version_holder<v> _v, std::istream& istr)\
{\
	base::serializerRead(_v, istr);\
}\
template<TermWeight _tw> void cls<_tw>::serializerWrite(tomoto::serializer::version_holder<v> _v, std::ostream& ostr) const\
{\
	base::serializerWrite(_v, ostr);\
}


#define DECLARE_SERIALIZER_WITH_VERSION(v) void serializerRead(tomoto::serializer::version_holder<v> _v, std::istream& istr);\
void serializerWrite(tomoto::serializer::version_holder<v> _v, std::ostream& ostr) const;

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

#define DEFINE_OUT_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(cls, base, v, t,...) template<TermWeight _tw> void cls<_tw>::serializerRead(tomoto::serializer::version_holder<v> _v, std::istream& istr)\
{\
	base::serializerRead(_v, istr);\
	tomoto::serializer::readTaggedMany(istr, t, _TO_KEY_VALUE(__VA_ARGS__));\
}\
template<TermWeight _tw> void cls<_tw>::serializerWrite(tomoto::serializer::version_holder<v> _v, std::ostream& ostr) const\
{\
	base::serializerWrite(_v, ostr);\
	tomoto::serializer::writeTaggedMany(ostr, t, _TO_KEY_VALUE(__VA_ARGS__));\
}
