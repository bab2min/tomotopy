#include "serializer.hpp"

namespace tomoto
{
	namespace serializer
	{
		membuf::membuf(bool read, bool write, char* base, std::ptrdiff_t n)
		{
			if (read)
			{
				this->setg(base, base, base + n);
			}

			if (write)
			{
				this->setp(base, base + n);
			}
		}

		membuf::~membuf() = default;

		std::streampos membuf::seekpos(pos_type sp, std::ios_base::openmode which) 
		{
			return seekoff(sp - pos_type(off_type(0)), std::ios_base::beg, which);
		}

		std::streampos membuf::seekoff(off_type off,
			std::ios_base::seekdir dir,
			std::ios_base::openmode which
		) 
		{
			if (which & std::ios_base::in)
			{
				if (dir == std::ios_base::cur)
					gbump(off);
				else if (dir == std::ios_base::end)
					setg(eback(), egptr() + off, egptr());
				else if (dir == std::ios_base::beg)
					setg(eback(), eback() + off, egptr());
			}
			if (which & std::ios_base::out)
			{
				if (dir == std::ios_base::cur)
					pbump(off);
				else if (dir == std::ios_base::end)
					setp(epptr() + off, epptr());
				else if (dir == std::ios_base::beg)
					setp(pbase() + off, epptr());

				if (!(which & std::ios_base::in))
				{
					return pptr() - pbase();
				}
			}
			return gptr() - eback();
		}

		imstream::imstream(const char* base, std::ptrdiff_t n)
			: std::istream(&buf), buf(true, false, (char*)base, n)
		{
		}

		imstream::~imstream() = default;

		omstream::omstream(char* base, std::ptrdiff_t n)
			: std::ostream(&buf), buf(false, true, (char*)base, n)
		{
		}

		omstream::~omstream() = default;


		template<size_t block_size>
		BlockStreamBuffer<block_size>::BlockStreamBuffer()
		{
			buffers.emplace_back();
			this->setp((char*)buffers.back().data(), (char*)buffers.back().data() + buffers.back().size());
		}

		template<size_t block_size>
		BlockStreamBuffer<block_size>::~BlockStreamBuffer() = default;

		template<size_t block_size>
		int BlockStreamBuffer<block_size>::overflow(int c)
		{
			if (this->pptr() == this->epptr())
			{
				buffers.emplace_back();
				this->setp((char*)buffers.back().data(), (char*)buffers.back().data() + buffers.back().size());
			}
			else
			{
				*(this->pptr()) = c;
				this->pbump(1);
			}
			return c;
		}

		template<size_t block_size>
		std::streamsize BlockStreamBuffer<block_size>::xsputn(const char* s, std::streamsize n)
		{
			auto rest = n;
			auto buf_remain = this->epptr() - this->pptr();
			while (rest > buf_remain)
			{
				std::copy(s, s + buf_remain, this->pptr());
				this->pbump(buf_remain);
				buffers.emplace_back();
				this->setp((char*)buffers.back().data(), (char*)buffers.back().data() + buffers.back().size());
				rest -= buf_remain;
				s += buf_remain;
				buf_remain = block_size;
			}
			std::copy(s, s + rest, this->pptr());
			this->pbump(rest);
			return n;
		}

		template<size_t block_size>
		size_t BlockStreamBuffer<block_size>::totalSize() const
		{
			return (buffers.size() - 1) * block_size + (this->pptr() - this->pbase());
		}

		template class BlockStreamBuffer<4096>;

		TaggedDataMap readTaggedDataMap(std::istream& istr, uint32_t version)
		{
			std::unordered_map<std::string, std::pair<std::streampos, std::streampos>> ret;
			TaggedDataHeader h;
			do
			{
				istr.read((char*)&h, sizeof(h));
				if (h.key != taggedDataKeyUint)
				{
					throw UnfitException("tagged data key is not found");
				}
				const std::streampos totsize_pos = istr.tellg() - (std::streamoff)16;
				std::array<char, 256> key;
				istr.read(key.data(), h.keysize);
				const std::streampos start_pos = istr.tellg();
				const std::streampos end_pos = totsize_pos + (std::streamoff)h.totsize;
				ret.emplace(std::string{ key.data(), h.keysize }, std::make_pair(start_pos, end_pos));
				ret[""] = std::make_pair(start_pos, end_pos);
				istr.seekg(end_pos);
			} while (h.trailing_cnt);
			return ret;
		}
	}
}
