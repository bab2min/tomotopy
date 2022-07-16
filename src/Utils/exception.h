#pragma once
#include <exception>
#include "text.hpp"
namespace tomoto
{
	namespace exc
	{
		class TrainingError : public std::runtime_error
		{
		public:
			using std::runtime_error::runtime_error;
		};

		class Unimplemented : public std::runtime_error
		{
		public:
			using std::runtime_error::runtime_error;
		};

		class InvalidArgument : public std::invalid_argument
		{
		public:
			using std::invalid_argument::invalid_argument;
		};

		class EmptyWordArgument : public InvalidArgument
		{
		public:
			using InvalidArgument::InvalidArgument;
		};
	}
}

#define THROW_ERROR_WITH_INFO(exec, msg) do {throw exec(tomoto::text::format("%s (%d): ", __FILE__, __LINE__) + msg); } while(0)
