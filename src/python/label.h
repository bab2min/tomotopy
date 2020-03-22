#pragma once

#ifdef _DEBUG
//#undef _DEBUG
#define DEBUG_LOG(t) do{ cerr << t << endl; }while(0)
#include "PyUtils.h"
//#define _DEBUG
#else 
#define DEBUG_LOG(t)
#include "PyUtils.h"
#endif

PyObject* makeLabelModule();