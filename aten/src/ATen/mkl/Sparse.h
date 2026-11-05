#pragma once

#include <ATen/Config.h>

#if AT_MKL_ENABLED()
#define AT_USE_MKL_SPARSE() 1
#else
#define AT_USE_MKL_SPARSE() 0
#endif
#if AT_HML_ENABLED()
#define AT_USE_HSPARSE() 1
#else
#define AT_USE_HSPARSE() 0
#endif
