// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#pragma once

#include <bits/types/time_t.h>
#if (defined(__cplusplus) && __cplusplus >= 201703L) || \
    (defined(_MSC_VER) && _MSVC_LANG >= 201703L)
#  define CPP17_PERFORM_INVOCABLE_TEST
#endif

template <typename T>
__host__ __device__
constexpr bool unused(T &&) {return true;}