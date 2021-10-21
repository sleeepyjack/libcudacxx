//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <cuda/std/algorithm>

// template<class T, class Compare>
//   T
//   min(initializer_list<T> t, Compare comp);

#include <cuda/std/algorithm>
#include <cuda/std/functional>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    int i = cuda::std::min({2, 3, 1}, cuda::std::greater<int>());
    assert(i == 3);
    i = cuda::std::min({2, 1, 3}, cuda::std::greater<int>());
    assert(i == 3);
    i = cuda::std::min({3, 1, 2}, cuda::std::greater<int>());
    assert(i == 3);
    i = cuda::std::min({3, 2, 1}, cuda::std::greater<int>());
    assert(i == 3);
    i = cuda::std::min({1, 2, 3}, cuda::std::greater<int>());
    assert(i == 3);
    i = cuda::std::min({1, 3, 2}, cuda::std::greater<int>());
    assert(i == 3);
#if TEST_STD_VER >= 14
    {
    static_assert(cuda::std::min({1, 3, 2}, cuda::std::greater<int>()) == 3, "");
    static_assert(cuda::std::min({2, 1, 3}, cuda::std::greater<int>()) == 3, "");
    static_assert(cuda::std::min({3, 2, 1}, cuda::std::greater<int>()) == 3, "");
    }
#endif

  return 0;
}
