//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/algorithm>

// template<class T, StrictWeakOrder<auto, T> Compare>
//   requires !SameType<T, Compare> && CopyConstructible<Compare>
//   pair<const T&, const T&>
//   minmax(const T& a, const T& b, Compare comp);

#include <cuda/std/algorithm>
#include <cuda/std/functional>
#include <cuda/std/cassert>

#include "test_macros.h"

template <class T, class C>
void
__host__ __device__ test(const T& a, const T& b, C c, const T& x, const T& y)
{
    cuda::std::pair<const T&, const T&> p = cuda::std::minmax(a, b, c);
    assert(&p.first == &x);
    assert(&p.second == &y);
}


int main(int, char**)
{
    {
    int x = 0;
    int y = 0;
    test(x, y, cuda::std::greater<int>(), x, y);
    test(y, x, cuda::std::greater<int>(), y, x);
    }
    {
    int x = 0;
    int y = 1;
    test(x, y, cuda::std::greater<int>(), y, x);
    test(y, x, cuda::std::greater<int>(), y, x);
    }
    {
    int x = 1;
    int y = 0;
    test(x, y, cuda::std::greater<int>(), x, y);
    test(y, x, cuda::std::greater<int>(), x, y);
    }
#if TEST_STD_VER >= 14
    {
//  Note that you can't take a reference to a local var, since
//  its address is not a compile-time constant.
    constexpr static int x = 1;
    constexpr static int y = 0;
    constexpr auto p1 = cuda::std::minmax(x, y, cuda::std::greater<>());
    static_assert(p1.first  == x, "");
    static_assert(p1.second == y, "");
    constexpr auto p2 = cuda::std::minmax(y, x, cuda::std::greater<>());
    static_assert(p2.first  == x, "");
    static_assert(p2.second == y, "");
    }
#endif

  return 0;
}
