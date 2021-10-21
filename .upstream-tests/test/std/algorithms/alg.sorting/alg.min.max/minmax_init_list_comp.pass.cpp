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
//   pair<T, T>
//   minmax(initializer_list<T> t, Compare comp);
//
//  Complexity: At most (3/2) * t.size() applications of the corresponding predicate.

#include <cuda/std/algorithm>
#include <cuda/std/functional>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "counting_predicates.h"

bool all_equal(int, int) { return false; } // everything is equal

void test_all_equal(cuda::std::initializer_list<int> il)
{
    binary_counting_predicate<bool(*)(int, int), int, int> pred (all_equal);
    cuda::std::pair<int, int> p = cuda::std::minmax(il, cuda::std::ref(pred));
    const int *ptr = il.end();
    assert(p.first == *il.begin());
    assert(p.second == *--ptr);
    assert(pred.count() <= ((3 * il.size()) / 2));
}

int main(int, char**)
{
    assert((cuda::std::minmax({1, 2, 3}, cuda::std::greater<int>()) == cuda::std::pair<int, int>(3, 1)));
    assert((cuda::std::minmax({1, 3, 2}, cuda::std::greater<int>()) == cuda::std::pair<int, int>(3, 1)));
    assert((cuda::std::minmax({2, 1, 3}, cuda::std::greater<int>()) == cuda::std::pair<int, int>(3, 1)));
    assert((cuda::std::minmax({2, 3, 1}, cuda::std::greater<int>()) == cuda::std::pair<int, int>(3, 1)));
    assert((cuda::std::minmax({3, 1, 2}, cuda::std::greater<int>()) == cuda::std::pair<int, int>(3, 1)));
    assert((cuda::std::minmax({3, 2, 1}, cuda::std::greater<int>()) == cuda::std::pair<int, int>(3, 1)));
    assert((cuda::std::minmax({1, 2, 3}, all_equal          ) == cuda::std::pair<int, int>(1, 3)));

    binary_counting_predicate<cuda::std::greater<int>, int, int> pred ((cuda::std::greater<int>()));
    assert((cuda::std::minmax({1, 2, 2, 3, 3, 3, 5, 5, 5, 5, 5, 3}, cuda::std::ref(pred)) == cuda::std::pair<int, int>(5, 1)));
    assert(pred.count() <= 18); // size == 12

    test_all_equal({0});
    test_all_equal({0,1});
    test_all_equal({0,1,2});
    test_all_equal({0,1,2,3});
    test_all_equal({0,1,2,3,4});
    test_all_equal({0,1,2,3,4,5});
    test_all_equal({0,1,2,3,4,5,6});
    test_all_equal({0,1,2,3,4,5,6,7});
    test_all_equal({0,1,2,3,4,5,6,7,8});
    test_all_equal({0,1,2,3,4,5,6,7,8,9});
    test_all_equal({0,1,2,3,4,5,6,7,8,9,10});
    test_all_equal({0,1,2,3,4,5,6,7,8,9,10,11});

#if TEST_STD_VER >= 14
    {
    static_assert((cuda::std::minmax({1, 2, 3}, cuda::std::greater<int>()) == cuda::std::pair<int, int>(3, 1)), "");
    static_assert((cuda::std::minmax({1, 3, 2}, cuda::std::greater<int>()) == cuda::std::pair<int, int>(3, 1)), "");
    static_assert((cuda::std::minmax({2, 1, 3}, cuda::std::greater<int>()) == cuda::std::pair<int, int>(3, 1)), "");
    static_assert((cuda::std::minmax({2, 3, 1}, cuda::std::greater<int>()) == cuda::std::pair<int, int>(3, 1)), "");
    static_assert((cuda::std::minmax({3, 1, 2}, cuda::std::greater<int>()) == cuda::std::pair<int, int>(3, 1)), "");
    static_assert((cuda::std::minmax({3, 2, 1}, cuda::std::greater<int>()) == cuda::std::pair<int, int>(3, 1)), "");
    }
#endif

  return 0;
}
