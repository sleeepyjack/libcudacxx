//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/algorithm>

// template<ForwardIterator Iter>
//   max_element(Iter first, Iter last);

#include <cuda/std/algorithm>
#include <cuda/std/cassert>

#include "test_iterators.h"

int main(int, char**) {
  int arr[] = {1, 2, 3};
  const int *b = cuda::std::begin(arr), *e = cuda::std::end(arr);
  typedef input_iterator<const int*> Iter;
  {
    // expected-error@algorithm:* {{"cuda::std::min_element requires a ForwardIterator"}}
    cuda::std::min_element(Iter(b), Iter(e));
  }
  {
    // expected-error@algorithm:* {{"cuda::std::max_element requires a ForwardIterator"}}
    cuda::std::max_element(Iter(b), Iter(e));
  }
  {
    // expected-error@algorithm:* {{"cuda::std::minmax_element requires a ForwardIterator"}}
    cuda::std::minmax_element(Iter(b), Iter(e));
  }


  return 0;
}
