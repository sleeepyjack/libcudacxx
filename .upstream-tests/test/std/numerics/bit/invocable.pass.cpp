//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03

// Test the ability of each <bit> function to be invoked

#include <bits/types/time_t.h>
#include <cuda/std/cstdint>
#include <cuda/std/bit>

#if (defined(__cplusplus) && __cplusplus >= 201703L) || \
    (defined(_MSC_VER) && _MSVC_LANG >= 201703L)
#  define CPP17_PERFORM_INVOCABLE_TEST
#endif

class A{};
enum       E1 : unsigned char { rEd };
enum class E2 : unsigned char { red };

// rotr
struct i_rotr {
    template <typename T>
    __host__ __device__ constexpr auto operator()(T x) const -> decltype(cuda::std::rotr(x, 1U));
};

// rotl
struct i_rotl {
    template <typename T>
    __host__ __device__ constexpr auto operator()(T x) const -> decltype(cuda::std::rotl(x, 1U));
};

// popcount
struct i_popcount {
    template <typename T>
    __host__ __device__ constexpr auto operator()(T x) const -> decltype(cuda::std::popcount(x));
};

// countr_zero
struct i_countr_zero {
    template <typename T>
    __host__ __device__ constexpr auto operator()(T x) const -> decltype(cuda::std::countr_zero(x));
};

// countr_one
struct i_countr_one {
    template <typename T>
    __host__ __device__ constexpr auto operator()(T x) const -> decltype(cuda::std::countr_one(x));
};

// countl_zero
struct i_countl_zero {
    template <typename T>
    __host__ __device__ constexpr auto operator()(T x) const -> decltype(cuda::std::countl_zero(x));
};

// countl_one
struct i_countl_one {
    template <typename T>
    __host__ __device__ constexpr auto operator()(T x) const -> decltype(cuda::std::countl_one(x));
};

// log2p1
struct i_log2p1 {
    template <typename T>
    __host__ __device__ constexpr auto operator()(T x) const -> decltype(cuda::std::log2p1(x));
};

// ispow2
struct i_ispow2 {
    template <typename T>
    __host__ __device__ constexpr auto operator()(T x) const -> decltype(cuda::std::ispow2(x));
};

// floor2
struct i_floor2 {
    template <typename T>
    __host__ __device__ constexpr auto operator()(T x) const -> decltype(cuda::std::floor2(x));
};

// ceil2
struct i_ceil2 {
    template <typename T>
    __host__ __device__ constexpr auto operator()(T x) const -> decltype(cuda::std::ceil2(x));
};

template <typename L>
__host__ __device__ void test_invocable() {
#if defined(CPP17_PERFORM_INVOCABLE_TEST)
    static_assert( cuda::std::is_invocable_v<L, unsigned char>, "");
    static_assert( cuda::std::is_invocable_v<L, unsigned int>, "");
    static_assert( cuda::std::is_invocable_v<L, unsigned long>, "");
    static_assert( cuda::std::is_invocable_v<L, unsigned long long>, "");

    static_assert( cuda::std::is_invocable_v<L, uint8_t>, "");
    static_assert( cuda::std::is_invocable_v<L, uint16_t>, "");
    static_assert( cuda::std::is_invocable_v<L, uint32_t>, "");
    static_assert( cuda::std::is_invocable_v<L, uint64_t>, "");
    static_assert( cuda::std::is_invocable_v<L, size_t>, "");

    static_assert( cuda::std::is_invocable_v<L, uintmax_t>, "");
    static_assert( cuda::std::is_invocable_v<L, uintptr_t>, "");


    static_assert(!cuda::std::is_invocable_v<L, int>, "");
    static_assert(!cuda::std::is_invocable_v<L, signed int>, "");
    static_assert(!cuda::std::is_invocable_v<L, long>, "");
    static_assert(!cuda::std::is_invocable_v<L, long long>, "");

    static_assert(!cuda::std::is_invocable_v<L, int8_t>, "");
    static_assert(!cuda::std::is_invocable_v<L, int16_t>, "");
    static_assert(!cuda::std::is_invocable_v<L, int32_t>, "");
    static_assert(!cuda::std::is_invocable_v<L, int64_t>, "");
    static_assert(!cuda::std::is_invocable_v<L, ptrdiff_t>, "");

    static_assert(!cuda::std::is_invocable_v<L, bool>, "");
    static_assert(!cuda::std::is_invocable_v<L, signed char>, "");
    static_assert(!cuda::std::is_invocable_v<L, char16_t>, "");
    static_assert(!cuda::std::is_invocable_v<L, char32_t>, "");

#ifndef _LIBCUDACXX_HAS_NO_INT128
    static_assert( cuda::std::is_invocable_v<L, __uint128_t>, "");
    static_assert(!cuda::std::is_invocable_v<L,  __int128_t>, "");
#endif

    static_assert(!cuda::std::is_invocable_v<L, A, unsigned>, "");
    static_assert(!cuda::std::is_invocable_v<L, E1, unsigned>, "");
    static_assert(!cuda::std::is_invocable_v<L, E2, unsigned>, "");
#endif // defined(CPP17_PERFORM_INVOCABLE_TEST)
}

int main(int, char **) {
  {
    test_invocable<i_rotr>();
    test_invocable<i_rotl>();
    test_invocable<i_popcount>();
    test_invocable<i_countr_zero>();
    test_invocable<i_countr_one>();
    test_invocable<i_countl_zero>();
    test_invocable<i_countl_one>();
    test_invocable<i_log2p1>();
    test_invocable<i_ispow2>();
    test_invocable<i_floor2>();
    test_invocable<i_ceil2>();
  }
  return 0;
}