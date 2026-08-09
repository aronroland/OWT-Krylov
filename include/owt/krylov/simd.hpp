#pragma once

#include <concepts>
#include <cstddef>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace owt::krylov::simd {

template<std::floating_point T>
inline void fused_multiply_add(T* out, const T* coefficient,
                               const T* input, std::size_t size) noexcept
{
    std::size_t i = 0;
#if defined(__AVX512F__)
    if constexpr (std::same_as<T, double>) {
        for (; i + 8 <= size; i += 8) {
            const __m512d out_values = _mm512_loadu_pd(out + i);
            const __m512d coefficients = _mm512_loadu_pd(coefficient + i);
            const __m512d input_values = _mm512_loadu_pd(input + i);
#if defined(__FMA__)
            _mm512_storeu_pd(out + i,
                             _mm512_fmadd_pd(coefficients, input_values, out_values));
#else
            _mm512_storeu_pd(out + i,
                             _mm512_add_pd(out_values,
                                           _mm512_mul_pd(coefficients, input_values)));
#endif
        }
    } else if constexpr (std::same_as<T, float>) {
        for (; i + 16 <= size; i += 16) {
            const __m512 out_values = _mm512_loadu_ps(out + i);
            const __m512 coefficients = _mm512_loadu_ps(coefficient + i);
            const __m512 input_values = _mm512_loadu_ps(input + i);
#if defined(__FMA__)
            _mm512_storeu_ps(out + i,
                             _mm512_fmadd_ps(coefficients, input_values, out_values));
#else
            _mm512_storeu_ps(out + i,
                             _mm512_add_ps(out_values,
                                           _mm512_mul_ps(coefficients, input_values)));
#endif
        }
    }
#elif defined(__AVX2__)
    if constexpr (std::same_as<T, double>) {
        for (; i + 4 <= size; i += 4) {
            const __m256d out_values = _mm256_loadu_pd(out + i);
            const __m256d coefficients = _mm256_loadu_pd(coefficient + i);
            const __m256d input_values = _mm256_loadu_pd(input + i);
#if defined(__FMA__)
            _mm256_storeu_pd(out + i,
                             _mm256_fmadd_pd(coefficients, input_values,
                                             out_values));
#else
            _mm256_storeu_pd(out + i,
                             _mm256_add_pd(out_values,
                                           _mm256_mul_pd(coefficients,
                                                         input_values)));
#endif
        }
    } else if constexpr (std::same_as<T, float>) {
        for (; i + 8 <= size; i += 8) {
            const __m256 out_values = _mm256_loadu_ps(out + i);
            const __m256 coefficients = _mm256_loadu_ps(coefficient + i);
            const __m256 input_values = _mm256_loadu_ps(input + i);
#if defined(__FMA__)
            _mm256_storeu_ps(out + i,
                             _mm256_fmadd_ps(coefficients, input_values,
                                             out_values));
#else
            _mm256_storeu_ps(out + i,
                             _mm256_add_ps(out_values,
                                           _mm256_mul_ps(coefficients,
                                                         input_values)));
#endif
        }
    }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    if constexpr (std::same_as<T, double>) {
#if defined(__aarch64__)
        for (; i + 2 <= size; i += 2) {
            const float64x2_t out_values = vld1q_f64(out + i);
            const float64x2_t coefficients = vld1q_f64(coefficient + i);
            const float64x2_t input_values = vld1q_f64(input + i);
            vst1q_f64(out + i,
                      vmlaq_f64(out_values, coefficients, input_values));
        }
#endif
    } else if constexpr (std::same_as<T, float>) {
        for (; i + 4 <= size; i += 4) {
            const float32x4_t out_values = vld1q_f32(out + i);
            const float32x4_t coefficients = vld1q_f32(coefficient + i);
            const float32x4_t input_values = vld1q_f32(input + i);
            vst1q_f32(out + i,
                      vmlaq_f32(out_values, coefficients, input_values));
        }
    }
#endif
    for (; i < size; ++i) {
        out[i] += coefficient[i] * input[i];
    }
}

template<std::floating_point T>
inline void axpy(T* out, T alpha, const T* input, std::size_t size) noexcept
{
    std::size_t i = 0;
#if defined(__AVX512F__)
    if constexpr (std::same_as<T, double>) {
        const __m512d alpha_values = _mm512_set1_pd(alpha);
        for (; i + 8 <= size; i += 8) {
            const __m512d out_values = _mm512_loadu_pd(out + i);
            const __m512d input_values = _mm512_loadu_pd(input + i);
#if defined(__FMA__)
            _mm512_storeu_pd(out + i,
                             _mm512_fmadd_pd(alpha_values, input_values, out_values));
#else
            _mm512_storeu_pd(out + i,
                             _mm512_add_pd(out_values,
                                           _mm512_mul_pd(alpha_values, input_values)));
#endif
        }
    } else if constexpr (std::same_as<T, float>) {
        const __m512 alpha_values = _mm512_set1_ps(alpha);
        for (; i + 16 <= size; i += 16) {
            const __m512 out_values = _mm512_loadu_ps(out + i);
            const __m512 input_values = _mm512_loadu_ps(input + i);
#if defined(__FMA__)
            _mm512_storeu_ps(out + i,
                             _mm512_fmadd_ps(alpha_values, input_values, out_values));
#else
            _mm512_storeu_ps(out + i,
                             _mm512_add_ps(out_values,
                                           _mm512_mul_ps(alpha_values, input_values)));
#endif
        }
    }
#elif defined(__AVX2__)
    if constexpr (std::same_as<T, double>) {
        const __m256d alpha_values = _mm256_set1_pd(alpha);
        for (; i + 4 <= size; i += 4) {
            const __m256d out_values = _mm256_loadu_pd(out + i);
            const __m256d input_values = _mm256_loadu_pd(input + i);
#if defined(__FMA__)
            _mm256_storeu_pd(out + i,
                             _mm256_fmadd_pd(alpha_values, input_values,
                                             out_values));
#else
            _mm256_storeu_pd(out + i,
                             _mm256_add_pd(out_values,
                                           _mm256_mul_pd(alpha_values,
                                                         input_values)));
#endif
        }
    } else if constexpr (std::same_as<T, float>) {
        const __m256 alpha_values = _mm256_set1_ps(alpha);
        for (; i + 8 <= size; i += 8) {
            const __m256 out_values = _mm256_loadu_ps(out + i);
            const __m256 input_values = _mm256_loadu_ps(input + i);
#if defined(__FMA__)
            _mm256_storeu_ps(out + i,
                             _mm256_fmadd_ps(alpha_values, input_values,
                                             out_values));
#else
            _mm256_storeu_ps(out + i,
                             _mm256_add_ps(out_values,
                                           _mm256_mul_ps(alpha_values,
                                                         input_values)));
#endif
        }
    }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    if constexpr (std::same_as<T, double>) {
#if defined(__aarch64__)
        const float64x2_t alpha_values = vdupq_n_f64(alpha);
        for (; i + 2 <= size; i += 2) {
            const float64x2_t out_values = vld1q_f64(out + i);
            const float64x2_t input_values = vld1q_f64(input + i);
            vst1q_f64(out + i,
                      vmlaq_f64(out_values, alpha_values, input_values));
        }
#endif
    } else if constexpr (std::same_as<T, float>) {
        const float32x4_t alpha_values = vdupq_n_f32(alpha);
        for (; i + 4 <= size; i += 4) {
            const float32x4_t out_values = vld1q_f32(out + i);
            const float32x4_t input_values = vld1q_f32(input + i);
            vst1q_f32(out + i,
                      vmlaq_f32(out_values, alpha_values, input_values));
        }
    }
#endif
    for (; i < size; ++i) {
        out[i] += alpha * input[i];
    }
}

} // namespace owt::krylov::simd
