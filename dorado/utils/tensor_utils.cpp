#include "tensor_utils.h"

#include "simd.h"

#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/torch.h>

#include <cstddef>
#include <cstring>
#include <fstream>
#include <vector>

namespace {

#if ENABLE_AVX2_IMPL
__attribute__((target("default")))
#endif
void convert_f32_to_f16_impl(c10::Half* const dest, const float* const src, std::size_t count) {
    // TODO -- handle large counts properly.
    assert(count <= std::numeric_limits<int>::max());
    auto src_tensor_f32 = torch::from_blob(const_cast<float*>(src), {static_cast<int>(count)});
    auto src_tensor_f16 = src_tensor_f32.to(torch::kFloat16);
    std::memcpy(dest, src_tensor_f16.data_ptr(), count * sizeof(c10::Half));
}

#if ENABLE_AVX2_IMPL
// We have to specify f16c to have _mm256_cvtps_ph available, as strictly speaking it's a separate
// feature from AVX2.  All relevant CPUs have it.
__attribute__((target("avx2,f16c"))) void convert_f32_to_f16_impl(c10::Half* const dest,
                                                                  const float* const src,
                                                                  std::size_t count) {
    if (!count)
        return;

    // Unroll to AVX register size: 8 floats.
    static constexpr size_t kUnroll = 8;

    // Matches torch behaviour.
    const int kRoundNearestEven = 0;

    // Main vectorised loop: 8 floats per iteration.
    const auto* src_ptr = src;
    auto* dest_ptr = dest;
    for (size_t chunk_i = 0; chunk_i < count / kUnroll; ++chunk_i) {
        const __m256 elems_f32 = _mm256_loadu_ps(src_ptr);
        const __m128i elems_f16 = _mm256_cvtps_ph(elems_f32, kRoundNearestEven);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dest_ptr), elems_f16);
        src_ptr += kUnroll;
        dest_ptr += kUnroll;
    }

    // Loop for final 0-7 floats.
    // TODO -- probably nicer to use masked loads/stores.
    const size_t remaining_count = count % kUnroll;
    for (size_t i = 0; i < remaining_count; ++i) {
        const __m256 elem_f32 = _mm256_broadcast_ss(src_ptr);
        const __m128i elem_f16 = _mm256_cvtps_ph(elem_f32, kRoundNearestEven);
        *(reinterpret_cast<std::int16_t*>(dest_ptr)) =
                static_cast<std::int16_t>(_mm_extract_epi16(elem_f16, 0));
        ++src_ptr;
        ++dest_ptr;
    }
}
#endif

}  // namespace

namespace dorado::utils {

void serialise_tensor(torch::Tensor t, const std::string& path) {
    auto bytes = torch::jit::pickle_save(t);
    std::ofstream fout(path);
    fout.write(bytes.data(), bytes.size());
    fout.close();
}

std::vector<torch::Tensor> load_tensors(const std::filesystem::path& dir,
                                        const std::vector<std::string>& tensors) {
    auto weights = std::vector<torch::Tensor>();
    for (auto tensor : tensors) {
        auto path = dir / tensor;
        torch::load(weights, path.string());
    }

    return weights;
}

torch::Tensor quantile(const torch::Tensor t, const torch::Tensor q) {
    assert(q.dtype() == torch::kF32);

    auto tmp = t.clone();
    auto [qval, qidx] = q.sort();
    auto res = torch::empty_like(q);

    auto start = tmp.data_ptr<float>();
    auto end = tmp.data_ptr<float>() + tmp.size(0);

    for (int i = 0; i < q.size(0); i++) {
        auto m = tmp.data_ptr<float>() +
                 static_cast<size_t>((tmp.size(0) - 1) * qval[i].item<float>());
        std::nth_element(start, m, end);
        res[qidx[i]] = *m;
        start = m;
    }

    return res;
}

torch::Tensor quantile_counting(const torch::Tensor t, const torch::Tensor q) {
    assert(q.dtype() == torch::kF32);

    auto p = t.data_ptr<int16_t>();
    auto range_min = t.min().item<int16_t>();
    auto range_max = t.max().item<int16_t>();

    int size = t.size(0);

    std::vector<int> counts(range_max - range_min + 1, 0);
    for (int i = 0; i < size; ++i) {
        counts[p[i] - range_min]++;
    }
    std::partial_sum(counts.begin(), counts.end(), counts.begin());

    auto res = torch::empty_like(q);

    for (size_t idx = 0; idx < q.numel(); idx++) {
        int threshold = q[idx].item<float>() * (size - 1);
        for (int i = 0; i < counts.size(); ++i) {
            if (counts[i] > threshold) {
                res[idx] = i + range_min;
                break;
            }
        }
    }

    return res;
}

// Multiversioned function dispatch doesn't work across the dorado_lib linking
// boundary.  Without this wrapper, AVX machines still only execute the default
// version.
void convert_f32_to_f16(c10::Half* const dest, const float* const src, std::size_t count) {
    return convert_f32_to_f16_impl(dest, src, count);
}

void copy_tensor_elems(torch::Tensor& dest_tensor,
                       std::size_t dest_offset,
                       const torch::Tensor& src_tensor,
                       std::size_t src_offset,
                       std::size_t count) {
    assert(dest_tensor.is_contiguous());
    assert(src_tensor.is_contiguous());
    assert(dest_offset + count <= dest_tensor.numel());
    assert(src_offset + count <= src_tensor.numel());

    if (dest_tensor.dtype() == src_tensor.dtype()) {
        // No conversion.
        auto* const dest_ptr = reinterpret_cast<std::byte*>(dest_tensor.data_ptr());
        const auto* const src_ptr = reinterpret_cast<std::byte*>(src_tensor.data_ptr());
        const size_t elem_size = dest_tensor.element_size();
        std::memcpy(&dest_ptr[dest_offset * elem_size], &src_ptr[src_offset * elem_size],
                    count * elem_size);
    } else if (dest_tensor.dtype() == torch::kFloat16 && src_tensor.dtype() == torch::kFloat32) {
        // float32 -> float16 conversion.
        auto* const dest_ptr = dest_tensor.data_ptr<c10::Half>();
        const auto* const src_ptr = src_tensor.data_ptr<float>();
        convert_f32_to_f16_impl(&dest_ptr[dest_offset], &src_ptr[src_offset], count);
    } else {
        // Slow fallback path for other conversions.
        using torch::indexing::Slice;
        dest_tensor.flatten().index_put_(
                {Slice(dest_offset, dest_offset + count)},
                src_tensor.flatten().index({Slice(src_offset, src_offset + count)}));
    }
}

}  // namespace dorado::utils
