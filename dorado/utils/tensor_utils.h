#pragma once

#include <torch/torch.h>

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace dorado::utils {

// Serialise Torch tensor to disk.
void serialise_tensor(torch::Tensor t, const std::string& path);
// Load serialised tensor from disk.
std::vector<torch::Tensor> load_tensors(const std::filesystem::path& dir,
                                        const std::vector<std::string>& tensors);

// Computes the q-th quantiles of each row of the input tensor `t`
// using a partial sort as opposed a full sort per torch::quantiles
// Only `interpolation='lower'` is currently implemented.
torch::Tensor quantile(const torch::Tensor t, const torch::Tensor q);

// Computes the q-th quantiles of each row of the input tensor `t`
// using a counting sort which is extremely fast for low range integers.
// Only `interpolation='lower'` is currently implemented.
torch::Tensor quantile_counting(const torch::Tensor t, const torch::Tensor q);

// Converts count float elements pointed to by src to half precision, with
// the result pointed to by dest.
void convert_f32_to_f16(c10::Half* dest, const float* src, std::size_t count);

// Copies count elements from src_offset elements into src to
// dest_elements into dst.  The tensors must be contiguous.
void copy_tensor_elems(torch::Tensor& dest_tensor,
                       size_t dest_offset,
                       const torch::Tensor& src_tensor,
                       std::size_t src_offset,
                       std::size_t count);

}  // namespace dorado::utils
