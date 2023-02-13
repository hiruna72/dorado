#pragma once

#include <torch/torch.h>

#include <vector>

namespace dorado::utils {

inline void load_state_dict(torch::nn::Module& module,
                            const std::vector<torch::Tensor>& weights,
                            const std::vector<torch::Tensor>& buffers = {}) {
    assert(weights.size() == module.parameters().size());
    for (size_t idx = 0; idx < weights.size(); idx++) {
        module.parameters()[idx].data() = weights[idx].data();
    }

    assert(buffers.size() == module.buffers().size());
    for (size_t idx = 0; idx < buffers.size(); idx++) {
        module.buffers()[idx].data() = buffers[idx].data();
    }
}

}  // namespace dorado::utils
