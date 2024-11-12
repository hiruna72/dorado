#pragma once

#include <string>

namespace dorado::utils {

struct DefaultParameters {
    int batchsize{0};
    int chunksize{10000};
    int overlap{500};
    int num_runners{2};
#ifdef DORADO_TX2
    int remora_batchsize{128};
#else
    int remora_batchsize{1024};
#endif
    int remora_threads{4};
    int mod_base_runners_per_caller{2};
    float methylation_threshold{0.05f};

    // Minimum length for a sequence to be outputted.
    size_t min_sequence_length{5};
    int32_t slow5_threads{8};
    int64_t slow5_batchsize{4000};
};

static const DefaultParameters default_parameters{};

struct ThreadAllocations {
    int writer_threads{0};
    int read_converter_threads{0};
    int read_filter_threads{0};
    int remora_threads{0};
    int scaler_node_threads{0};
    int splitter_node_threads{0};
    int loader_threads{0};
    int aligner_threads{0};
    int barcoder_threads{0};
    int adapter_threads{0};
};

ThreadAllocations default_thread_allocations(int num_devices,
                                             int num_remora_threads,
                                             bool enable_aligner,
                                             bool enable_barcoder,
                                             bool adapter_trimming);

}  // namespace dorado::utils
