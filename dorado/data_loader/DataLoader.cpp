#if defined(USE_FAST5) || defined(USE_POD5)
#include "DataLoader.h"

#include "../utils/compat_utils.h"
#include "../utils/types.h"
#include "cxxpool.h"
#include "pod5_format/c_api.h"
#include "read_pipeline/ReadPipeline.h"
#include "utils/time_utils.h"
#include "vbz_plugin_user_utils.h"

#include <highfive/H5Easy.hpp>
#include <highfive/H5File.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <mutex>
#include <optional>

namespace {

// ReadID should be a drop-in replacement for read_id_t
    static_assert(sizeof(dorado::ReadID) == sizeof(read_id_t));

// 37 = number of bytes in UUID (32 hex digits + 4 dashes + null terminator)
    const uint32_t POD5_READ_ID_LEN = 37;

    void string_reader(HighFive::Attribute& attribute, std::string& target_str) {
        // Load as a variable string if possible
        if (attribute.getDataType().isVariableStr()) {
            attribute.read(target_str);
            return;
        }

        // Process as a fixed length string
        // Create landing buffer and H5 datatype
        size_t size = attribute.getDataType().getSize();
        std::vector<char> target_array(size);
        hid_t dtype = H5Tcopy(H5T_C_S1);
        H5Tset_size(dtype, size);

        // Copy to landing buffer
        if (H5Aread(attribute.getId(), dtype, target_array.data()) < 0) {
            throw std::runtime_error("Error during H5Aread of fixed length string");
        }

        // Extract to string
        target_str = std::string(target_array.data(), size);
        // It's possible the null terminator appears before the end of the string
        size_t eol_pos = target_str.find(char(0));
        if (eol_pos < target_str.size()) {
            target_str.resize(eol_pos);
        }
    };
}  // namespace

namespace dorado {

    std::shared_ptr<dorado::Read> process_pod5_read(size_t row,
                                                    Pod5ReadRecordBatch* batch,
                                                    Pod5FileReader* file,
                                                    const std::string path,
                                                    std::string device) {
        uint16_t read_table_version = 0;
        ReadBatchRowInfo_t read_data;
        if (pod5_get_read_batch_row_info_data(batch, row, READ_BATCH_ROW_INFO_VERSION, &read_data,
                                              &read_table_version) != POD5_OK) {
            spdlog::error("Failed to get read {}", row);
        }

        //Retrieve global information for the run
        RunInfoDictData_t* run_info_data;
        if (pod5_get_run_info(batch, read_data.run_info, &run_info_data) != POD5_OK) {
            spdlog::error("Failed to get Run Info {}{}", row, pod5_get_error_string());
        }
        auto run_acquisition_start_time_ms = run_info_data->acquisition_start_time_ms;
        auto run_sample_rate = run_info_data->sample_rate;

        char read_id_tmp[POD5_READ_ID_LEN];
        pod5_error_t err = pod5_format_read_id(read_data.read_id, read_id_tmp);
        std::string read_id_str(read_id_tmp);

        auto options = torch::TensorOptions().dtype(torch::kInt16);
        auto samples = torch::empty(read_data.num_samples, options);

        if (pod5_get_read_complete_signal(file, batch, row, read_data.num_samples,
                                          samples.data_ptr<int16_t>()) != POD5_OK) {
            spdlog::error("Failed to get read {} signal: {}", row, pod5_get_error_string());
        }

        auto new_read = std::make_shared<dorado::Read>();
        new_read->raw_data = samples;
        new_read->sample_rate = run_sample_rate;

        auto start_time_ms = run_acquisition_start_time_ms +
                             ((read_data.start_sample * 1000) /
                              (uint64_t)run_sample_rate);  // TODO check if this cast is needed
        auto start_time = utils::get_string_timestamp_from_unix_time(start_time_ms);
        new_read->run_acquisition_start_time_ms = run_acquisition_start_time_ms;
        new_read->start_time_ms = start_time_ms;
        new_read->scaling = read_data.calibration_scale;
        new_read->offset = read_data.calibration_offset;
        new_read->read_id = std::move(read_id_str);
        new_read->num_trimmed_samples = 0;
        new_read->attributes.read_number = read_data.read_number;
        new_read->attributes.fast5_filename = std::filesystem::path(path.c_str()).filename().string();
        new_read->attributes.mux = read_data.well;
        new_read->attributes.num_samples = read_data.num_samples;
        new_read->attributes.channel_number = read_data.channel;
        new_read->attributes.start_time = start_time;
        new_read->run_id = run_info_data->acquisition_id;
        new_read->start_sample = read_data.start_sample;
        new_read->end_sample = read_data.start_sample + read_data.num_samples;
        new_read->flowcell_id = run_info_data->flow_cell_id;
        new_read->is_duplex = false;

        if (pod5_free_run_info(run_info_data) != POD5_OK) {
            spdlog::error("Failed to free run info");
        }
        return new_read;
    }

    bool can_process_pod5_row(Pod5ReadRecordBatch_t* batch,
                              int row,
                              const std::optional<std::unordered_set<std::string>>& allowed_read_ids,
                              const std::unordered_set<std::string>& ignored_read_ids) {
        uint16_t read_table_version = 0;
        ReadBatchRowInfo_t read_data;
        if (pod5_get_read_batch_row_info_data(batch, row, READ_BATCH_ROW_INFO_VERSION, &read_data,
                                              &read_table_version) != POD5_OK) {
            spdlog::error("Failed to get read {}", row);
            return false;
        }

        char read_id_tmp[POD5_READ_ID_LEN];
        pod5_error_t err = pod5_format_read_id(read_data.read_id, read_id_tmp);
        std::string read_id_str(read_id_tmp);
        bool read_in_ignore_list = ignored_read_ids.find(read_id_str) != ignored_read_ids.end();
        bool read_in_read_list =
                !allowed_read_ids || (allowed_read_ids->find(read_id_str) != allowed_read_ids->end());
        if (!read_in_ignore_list && read_in_read_list) {
            return true;
        }
        return false;
    }

    void Pod5Destructor::operator()(Pod5FileReader_t* pod5) { pod5_close_and_free_reader(pod5); }

    void DataLoader::load_reads(const std::string& path,
                                bool recursive_file_loading,
                                ReadOrder traversal_order) {
        if (!std::filesystem::exists(path)) {
            spdlog::error("Requested input path {} does not exist!", path);
            return;
        }
        if (!std::filesystem::is_directory(path)) {
            spdlog::error("Requested input path {} is not a directory!", path);
            return;
        }

        auto iterate_directory = [&](const auto& iterator_fn) {
            switch (traversal_order) {
                case ReadOrder::BY_CHANNEL:
                    // If traversal in channel order is required, the following algorithm
                    // is used -
                    // 1. iterate through all the read metadata to collect channel information
                    // across all pod5 files
                    // 2. store the read list sorted by channel number
                    spdlog::info("> Reading read channel info");
                    load_read_channels(path, recursive_file_loading);
                    spdlog::info("> Processed read channel info");
                    // 3. for each channel, iterate through all files and in each iteration
                    // only load the reads that correspond to that channel.
                    for (int channel = 0; channel <= m_max_channel; channel++) {
                        for (const auto& entry : iterator_fn(path)) {
                            if (m_loaded_read_count == m_max_reads) {
                                break;
                            }
                            auto path = std::filesystem::path(entry);
                            std::string ext = path.extension().string();
                            std::transform(ext.begin(), ext.end(), ext.begin(),
                                           [](unsigned char c) { return std::tolower(c); });
                            if (ext == ".fast5") {
                                throw std::runtime_error(
                                        "Traversing reads by channel is only availabls for POD5. "
                                        "Encountered FAST5 at " +
                                        path.string());
                            } else if (ext == ".pod5") {
                                auto& channel_to_read_ids = m_file_channel_read_order_map.at(path.string());
                                auto& read_ids = channel_to_read_ids[channel];
                                if (!read_ids.empty()) {
                                    load_pod5_reads_from_file_by_read_ids(path.string(), read_ids);
                                }
                            }
                        }
                    }
                    break;
                case ReadOrder::UNRESTRICTED:
                    for (const auto& entry : iterator_fn(path)) {
                        if (m_loaded_read_count == m_max_reads) {
                            break;
                        }
                        std::string ext = std::filesystem::path(entry).extension().string();
                        std::transform(ext.begin(), ext.end(), ext.begin(),
                                       [](unsigned char c) { return std::tolower(c); });
                        if (ext == ".fast5") {
                            load_fast5_reads_from_file(entry.path().string());
                        } else if (ext == ".pod5") {
                            load_pod5_reads_from_file(entry.path().string());
                        }
                    }
                    break;
                default:
                    throw std::runtime_error("Unsupported traversal order detected: " +
                                             dorado::to_string(traversal_order));
            }
        };

        if (recursive_file_loading) {
            iterate_directory([](const auto& path) {
                return std::filesystem::recursive_directory_iterator(path);
            });
        } else {
            iterate_directory(
                    [](const auto& path) { return std::filesystem::directory_iterator(path); });
        }
    }

    int DataLoader::get_num_reads(std::string data_path,
                                  std::optional<std::unordered_set<std::string>> read_list,
                                  const std::unordered_set<std::string>& ignore_read_list,
                                  bool recursive_file_loading) {
        size_t num_reads = 0;

        auto iterate_directory = [&](const auto& iterator_fn) {
            for (const auto& entry : iterator_fn(data_path)) {
                std::string ext = std::filesystem::path(entry).extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(),
                               [](unsigned char c) { return std::tolower(c); });
                if (ext == ".pod5") {
                    pod5_init();

                    // Open the file ready for walking:
                    Pod5FileReader_t* file = pod5_open_file(entry.path().string().c_str());

                    size_t read_count;
                    pod5_get_read_count(file, &read_count);
                    if (!file) {
                        spdlog::error("Failed to open file {}: {}", entry.path().string().c_str(),
                                      pod5_get_error_string());
                    }

                    num_reads += read_count;
                    if (pod5_close_and_free_reader(file) != POD5_OK) {
                        spdlog::error("Failed to close and free POD5 reader");
                    }
                }
            }
        };

        if (recursive_file_loading) {
            iterate_directory([](const auto& path) {
                return std::filesystem::recursive_directory_iterator(path);
            });
        } else {
            iterate_directory(
                    [](const auto& path) { return std::filesystem::directory_iterator(path); });
        }

        // Remove the reads in the ignore list from the total dataset read count.
        num_reads -= ignore_read_list.size();

        if (read_list) {
            // Get the unique read ids in the read list, since everything in the ignore
            // list will be skipped over.
            std::vector<std::string> final_read_list;
            std::set_difference(read_list->begin(), read_list->end(), ignore_read_list.begin(),
                                ignore_read_list.end(),
                                std::inserter(final_read_list, final_read_list.begin()));
            num_reads = std::min(num_reads, final_read_list.size());
        }

        return num_reads;
    }

    void DataLoader::load_read_channels(std::string data_path, bool recursive_file_loading) {
        auto iterate_directory = [&](const auto& iterator_fn) {
            for (const auto& entry : iterator_fn(data_path)) {
                auto file_path = std::filesystem::path(entry);
                std::string ext = file_path.extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(),
                               [](unsigned char c) { return std::tolower(c); });
                if (ext != ".pod5") {
                    continue;
                }
                pod5_init();

                // Use a std::map to store by sorted channel order.
                m_file_channel_read_order_map.emplace(file_path.string(), channel_to_read_id_t());
                auto& channel_to_read_id = m_file_channel_read_order_map[file_path.string()];

                // Open the file ready for walking:
                Pod5FileReader_t* file = pod5_open_file(file_path.string().c_str());

                if (!file) {
                    spdlog::error("Failed to open file {}: {}", file_path.string().c_str(),
                                  pod5_get_error_string());
                    continue;
                }
                std::size_t batch_count = 0;
                if (pod5_get_read_batch_count(&batch_count, file) != POD5_OK) {
                    spdlog::error("Failed to query batch count: {}", pod5_get_error_string());
                }

                for (std::size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
                    Pod5ReadRecordBatch_t* batch = nullptr;
                    if (pod5_get_read_batch(&batch, file, batch_index) != POD5_OK) {
                        spdlog::error("Failed to get batch: {}", pod5_get_error_string());
                        continue;
                    }

                    std::size_t batch_row_count = 0;
                    if (pod5_get_read_batch_row_count(&batch_row_count, batch) != POD5_OK) {
                        spdlog::error("Failed to get batch row count");
                        continue;
                    }

                    for (std::size_t row = 0; row < batch_row_count; ++row) {
                        uint16_t read_table_version = 0;
                        ReadBatchRowInfo_t read_data;
                        if (pod5_get_read_batch_row_info_data(batch, row, READ_BATCH_ROW_INFO_VERSION,
                                                              &read_data,
                                                              &read_table_version) != POD5_OK) {
                            spdlog::error("Failed to get read {}", row);
                            continue;
                        }

                        int channel = read_data.channel;

                        // Update maximum number of channels encountered.
                        m_max_channel = std::max(m_max_channel, channel);

                        // Store the read_id in the channel's list.
                        ReadID read_id;
                        std::memcpy(read_id.data(), read_data.read_id, POD5_READ_ID_SIZE);
                        channel_to_read_id[channel].push_back(std::move(read_id));
                    }

                    if (pod5_free_read_batch(batch) != POD5_OK) {
                        spdlog::error("Failed to release batch");
                    }
                }
                if (pod5_close_and_free_reader(file) != POD5_OK) {
                    spdlog::error("Failed to close and free POD5 reader");
                }
            }
        };

        if (recursive_file_loading) {
            iterate_directory([](const auto& path) {
                return std::filesystem::recursive_directory_iterator(path);
            });
        } else {
            iterate_directory(
                    [](const auto& path) { return std::filesystem::directory_iterator(path); });
        }
    }

    std::unordered_map<std::string, ReadGroup> DataLoader::load_read_groups(
            std::string data_path,
            std::string model_path,
            bool recursive_file_loading) {
        std::unordered_map<std::string, ReadGroup> read_groups;

        auto iterate_directory = [&](const auto& iterator_fn) {
            for (const auto& entry : iterator_fn(data_path)) {
                std::string ext = std::filesystem::path(entry).extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(),
                               [](unsigned char c) { return std::tolower(c); });
                if (ext == ".pod5") {
                    pod5_init();

                    // Open the file ready for walking:
                    Pod5FileReader_t* file = pod5_open_file(entry.path().string().c_str());

                    if (!file) {
                        spdlog::error("Failed to open file {}: {}", entry.path().string().c_str(),
                                      pod5_get_error_string());
                    } else {
                        // First get the run info count
                        run_info_index_t run_info_count;
                        pod5_get_file_run_info_count(file, &run_info_count);
                        for (run_info_index_t idx = 0; idx < run_info_count; idx++) {
                            RunInfoDictData_t* run_info_data;
                            pod5_get_file_run_info(file, idx, &run_info_data);

                            auto exp_start_time_ms = run_info_data->acquisition_start_time_ms;
                            std::string flowcell_id = run_info_data->flow_cell_id;
                            std::string device_id = run_info_data->system_name;
                            std::string run_id = run_info_data->acquisition_id;
                            std::string sample_id = run_info_data->sample_id;

                            if (pod5_free_run_info(run_info_data) != POD5_OK) {
                                spdlog::error("Failed to free run info");
                            }

                            std::string id = run_id + "_" + model_path;
                            read_groups[id] = ReadGroup{
                                    run_id,
                                    model_path,
                                    flowcell_id,
                                    device_id,
                                    utils::get_string_timestamp_from_unix_time(exp_start_time_ms),
                                    sample_id};
                        }
                        if (pod5_close_and_free_reader(file) != POD5_OK) {
                            spdlog::error("Failed to close and free POD5 reader");
                        }
                    }
                }
            }
        };

        if (recursive_file_loading) {
            iterate_directory([](const auto& path) {
                return std::filesystem::recursive_directory_iterator(path);
            });
        } else {
            iterate_directory(
                    [](const auto& path) { return std::filesystem::directory_iterator(path); });
        }

        return read_groups;
    }

    uint16_t DataLoader::get_sample_rate(std::string data_path, bool recursive_file_loading) {
        std::optional<uint16_t> sample_rate = std::nullopt;

        auto iterate_directory = [&](const auto& iterator_fn) {
            for (const auto& entry : iterator_fn((data_path))) {
                std::string ext = std::filesystem::path(entry).extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(),
                               [](unsigned char c) { return std::tolower(c); });
                auto file_path = entry.path().string();
                if (ext == ".pod5") {
                    pod5_init();

                    // Open the file ready for walking:
                    Pod5FileReader_t* file = pod5_open_file(file_path.c_str());

                    if (!file) {
                        spdlog::error("Failed to open file {}: {}", file_path.c_str(),
                                      pod5_get_error_string());
                    } else {
                        // First get the run info count
                        run_info_index_t run_info_count;
                        if (pod5_get_file_run_info_count(file, &run_info_count) != POD5_OK) {
                            spdlog::error("Failed to fetch POD5 run info count for file {} : {}",
                                          file_path.c_str(), pod5_get_error_string());
                            continue;
                        }
                        if (run_info_count > static_cast<run_info_index_t>(0)) {
                            RunInfoDictData_t* run_info_data;
                            if (pod5_get_file_run_info(file, 0, &run_info_data) != POD5_OK) {
                                spdlog::error(
                                        "Failed to fetch POD5 run info dict for file {} and run info "
                                        "index 0: {}",
                                        file_path.c_str(), pod5_get_error_string());
                                continue;
                            }
                            sample_rate = run_info_data->sample_rate;

                            if (pod5_free_run_info(run_info_data) != POD5_OK) {
                                spdlog::error(
                                        "Failed to free POD5 run info for file {} and run info index 0",
                                        file_path.c_str());
                            }
                        }
                    }

                    if (pod5_close_and_free_reader(file) != POD5_OK) {
                        spdlog::error("Failed to close and free POD5 reader for file {}",
                                      file_path.c_str());
                    }
                } else if (ext == ".fast5") {
                    H5Easy::File file(entry.path().string(), H5Easy::File::ReadOnly);
                    HighFive::Group reads = file.getGroup("/");
                    int num_reads = reads.getNumberObjects();

                    if (num_reads > 0) {
                        auto read_id = reads.getObjectName(0);
                        HighFive::Group read = reads.getGroup(read_id);

                        HighFive::Group channel_id_group = read.getGroup("channel_id");
                        HighFive::Attribute sampling_rate_attr =
                                channel_id_group.getAttribute("sampling_rate");

                        float sampling_rate;
                        sampling_rate_attr.read(sampling_rate);
                        sample_rate = static_cast<uint16_t>(sampling_rate);
                    }
                }

                // Break out of loop if sample rate is found.
                if (sample_rate) {
                    break;
                }
            }
        };

        if (recursive_file_loading) {
            iterate_directory([](const auto& path) {
                return std::filesystem::recursive_directory_iterator(path);
            });
        } else {
            iterate_directory(
                    [](const auto& path) { return std::filesystem::directory_iterator(path); });
        }

        if (sample_rate) {
            return *sample_rate;
        } else {
            throw std::runtime_error("Unable to determine sample rate for data.");
        }
    }

    void DataLoader::load_pod5_reads_from_file_by_read_ids(const std::string& path,
                                                           const std::vector<ReadID>& read_ids) {
        pod5_init();

        // Open the file ready for walking:
        // TODO: The earlier implementation was caching the pod5 readers into a
        // map and re-using it during each iteration. However, we found a leak
        // in the pod5 traversal API which persists unless the reader is opened
        // and closed everytime. So the caching logic was reverted until the
        // leak is fixed in pod5 API.
        Pod5FileReader_t* file = pod5_open_file(path.c_str());

        if (!file) {
            spdlog::error("Failed to open file {}: {}", path, pod5_get_error_string());
            return;
        }

        std::vector<uint8_t> read_id_array(POD5_READ_ID_SIZE * read_ids.size());
        for (int i = 0; i < read_ids.size(); i++) {
            std::memcpy(read_id_array.data() + POD5_READ_ID_SIZE * i, read_ids[i].data(),
                        POD5_READ_ID_SIZE);
        }

        std::size_t batch_count = 0;
        if (pod5_get_read_batch_count(&batch_count, file) != POD5_OK) {
            spdlog::error("Failed to query batch count: {}", pod5_get_error_string());
        }

        std::vector<std::uint32_t> traversal_batch_counts(batch_count);
        std::vector<std::uint32_t> traversal_batch_rows(read_ids.size());
        size_t find_success_count;
        pod5_error_t err = pod5_plan_traversal(file, read_id_array.data(), read_ids.size(),
                                               traversal_batch_counts.data(),
                                               traversal_batch_rows.data(), &find_success_count);
        if (err != POD5_OK) {
            spdlog::error("Couldn't create plan for {} with reads {}", path, read_ids.size());
            return;
        }

        if (find_success_count != read_ids.size()) {
            spdlog::error("Reads found by plan {}, reads in input {}", find_success_count,
                          read_ids.size());
            throw std::runtime_error("Plan traveral didn't yield correct number of reads");
        }

        cxxpool::thread_pool pool{m_num_worker_threads};

        uint32_t row_offset = 0;
        for (std::size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
            if (m_loaded_read_count == m_max_reads) {
                break;
            }
            Pod5ReadRecordBatch_t* batch = nullptr;
            if (pod5_get_read_batch(&batch, file, batch_index) != POD5_OK) {
                spdlog::error("Failed to get batch: {}", pod5_get_error_string());
                continue;
            }

            std::vector<std::future<std::shared_ptr<Read>>> futures;
            for (std::size_t row_idx = 0; row_idx < traversal_batch_counts[batch_index]; row_idx++) {
                uint32_t row = traversal_batch_rows[row_idx + row_offset];

                if (can_process_pod5_row(batch, row, m_allowed_read_ids, m_ignored_read_ids)) {
                    futures.push_back(pool.push(process_pod5_read, row, batch, file, path, m_device));
                }
            }

            for (auto& v : futures) {
                auto read = v.get();
                m_pipeline.push_message(std::move(read));
                m_loaded_read_count++;
            }

            if (pod5_free_read_batch(batch) != POD5_OK) {
                spdlog::error("Failed to release batch");
            }

            row_offset += traversal_batch_counts[batch_index];
        }
        if (pod5_close_and_free_reader(file) != POD5_OK) {
            spdlog::error("Failed to close and free POD5 reader");
        }
    }

    void DataLoader::load_pod5_reads_from_file(const std::string& path) {
        pod5_init();

        // Open the file ready for walking:
        Pod5FileReader_t* file = pod5_open_file(path.c_str());

        if (!file) {
            spdlog::error("Failed to open file {}: {}", path, pod5_get_error_string());
        }

        std::size_t batch_count = 0;
        if (pod5_get_read_batch_count(&batch_count, file) != POD5_OK) {
            spdlog::error("Failed to query batch count: {}", pod5_get_error_string());
        }

        cxxpool::thread_pool pool{m_num_worker_threads};

        for (std::size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
            if (m_loaded_read_count == m_max_reads) {
                break;
            }
            Pod5ReadRecordBatch_t* batch = nullptr;
            if (pod5_get_read_batch(&batch, file, batch_index) != POD5_OK) {
                spdlog::error("Failed to get batch: {}", pod5_get_error_string());
            }

            std::size_t batch_row_count = 0;
            if (pod5_get_read_batch_row_count(&batch_row_count, batch) != POD5_OK) {
                spdlog::error("Failed to get batch row count");
            }
            batch_row_count = std::min(batch_row_count, m_max_reads - m_loaded_read_count);

            std::vector<std::future<std::shared_ptr<Read>>> futures;

            for (std::size_t row = 0; row < batch_row_count; ++row) {
                // TODO - check the read ID here, for each one, only send the row if it is in the list of ones we care about

                if (can_process_pod5_row(batch, row, m_allowed_read_ids, m_ignored_read_ids)) {
                    futures.push_back(pool.push(process_pod5_read, row, batch, file, path, m_device));
                }
            }

            for (auto& v : futures) {
                auto read = v.get();
                m_pipeline.push_message(std::move(read));
                m_loaded_read_count++;
            }

            if (pod5_free_read_batch(batch) != POD5_OK) {
                spdlog::error("Failed to release batch");
            }
        }
        if (pod5_close_and_free_reader(file) != POD5_OK) {
            spdlog::error("Failed to close and free POD5 reader");
        }
    }

    void DataLoader::load_fast5_reads_from_file(const std::string& path) {
        // Read the file into a vector of torch tensors
        H5Easy::File file(path, H5Easy::File::ReadOnly);
        HighFive::Group reads = file.getGroup("/");
        int num_reads = reads.getNumberObjects();

        for (int i = 0; i < num_reads && m_loaded_read_count < m_max_reads; i++) {
            auto read_id = reads.getObjectName(i);
            HighFive::Group read = reads.getGroup(read_id);

            // Fetch the digitisation parameters
            HighFive::Group channel_id_group = read.getGroup("channel_id");
            HighFive::Attribute digitisation_attr = channel_id_group.getAttribute("digitisation");
            HighFive::Attribute range_attr = channel_id_group.getAttribute("range");
            HighFive::Attribute offset_attr = channel_id_group.getAttribute("offset");
            HighFive::Attribute sampling_rate_attr = channel_id_group.getAttribute("sampling_rate");
            HighFive::Attribute channel_number_attr = channel_id_group.getAttribute("channel_number");

            int32_t channel_number;
            if (channel_number_attr.getDataType().string().substr(0, 6) == "String") {
                std::string channel_number_string;
                string_reader(channel_number_attr, channel_number_string);
                std::istringstream channel_stream(channel_number_string);
                channel_stream >> channel_number;
            } else {
                channel_number_attr.read(channel_number);
            }

            float digitisation;
            digitisation_attr.read(digitisation);
            float range;
            range_attr.read(range);
            float offset;
            offset_attr.read(offset);
            float sampling_rate;
            sampling_rate_attr.read(sampling_rate);

            HighFive::Group raw = read.getGroup("Raw");
            auto ds = raw.getDataSet("Signal");
            if (ds.getDataType().string() != "Integer16")
                throw std::runtime_error("Invalid FAST5 Signal data type of " +
                                         ds.getDataType().string());

            auto options = torch::TensorOptions().dtype(torch::kInt16);
            auto samples = torch::empty(ds.getElementCount(), options);
            ds.read(samples.data_ptr<int16_t>());

            HighFive::Attribute mux_attr = raw.getAttribute("start_mux");
            HighFive::Attribute read_number_attr = raw.getAttribute("read_number");
            HighFive::Attribute start_time_attr = raw.getAttribute("start_time");
            HighFive::Attribute read_id_attr = raw.getAttribute("read_id");
            uint32_t mux;
            uint32_t read_number;
            uint64_t start_time;
            mux_attr.read(mux);
            read_number_attr.read(read_number);
            start_time_attr.read(start_time);
            string_reader(read_id_attr, read_id);

            std::string fast5_filename = std::filesystem::path(path).filename().string();

            HighFive::Group tracking_id_group = read.getGroup("tracking_id");
            HighFive::Attribute exp_start_time_attr = tracking_id_group.getAttribute("exp_start_time");
            std::string exp_start_time;
            string_reader(exp_start_time_attr, exp_start_time);

            auto start_time_str = utils::adjust_time(exp_start_time,
                                                     static_cast<uint32_t>(start_time / sampling_rate));

            auto new_read = std::make_shared<Read>();
            new_read->sample_rate = sampling_rate;
            new_read->raw_data = samples;
            new_read->digitisation = digitisation;
            new_read->range = range;
            new_read->offset = offset;
            new_read->scaling = range / digitisation;
            new_read->read_id = read_id;
            new_read->num_trimmed_samples = 0;
            new_read->attributes.mux = mux;
            new_read->attributes.read_number = read_number;
            new_read->attributes.channel_number = channel_number;
            new_read->attributes.start_time = start_time_str;
            new_read->attributes.fast5_filename = fast5_filename;
            new_read->is_duplex = false;

            if (!m_allowed_read_ids ||
                (m_allowed_read_ids->find(new_read->read_id) != m_allowed_read_ids->end())) {
                m_pipeline.push_message(std::move(new_read));
                m_loaded_read_count++;
            }
        }
    }

    DataLoader::DataLoader(Pipeline& pipeline,
                           const std::string& device,
                           size_t num_worker_threads,
                           size_t max_reads,
                           std::optional<std::unordered_set<std::string>> read_list,
                           std::unordered_set<std::string> read_ignore_list)
            : m_pipeline(pipeline),
              m_device(device),
              m_num_worker_threads(num_worker_threads),
              m_allowed_read_ids(std::move(read_list)),
              m_ignored_read_ids(std::move(read_ignore_list)) {
        m_max_reads = max_reads == 0 ? std::numeric_limits<decltype(m_max_reads)>::max() : max_reads;
        assert(m_num_worker_threads > 0);
        static std::once_flag vbz_init_flag;
        std::call_once(vbz_init_flag, vbz_register);
    }

    stats::NamedStats DataLoader::sample_stats() const {
        return stats::NamedStats{{"loaded_read_count", static_cast<double>(m_loaded_read_count)}};
    }
}  // namespace dorado
#else
#include "DataLoader.h"

#include "../utils/compat_utils.h"
#include "../utils/types.h"
#include "cxxpool.h"
#include "read_pipeline/ReadPipeline.h"
#include "utils/time_utils.h"
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <mutex>
#include <optional>

#include "slow5/slow5.h"
#include "slow5_extra.h"
#include "slow5_thread.h"
#include <slow5/slow5_mt.h>

namespace dorado {
    void DataLoader::load_reads(const std::string &path,
                                bool recursive_file_loading,
                                ReadOrder traversal_order) {
        if (!std::filesystem::exists(path)) {
            spdlog::error("Requested input path {} does not exist!", path);
            return;
        }

        auto iterate_directory = [&](const auto& iterator_fn) {
            switch (traversal_order) {
            case ReadOrder::BY_CHANNEL:
                // If traversal in channel order is required, the following algorithm
                // is used -
                // 1. iterate through all the read metadata to collect channel information
                // across all pod5 files
                // 2. store the read list sorted by channel number
                spdlog::info("> Reading read channel info");
                load_read_channels(path, recursive_file_loading);
                spdlog::info("> Processed read channel info");
                // 3. for each channel, iterate through all files and in each iteration
                // only load the reads that correspond to that channel.
                for (int channel = 0; channel <= m_max_channel; channel++) {
                    for (const auto& entry : iterator_fn(path)) {
                        if (m_loaded_read_count == m_max_reads) {
                            break;
                        }
                        auto path = std::filesystem::path(entry);
                        std::string ext = path.extension().string();
                        std::transform(ext.begin(), ext.end(), ext.begin(),
                                       [](unsigned char c) { return std::tolower(c); });
                        if (ext == ".slow5" || ext == ".blow5") {
                            auto& channel_to_read_ids = m_file_channel_read_order_map.at(path.string());
                            auto& read_ids = channel_to_read_ids[channel];
                            if (!read_ids.empty()) {
                                load_slow5_reads_from_file_by_read_ids(entry.path().string(), read_ids);
                            }
                        }
                    }
                }
                break;
            case ReadOrder::UNRESTRICTED:
                for (const auto& entry : iterator_fn(path)) {
                    if (m_loaded_read_count == m_max_reads) {
                        break;
                    }
                    std::string ext = std::filesystem::path(entry).extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(),
                                   [](unsigned char c) { return std::tolower(c); });
                    if (ext == ".slow5" || ext == ".blow5") {
                        load_slow5_reads_from_file(entry.path().string());
                    }
                }
                break;
            default:
                throw std::runtime_error("Unsupported traversal order detected: " +
                                         dorado::to_string(traversal_order));
            }
        };

        if (recursive_file_loading) {
            iterate_directory([](const auto& path) {
                return std::filesystem::recursive_directory_iterator(path);
            });
        } else {
            if (!std::filesystem::is_directory(path)) {
                std::string ext = std::filesystem::path(path).extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
                if (ext == ".slow5" || ext == ".blow5") {
                    switch (traversal_order) {
                    case ReadOrder::BY_CHANNEL:
                        spdlog::info("> Reading read channel info");
                        load_read_channels(path, recursive_file_loading);
                        spdlog::info("> Processed read channel info");
                        // 3. for each channel, iterate through all files and in each iteration
                        // only load the reads that correspond to that channel.
                        for (int channel = 0; channel <= m_max_channel; channel++) {
                            if (m_loaded_read_count == m_max_reads) {
                                break;
                            }
                            auto& channel_to_read_ids = m_file_channel_read_order_map.at(path);
                            auto& read_ids = channel_to_read_ids[channel];
                            if (!read_ids.empty()) {
                                load_slow5_reads_from_file_by_read_ids(path, read_ids);
                            }
                        }
                        break;
                    case ReadOrder::UNRESTRICTED:
                        if (m_loaded_read_count == m_max_reads) {
                            break;
                        }
                        load_slow5_reads_from_file(path);
                        break;
                    default:
                        throw std::runtime_error("Unsupported traversal order detected: " +
                                                 dorado::to_string(traversal_order));
                    }
                }
            }else{
                iterate_directory(
                        [](const auto &path) { return std::filesystem::directory_iterator(path); });
            }
        }
    }

    int DataLoader::get_num_reads(std::string data_path,
                                  std::optional<std::unordered_set<std::string>> read_list,
                                  const std::unordered_set<std::string> &ignore_read_list,
                                  bool recursive_file_loading) {
        size_t num_reads = 0;

        auto iterate_directory = [&](const auto &iterator_fn) {
            for (const auto &entry: iterator_fn(data_path)) {
                std::string ext = std::filesystem::path(entry).extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(),
                               [](unsigned char c) { return std::tolower(c); });
                auto file_path = entry.path().string();
                if (ext != ".slow5" and ext != ".blow5") {
                    continue;
                }
                slow5_file_t *sp = slow5_open(file_path.c_str(),"r");
                if(sp==NULL){
                    fprintf(stderr,"Error in opening file\n");
                    exit(EXIT_FAILURE);
                }
                size_t bytes;
                char *mem;
                while ((mem = (char *) slow5_get_next_mem(&bytes, sp))) {
                    free(mem);
                    num_reads++;
                }
                if (slow5_errno != SLOW5_ERR_EOF) {
                    fprintf(stderr,"Error reading the file.%s","");
                    exit(EXIT_FAILURE);
                }
                slow5_close(sp);
            }
        };

        if (recursive_file_loading) {
            iterate_directory([](const auto &path) {
                return std::filesystem::recursive_directory_iterator(path);
            });
        } else {
            if(!std::filesystem::is_directory(data_path)) {
                std::string ext = std::filesystem::path(data_path).extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(),
                               [](unsigned char c) { return std::tolower(c); });
                slow5_file_t *sp = slow5_open(data_path.c_str(),"r");
                if(sp==NULL){
                    fprintf(stderr,"Error in opening file\n");
                    exit(EXIT_FAILURE);
                }
                size_t bytes;
                char *mem;
                while ((mem = (char *) slow5_get_next_mem(&bytes, sp))) {
                    free(mem);
                    num_reads++;
                }
                if (slow5_errno != SLOW5_ERR_EOF) {
                    fprintf(stderr,"Error reading the file.%s","");
                    exit(EXIT_FAILURE);
                }
                slow5_close(sp);
            }else{
                iterate_directory(
                        [](const auto &path) { return std::filesystem::directory_iterator(path); });
            }
        }

        // Remove the reads in the ignore list from the total dataset read count.
        num_reads -= ignore_read_list.size();

        if (read_list) {
            // Get the unique read ids in the read list, since everything in the ignore
            // list will be skipped over.
            std::vector<std::string> final_read_list;
            std::set_difference(read_list->begin(), read_list->end(), ignore_read_list.begin(),
                                ignore_read_list.end(),
                                std::inserter(final_read_list, final_read_list.begin()));
            num_reads = std::min(num_reads, final_read_list.size());
        }
        spdlog::debug("num_reads {}", num_reads);
        return num_reads;
    }

    void DataLoader::load_read_channels(std::string data_path, bool recursive_file_loading) {
        auto iterate_directory = [&](const auto &iterator_fn) {
            for (const auto& entry : iterator_fn(data_path)) {
                auto file_path = std::filesystem::path(entry);
                std::string ext = file_path.extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(),
                               [](unsigned char c) { return std::tolower(c); });
                if (ext != ".blow5" and ext != ".slow5") {
                    continue;
                }
                std::string file_path_str = file_path.string();
                // Use a std::map to store by sorted channel order.
                m_file_channel_read_order_map.emplace(file_path_str, channel_to_read_id_t());
                auto& channel_to_read_id = m_file_channel_read_order_map[file_path_str];

                slow5_file_t *sp = slow5_open(file_path_str.c_str(),"r");
                if(sp==NULL){
                    fprintf(stderr,"Error in opening file\n");
                    exit(EXIT_FAILURE);
                }
                slow5_rec_t **rec = NULL;
                int ret=0;

                slow5_mt_t *mt = slow5_init_mt(slow5_threads,sp);
                slow5_batch_t *read_batch = slow5_init_batch(slow5_batchsize);

                while((ret = slow5_get_next_batch(mt,read_batch,slow5_batchsize)) > 0){

                    for(int i=0;i<ret;i++){
                        rec = read_batch->slow5_rec;

                        uint64_t len; //length of the array
                        char* channel_number = slow5_aux_get_string(rec[i], "channel_number", &len, &ret);
                        if(ret!=0){
                            fprintf(stderr,"Error in getting auxiliary attribute from the file. Error code %d\n",ret);
                            exit(EXIT_FAILURE);
                        }
                        if (channel_number == NULL){ //check if the field value exists and print the value
                            fprintf(stderr,"channel_number is missing for the record %s\n", rec[i]->read_id);
                            exit(EXIT_FAILURE);
                        } else{
                            int channel = atoi(channel_number);
                            // Update maximum number of channels encountered.
                            m_max_channel = std::max(m_max_channel, channel);
                            // Store the read_id in the channel's list.
                            ReadID read_id;
                            std::memcpy(read_id.data(), rec[i]->read_id, POD5_READ_ID_SIZE);
                            channel_to_read_id[channel].push_back(std::move(read_id));
                        }
                    }

                    if(ret<slow5_batchsize){ //this indicates nothing left to read //need to handle errors
                        break;
                    }
                }

                slow5_free_batch(read_batch);
                slow5_free_mt(mt);
                slow5_close(sp);
            }
        };

        if (recursive_file_loading) {
            iterate_directory([](const auto &path) {
                return std::filesystem::recursive_directory_iterator(path);
            });
        } else {
            if(!std::filesystem::is_directory(data_path)) {
                std::string ext = std::filesystem::path(data_path).extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(),
                               [](unsigned char c) { return std::tolower(c); });
                // Use a std::map to store by sorted channel order.
                m_file_channel_read_order_map.emplace(data_path, channel_to_read_id_t());
                auto& channel_to_read_id = m_file_channel_read_order_map[data_path];

                slow5_file_t *sp = slow5_open(data_path.c_str(),"r");
                if(sp==NULL){
                    fprintf(stderr,"Error in opening file\n");
                    exit(EXIT_FAILURE);
                }
                slow5_rec_t **rec = NULL;
                int ret=0;

                slow5_mt_t *mt = slow5_init_mt(slow5_threads,sp);
                slow5_batch_t *read_batch = slow5_init_batch(slow5_batchsize);

                while((ret = slow5_get_next_batch(mt,read_batch,slow5_batchsize)) > 0){

                    for(int i=0;i<ret;i++){
                        rec = read_batch->slow5_rec;

                        uint64_t len; //length of the array
                        char* channel_number = slow5_aux_get_string(rec[i], "channel_number", &len, &ret);
                        if(ret!=0){
                            fprintf(stderr,"Error in getting auxiliary attribute from the file. Error code %d\n",ret);
                            exit(EXIT_FAILURE);
                        }
                        if (channel_number == NULL){ //check if the field value exists and print the value
                            fprintf(stderr,"channel_number is missing for the record %s\n", rec[i]->read_id);
                            exit(EXIT_FAILURE);
                        } else{
                            int channel = atoi(channel_number);
                            // Update maximum number of channels encountered.
                            m_max_channel = std::max(m_max_channel, channel);
                            // Store the read_id in the channel's list.
                            ReadID read_id;
                            std::memcpy(read_id.data(), rec[i]->read_id, POD5_READ_ID_SIZE);
                            channel_to_read_id[channel].push_back(std::move(read_id));
                        }
                    }

                    if(ret<slow5_batchsize){ //this indicates nothing left to read //need to handle errors
                        break;
                    }
                }

                slow5_free_batch(read_batch);
                slow5_free_mt(mt);
                slow5_close(sp);
            }else{
                iterate_directory(
                        [](const auto &path) { return std::filesystem::directory_iterator(path); });
            }
        }
    }

    std::unordered_map<std::string, ReadGroup> DataLoader::load_read_groups(
            std::string data_path,
            std::string model_path,
            bool recursive_file_loading) {
        std::unordered_map<std::string, ReadGroup> read_groups;
        auto iterate_directory = [&](const auto &iterator_fn) {
            for (const auto &entry: iterator_fn(data_path)) {
                std::string ext = std::filesystem::path(entry).extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(),
                               [](unsigned char c) { return std::tolower(c); });
                auto file_path = entry.path().string();
                if (ext != ".slow5" and ext != ".blow5") {
                    continue;
                }
                slow5_file_t *sp = slow5_open(file_path.c_str(),"r");
                if(sp==NULL){
                    fprintf(stderr,"Error in opening file\n");
                    exit(EXIT_FAILURE);
                }
                int64_t read_group_count = sp->header->num_read_groups;
                for(int64_t j=0; j<read_group_count; j++){
                    char* run_id_c = slow5_hdr_get("run_id", j, sp->header);
                    if(!run_id_c){
                        fprintf(stderr,"No run_id found in %s.", file_path.c_str());
                        exit(EXIT_FAILURE);
                    }
                    std::string run_id = std::string(run_id_c);
                    char* flow_cell_id_c = slow5_hdr_get("flow_cell_id", j, sp->header);
                    if(!flow_cell_id_c){
                        fprintf(stderr,"No flowcell_id found in %s.", file_path.c_str());
                        exit(EXIT_FAILURE);
                    }
                    std::string flowcell_id = std::string(flow_cell_id_c);
                    char* device_id_c = slow5_hdr_get("device_id", j, sp->header);
                    if(!device_id_c){
                        fprintf(stderr,"No device_id found in %s.", file_path.c_str());
                        exit(EXIT_FAILURE);
                    }
                    std::string device_id = std::string(device_id_c);
                    char* exp_start_time_ms_c = slow5_hdr_get("acquisition_start_time", j, sp->header);
                    if(!exp_start_time_ms_c){
                        exp_start_time_ms_c = slow5_hdr_get("exp_start_time", j, sp->header);
                        if(!exp_start_time_ms_c) {
                            fprintf(stderr, "Neither acquisition_start_time nor exp_start_time found in %s.",
                                    file_path.c_str());
                            exit(EXIT_FAILURE);
                        }
                    }
                    std::string exp_start_time_ms = std::string(exp_start_time_ms_c);
                    char* sample_id_c = slow5_hdr_get("sample_id", j, sp->header);
                    if(!sample_id_c){
                        fprintf(stderr,"No sample_id found in %s.", file_path.c_str());
                        exit(EXIT_FAILURE);
                    }
                    std::string sample_id = std::string(sample_id_c);
                    std::string id = run_id + "_" + model_path;
                    read_groups[id] = ReadGroup{
                            run_id,
                            model_path,
                            flowcell_id,
                            device_id,
                            exp_start_time_ms,
                            sample_id};
                }
                slow5_close(sp);
            }
        };

        if (recursive_file_loading) {
            iterate_directory([](const auto &path) {
                return std::filesystem::recursive_directory_iterator(path);
            });
        } else {
            if(std::filesystem::is_directory(data_path)){
                iterate_directory(
                        [](const auto &path) { return std::filesystem::directory_iterator(path); });
            } else{
                std::string ext = std::filesystem::path(data_path).extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(),
                               [](unsigned char c) { return std::tolower(c); });
                slow5_file_t *sp = slow5_open(data_path.c_str(),"r");
                if(sp==NULL){
                    fprintf(stderr,"Error in opening file\n");
                    exit(EXIT_FAILURE);
                }
                int64_t read_group_count = sp->header->num_read_groups;
                for(int64_t j=0; j<read_group_count; j++){
                    char* run_id_c = slow5_hdr_get("run_id", j, sp->header);
                    if(!run_id_c){
                        fprintf(stderr,"No run_id found in %s.", data_path.c_str());
                        exit(EXIT_FAILURE);
                    }
                    std::string run_id = std::string(run_id_c);
                    char* flow_cell_id_c = slow5_hdr_get("flow_cell_id", j, sp->header);
                    if(!flow_cell_id_c){
                        fprintf(stderr,"No flowcell_id found in %s.", data_path.c_str());
                        exit(EXIT_FAILURE);
                    }
                    std::string flowcell_id = std::string(flow_cell_id_c);
                    char* device_id_c = slow5_hdr_get("device_id", j, sp->header);
                    if(!device_id_c){
                        fprintf(stderr,"No device_id found in %s.", data_path.c_str());
                        exit(EXIT_FAILURE);
                    }
                    std::string device_id = std::string(device_id_c);
                    char* exp_start_time_ms_c = slow5_hdr_get("acquisition_start_time", j, sp->header);
                    if(!exp_start_time_ms_c){
                        exp_start_time_ms_c = slow5_hdr_get("exp_start_time", j, sp->header);
                        if(!exp_start_time_ms_c) {
                            fprintf(stderr, "Neither acquisition_start_time nor exp_start_time found in %s.",
                                    data_path.c_str());
                            exit(EXIT_FAILURE);
                        }
                    }
                    std::string exp_start_time_ms = std::string(exp_start_time_ms_c);
                    char* sample_id_c = slow5_hdr_get("sample_id", j, sp->header);
                    if(!sample_id_c){
                        fprintf(stderr,"No sample_id found in %s.", data_path.c_str());
                        exit(EXIT_FAILURE);
                    }
                    std::string sample_id = std::string(sample_id_c);
                    std::string id = run_id + "_" + model_path;
                    read_groups[id] = ReadGroup{
                            run_id,
                            model_path,
                            flowcell_id,
                            device_id,
                            exp_start_time_ms,
                            sample_id};
                }
                slow5_close(sp);
            }
        }
        spdlog::debug("num read_groups {}", read_groups.size());
        return read_groups;
    }

    bool DataLoader::is_read_data_present(std::string data_path, bool recursive_file_loading) {
        auto check_directory = [&](const auto& iterator_fn) {
            for (const auto& entry : iterator_fn(data_path)) {
                std::string ext = std::filesystem::path(entry).extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(),
                               [](unsigned char c) { return std::tolower(c); });
                if (ext == ".slow5" || ext == ".blow5") {
                    return true;
                }
            }
            return false;
        };

        if (recursive_file_loading) {
            return check_directory([](const auto& path) {
                return std::filesystem::recursive_directory_iterator(path);
            });
        } else {
            if(std::filesystem::is_directory(data_path)){
                return check_directory(
                        [](const auto& path) { return std::filesystem::directory_iterator(path); });
            } else{
                std::string ext = std::filesystem::path(data_path).extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(),
                               [](unsigned char c) { return std::tolower(c); });
                if (ext == ".slow5" || ext == ".blow5") {
                    return true;
                }
            }
        }
        return false;
    }

    uint16_t DataLoader::get_sample_rate(std::string data_path, bool recursive_file_loading) {
        std::optional<uint16_t> sample_rate = std::nullopt;

        auto iterate_directory = [&](const auto &iterator_fn) {
            for (const auto &entry: iterator_fn((data_path))) {
                std::string ext = std::filesystem::path(entry).extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(),
                               [](unsigned char c) { return std::tolower(c); });
                auto file_path = entry.path().string();
                if (ext != ".slow5" and ext != ".blow5") {
                    continue;
                }
                slow5_file_t *sp = slow5_open(file_path.c_str(),"r");
                if(sp==NULL){
                    fprintf(stderr,"Error in opening file\n");
                    exit(EXIT_FAILURE);
                }
                slow5_rec_t *rec = NULL;
                int ret=0;
                std::optional<uint16_t> sample_rate_local = std::nullopt;
                while((ret = slow5_get_next(&rec,sp)) >= 0){
                    sample_rate_local = rec->sampling_rate;
                    if (sample_rate and sample_rate_local!=sample_rate) {
                        throw std::runtime_error("different sample rates found in the files");
                    } else if(!sample_rate){
                        sample_rate = sample_rate_local;
                        break;
                    }
                }
                slow5_rec_free(rec);
                slow5_close(sp);
                if (!sample_rate_local) {
                    throw std::runtime_error("Unable to determine sample rate for data.");
                }
            }
        };

        if (recursive_file_loading) {
            iterate_directory([](const auto &path) {
                return std::filesystem::recursive_directory_iterator(path);
            });
        } else {
            if(!std::filesystem::is_directory(data_path)) {
                std::string ext = std::filesystem::path(data_path).extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(),
                               [](unsigned char c) { return std::tolower(c); });
                slow5_file_t *sp = slow5_open(data_path.c_str(),"r");
                if(sp==NULL){
                    fprintf(stderr,"Error in opening file\n");
                    exit(EXIT_FAILURE);
                }
                slow5_rec_t *rec = NULL;
                int ret=0;
                while((ret = slow5_get_next(&rec,sp)) >= 0){
                    sample_rate = rec->sampling_rate;
                    if (sample_rate) {
                        break;
                    }
                }
                slow5_rec_free(rec);
                slow5_close(sp);
            }else{
                iterate_directory(
                        [](const auto &path) { return std::filesystem::directory_iterator(path); });
            }
        }

        if (sample_rate) {
            spdlog::debug("sample_rate {}", *sample_rate);
            return *sample_rate;
        } else {
            throw std::runtime_error("Unable to determine sample rate for data.");
        }
    }

    void create_read_data(core_t *core, db_t *db, int32_t i) {
        //
        struct slow5_rec *rec = NULL;
        if (slow5_rec_depress_parse(&db->mem_records[i], &db->mem_bytes[i], NULL, &rec, core->fp) != 0) {
            exit(EXIT_FAILURE);
        } else {
            free(db->mem_records[i]);
        }
        auto new_read = std::make_shared<dorado::Read>();

        //
        std::vector<int16_t> tmp(rec->raw_signal, rec->raw_signal + rec->len_raw_signal);
//    std::vector<float> floatTmp(tmp.begin(), tmp.end());
        char* run_id_c = slow5_hdr_get("run_id", rec->read_group, core->fp->header);
        if(!run_id_c){
            fprintf(stderr,"No run_id found in %s.", core->fp->meta.pathname);
            exit(EXIT_FAILURE);
        }
        std::string run_id = std::string(run_id_c);
        char* flow_cell_id_c = slow5_hdr_get("flow_cell_id", rec->read_group, core->fp->header);
        if(!flow_cell_id_c){
            fprintf(stderr,"No flowcell_id found in %s.", core->fp->meta.pathname);
            exit(EXIT_FAILURE);
        }
        std::string flowcell_id = std::string(flow_cell_id_c);

        int ret = 0;
        uint64_t start_time = slow5_aux_get_uint64(rec, "start_time", &ret);
        if (ret != 0) {
            throw std::runtime_error("Error in getting auxiliary attribute 'start_time' from the file.");
        }
        ret = 0;
        uint32_t mux = slow5_aux_get_uint8(rec, "start_mux", &ret);
        if (ret != 0) {
            throw std::runtime_error("Error in getting auxiliary attribute 'start_mux' from the file.");
        }
        ret = 0;
        int32_t read_number = slow5_aux_get_int32(rec, "read_number", &ret);
        if (ret != 0) {
            throw std::runtime_error("Error in getting auxiliary attribute 'read_number' from the file.");
        }
        ret = 0;
        uint64_t len;
        std::string channel_number_str = slow5_aux_get_string(rec, "channel_number", &len, &ret);
        if (ret != 0) {
            throw std::runtime_error("Error in getting auxiliary attribute 'channel_number' from the file.");
        }
        int32_t channel_number = static_cast<int32_t>(std::stol(channel_number_str));

        char* exp_start_time_ms_c = slow5_hdr_get("acquisition_start_time", rec->read_group, core->fp->header);
        if(!exp_start_time_ms_c){
            exp_start_time_ms_c = slow5_hdr_get("exp_start_time", rec->read_group, core->fp->header);
            if(!exp_start_time_ms_c) {
                throw std::runtime_error("Neither acquisition_start_time nor exp_start_time found");
            }
        }

        std::string exp_start_time_ms = std::string(exp_start_time_ms_c);

        auto run_acquisition_start_time_ms = utils::get_unix_time_from_string_timestamp(exp_start_time_ms);
        auto start_time_ms = run_acquisition_start_time_ms + ((start_time * 1000) /(uint64_t)rec->sampling_rate);
        auto start_time_str = utils::get_string_timestamp_from_unix_time(start_time_ms);

//    auto options = torch::TensorOptions().dtype(torch::kFloat32);
        auto options = torch::TensorOptions().dtype(torch::kInt16);
        new_read->raw_data = torch::from_blob(tmp.data(), tmp.size(), options).clone().to(core->m_device_);
        new_read->sample_rate = rec->sampling_rate;
        new_read->run_acquisition_start_time_ms = run_acquisition_start_time_ms;
        new_read->start_time_ms = start_time_ms;
        new_read->scaling = rec->range / rec->digitisation;
        new_read->offset = rec->offset;
        new_read->read_id = std::string(rec->read_id);
        new_read->num_trimmed_samples = 0;
        new_read->attributes.read_number = read_number;
        new_read->attributes.fast5_filename = std::string(core->fp->meta.pathname);
        new_read->attributes.mux = mux;
        new_read->attributes.num_samples = rec->len_raw_signal;
        new_read->attributes.channel_number = channel_number;
        new_read->attributes.start_time = start_time_str;
        new_read->run_id = run_id;
        new_read->start_sample = start_time;
        new_read->end_sample = start_time + rec->len_raw_signal;
        new_read->flowcell_id = flowcell_id;
        new_read->is_duplex = false;
        new_read->digitisation = rec->digitisation;
        new_read->range = rec->range;
//        spdlog::debug("attributes.fast5_filename {}", new_read->attributes.fast5_filename);
//        spdlog::debug("new_read->read_id {}", new_read->read_id);

        //
        db->read_data_ptrs[i] = new_read;
        slow5_rec_free(rec);
    }

    void DataLoader::load_slow5_reads_from_file(const std::string &path) {
        slow5_file_t *sp = slow5_open(path.c_str(), "r");
        if (sp == NULL) {
            fprintf(stderr, "Error in opening file\n");
            exit(EXIT_FAILURE);
        }
        int64_t batch_size = slow5_batchsize;
        int32_t num_threads = slow5_threads;


        while (1) {
            int flag_EOF = 0;
            db_t db = {0};
            db.mem_records = (char **) malloc(batch_size * sizeof *db.mem_records);
            db.mem_bytes = (size_t *) malloc(batch_size * sizeof *db.mem_bytes);

            int64_t record_count = 0;
            size_t bytes;
            char *mem;

            while (record_count < batch_size) {
                if (!(mem = (char *) slow5_get_next_mem(&bytes, sp))) {
                    if (slow5_errno != SLOW5_ERR_EOF) {
                        throw std::runtime_error("Error in slow5_get_next_mem.");
                    } else { //EOF file reached
                        flag_EOF = 1;
                        break;
                    }
                } else {
                    db.mem_records[record_count] = mem;
                    db.mem_bytes[record_count] = bytes;
                    record_count++;
                }
            }

            // Setup multithreading structures
            core_t core;
            core.num_thread = (num_threads > record_count) ? record_count : num_threads;
            if (record_count == 0) {
                core.num_thread = 1;
            }
            core.fp = sp;
            core.m_device_ = m_device;

            db.n_batch = record_count;
            db.read_data_ptrs = std::vector<std::shared_ptr<Read>>(record_count);

            work_db(&core, &db, create_read_data);

            for (int64_t i = 0; i < record_count; i++) {
                if (!m_allowed_read_ids ||
                    (m_allowed_read_ids->find(db.read_data_ptrs[i]->read_id) != m_allowed_read_ids->end())) {
                    m_pipeline.push_message(std::move(db.read_data_ptrs[i]));
                    m_loaded_read_count++;
                }
            }
            // Free everything
            free(db.mem_bytes);
            free(db.mem_records);

            if (flag_EOF == 1) {
                break;
            }
        }
    }
    void DataLoader::load_slow5_reads_from_file_by_read_ids(const std::string &path, const std::vector<ReadID>& read_ids) {
        spdlog::debug("{}", __func__);
        std::unordered_set<std::string> read_ids_map;
        for(int i=0; i<read_ids.size(); i++){
            std::string rid(read_ids[i].data(), read_ids[i].data()+POD5_READ_ID_SIZE );
            read_ids_map.insert(rid);
        }

        slow5_file_t *sp = slow5_open(path.c_str(), "r");
        if (sp == NULL) {
            fprintf(stderr, "Error in opening file\n");
            exit(EXIT_FAILURE);
        }
        int64_t batch_size = slow5_batchsize;
        int32_t num_threads = slow5_threads;


        while (1) {
            int flag_EOF = 0;
            db_t db = {0};
            db.mem_records = (char **) malloc(batch_size * sizeof *db.mem_records);
            db.mem_bytes = (size_t *) malloc(batch_size * sizeof *db.mem_bytes);

            int64_t record_count = 0;
            size_t bytes;
            char *mem;

            while (record_count < batch_size) {
                if (!(mem = (char *) slow5_get_next_mem(&bytes, sp))) {
                    if (slow5_errno != SLOW5_ERR_EOF) {
                        throw std::runtime_error("Error in slow5_get_next_mem.");
                    } else { //EOF file reached
                        flag_EOF = 1;
                        break;
                    }
                } else {
                    db.mem_records[record_count] = mem;
                    db.mem_bytes[record_count] = bytes;
                    record_count++;
                }
            }

            // Setup multithreading structures
            core_t core;
            core.num_thread = (num_threads > record_count) ? record_count : num_threads;
            if (record_count == 0) {
                core.num_thread = 1;
            }
            core.fp = sp;
            core.m_device_ = m_device;

            db.n_batch = record_count;
            db.read_data_ptrs = std::vector<std::shared_ptr<Read>>(record_count);

            work_db(&core, &db, create_read_data);

            for (int64_t i = 0; i < record_count; i++) {
                if(read_ids_map.find(db.read_data_ptrs[i]->read_id) != read_ids_map.end()){
                    if (!m_allowed_read_ids ||
                        (m_allowed_read_ids->find(db.read_data_ptrs[i]->read_id) != m_allowed_read_ids->end())) {
                        m_pipeline.push_message(std::move(db.read_data_ptrs[i]));
                        m_loaded_read_count++;
                    }
                }
            }
            // Free everything
            free(db.mem_bytes);
            free(db.mem_records);

            if (flag_EOF == 1) {
                break;
            }
        }
    }


    DataLoader::DataLoader(Pipeline & pipeline,
    const std::string &device,
    size_t num_worker_threads,
    size_t max_reads,
    std::optional<std::unordered_set<std::string>> read_list,
    std::unordered_set<std::string> read_ignore_list,
    int32_t slow5_threads_,
    int64_t slow5_batchsize_)
    : m_pipeline(pipeline),
            m_device(device),
            m_num_worker_threads(num_worker_threads),
            m_allowed_read_ids(std::move(read_list)),
            m_ignored_read_ids(std::move(read_ignore_list))
    {
        slow5_threads = slow5_threads_;
        slow5_batchsize = slow5_batchsize_;
        m_max_reads = max_reads == 0 ? std::numeric_limits<decltype(m_max_reads)>::max() : max_reads;
        assert(m_num_worker_threads > 0);
        static std::once_flag vbz_init_flag;
#ifdef USE_FAST5
        std::call_once(vbz_init_flag, vbz_register);
#endif
    }

    stats::NamedStats DataLoader::sample_stats() const {
        return stats::NamedStats{{"loaded_read_count", static_cast<double>(m_loaded_read_count)}};
    }
}  // namespace dorado
#endif