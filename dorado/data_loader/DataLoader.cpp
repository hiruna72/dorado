#include "DataLoader.h"

#include "models/kits.h"
#include "read_pipeline/ReadPipeline.h"
#include "read_pipeline/messages.h"
#include "utils/PostCondition.h"
#include "utils/thread_naming.h"
#include "utils/time_utils.h"
#include "utils/types.h"
#include "vbz_plugin_user_utils.h"

#include <ATen/Functions.h>
#include <cxxpool.h>
#include <highfive/H5Easy.hpp>
#include <highfive/H5File.hpp>
#include <pod5_format/c_api.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>
#include <iostream>

#include "slow5/slow5.h"
#include "slow5_extra.h"
#include "slow5_thread.h"
#include <slow5/slow5_mt.h>

namespace dorado {

namespace {

/**
 * @brief Fetches directory entries from a specified path.
 *
 * This function fetches all directory entries from the specified path. If the path is not a directory,
 * it will return a vector containing a single entry representing the specified file.
 * It can operate in two modes: recursive and non-recursive. In recursive mode, it fetches entries from
 * all subdirectories recursively. In non-recursive mode, it only fetches entries from the top-level directory.
 *
 * @param path The path from which to fetch the directory entries. It can be a path to a file or a directory.
 * @param recursive A boolean flag indicating whether to operate in recursive mode.
 *                  True for recursive mode, false for non-recursive mode.
 * @return A vector of directory entries fetched from the specified path.
 */
auto fetch_directory_entries(const std::filesystem::path& path, bool recursive) {
    using DirEntry = std::filesystem::directory_entry;

    std::vector<DirEntry> entries;

    if (std::filesystem::is_directory(path)) {
        if (recursive) {
            for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
                entries.push_back(entry);
            }
        } else {
            for (const auto& entry : std::filesystem::directory_iterator(path)) {
                entries.push_back(entry);
            }
        }
    } else {
        entries.push_back(DirEntry(path));
    }

    return entries;
}

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
}

std::string get_string_attribute(const HighFive::Group& group, const std::string& attr_name) {
    std::string attribute_string;
    if (group.hasAttribute(attr_name)) {
        HighFive::Attribute attribute = group.getAttribute(attr_name);
        string_reader(attribute, attribute_string);
    }
    return attribute_string;
}

std::vector<std::filesystem::directory_entry> filter_fast5_for_mixed_datasets(
        const std::vector<std::filesystem::directory_entry>& files) {
    std::vector<std::filesystem::directory_entry> pod5_entries;
    std::vector<std::filesystem::directory_entry> fast5_entries;

    bool issued_fast5_warn = false;

    for (const auto& entry : files) {
        auto entry_path = std::filesystem::path(entry);
        std::string ext = entry_path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (ext == ".fast5") {
            if (!issued_fast5_warn) {
                spdlog::warn(
                        "Deprecation Warning: FAST5 support in Dorado will be dropped in an "
                        "upcoming "
                        "release. "
                        "FAST5 loading is unoptimized and will result in poor performance. "
                        "Please convert your dataset to POD5: "
                        "https://pod5-file-format.readthedocs.io/en/latest/docs/"
                        "tools.html#pod5-convert-fast5");
                issued_fast5_warn = true;
            }

            fast5_entries.push_back(entry);
        } else if (ext == ".pod5") {
            pod5_entries.push_back(entry);
        } else if (ext == ".slow5" || ext == ".blow5") {
            pod5_entries.push_back(entry);
        }
    }

    if (pod5_entries.empty()) {
        return fast5_entries;
    } else if (!pod5_entries.empty() && !fast5_entries.empty()) {
        for (const auto& f5 : fast5_entries) {
            spdlog::warn(
                    "Data folder contains both POD5 and FAST5 files. Please basecall "
                    "FAST5 separately. Skipping FAST5 "
                    "file from {}.",
                    f5.path().string());
        }
    }
    return pod5_entries;
}

// Parses pod5 run_info data into a ChemistryKey which is used to lookup the sequencing chemistry
models::ChemistryKey get_chemistry_key(const RunInfoDictData_t* const run_info_data) {
    const auto fc = models::flowcell_code(run_info_data->flow_cell_product_code);
    const auto kit = models::kit_code(run_info_data->sequencing_kit);
    const auto key = models::ChemistryKey(fc, kit, run_info_data->sample_rate);
    return key;
}

/*
void dump_to_stderr(const SimplexReadPtr& simplex_read_ptr) {
    if (!simplex_read_ptr) {
        std::cerr << "Error: SimplexRead pointer is null.\n";
        return;
    }

    const auto& rc = simplex_read_ptr->read_common;

    // Dump ReadCommon members
    std::cerr << "Read ID: " << rc.read_id << '\n'
              << "Sequence: " << rc.seq << '\n'
              << "Qstring: " << rc.qstring << '\n'
              << "Run ID: " << rc.run_id << '\n'
              << "Flow Cell Product Code: " << rc.flow_cell_product_code << '\n'
              << "Flowcell ID: " << rc.flowcell_id << '\n'
              << "Position ID: " << rc.position_id << '\n'
              << "Experiment ID: " << rc.experiment_id << '\n'
              << "Model Name: " << rc.model_name << '\n'
              << "Model Stride: " << rc.model_stride << '\n'
              << "Start Time (ms): " << rc.start_time_ms << '\n'
              << "Pre-trim Sequence Length: " << rc.pre_trim_seq_length << '\n'
              << "Adapter Trim Interval: (" << rc.adapter_trim_interval.first << ", " << rc.adapter_trim_interval.second << ") " << '\n'
              << "Barcode Trim Interval: (" << rc.barcode_trim_interval.first << ", " << rc.barcode_trim_interval.second << ") " << '\n'
              << "Read Tag: " << rc.read_tag << '\n'
              << "Mean Qscore Start Pos: " << rc.mean_qscore_start_pos << '\n'
              << "Barcode: " << rc.barcode << '\n'
              << "Sample Rate: " << rc.sample_rate << '\n'
              << "Shift: " << rc.shift << '\n'
              << "Scale: " << rc.scale << '\n'
              << "Scaling Method: " << rc.scaling_method << '\n'
              << "Parent Read ID: " << rc.parent_read_id << '\n'
              << "Num Trimmed Samples: " << rc.num_trimmed_samples << '\n'
              << "Is Duplex: " << rc.is_duplex << '\n'
              << "Is RNA Model: " << rc.is_rna_model << '\n'
              << "RNA Poly Tail Length: " << rc.rna_poly_tail_length << '\n'
              << "RNA Adapter End Signal Pos: " << rc.rna_adapter_end_signal_pos << '\n'
              << "Subread ID: " << rc.subread_id << '\n'
              << "Split Count: " << rc.split_count << '\n'
              << "Split Point: " << rc.split_point << '\n'
              << "Model Q Bias: " << rc.model_q_bias << '\n'
              << "Model Q Scale: " << rc.model_q_scale << std::endl;


    // Dump moves, base_mod_probs, and alignment results sizes
    std::cerr << "Moves Size: " << rc.moves.size() << '\n'
              << "Base Mod Probs Size: " << rc.base_mod_probs.size() << '\n'
              << "Alignment Results Size: " << rc.alignment_results.size() << std::endl;

    // Dump SimplexRead members
    std::cerr << "Digitisation: " << simplex_read_ptr->digitisation << '\n'
              << "Range: " << simplex_read_ptr->range << '\n'
              << "Offset: " << simplex_read_ptr->offset << '\n'
              << "Scaling: " << simplex_read_ptr->scaling << '\n'
              << "Start Sample: " << simplex_read_ptr->start_sample << '\n'
              << "End Sample: " << simplex_read_ptr->end_sample << '\n'
              << "Run Acquisition Start Time (ms): " << simplex_read_ptr->run_acquisition_start_time_ms << '\n'
              << "Num Duplex Candidate Pairs: " << simplex_read_ptr->num_duplex_candidate_pairs.load() << '\n'
              << "Is Duplex Parent: " << simplex_read_ptr->is_duplex_parent.load() << '\n'
              << "Previous Read: " << simplex_read_ptr->prev_read << '\n'
              << "Next Read: " << simplex_read_ptr->next_read << std::endl;
    
        // Dump Attributes members
    const auto& attr = rc.attributes;
    std::cerr << "Mux: " << attr.mux << '\n'
              << "Read Number: " << attr.read_number << '\n'
              << "Channel Number: " << attr.channel_number << '\n'
              << "Start Time: " << attr.start_time << '\n'
              << "Fast5 Filename: " << attr.fast5_filename << '\n'
              << "Num Samples: " << attr.num_samples << '\n'
              << "Is End Reason Mux Change: " << (attr.is_end_reason_mux_change ? "true" : "false") << std::endl;

    
    // Print raw_data tensor dimensions
    if (rc.raw_data.defined()) {  // Check if the tensor is initialized
        std::cerr << "Raw Data Dimensions: ";
        for (const auto& dim : rc.raw_data.sizes()) {
            std::cerr << dim << " ";
        }
        std::cerr << '\n';
    } else {
        std::cerr << "Raw Data: Tensor is undefined.\n";
    }

    // Print Chemistry and RapidChemistry enum as integers
    std::cerr << "Chemistry Type: " << static_cast<int>(rc.chemistry) << '\n'
              << "Rapid Chemistry Type: " << static_cast<int>(rc.rapid_chemistry) << std::endl;

}
*/


SimplexReadPtr process_pod5_thread_fn(
        size_t row,
        Pod5ReadRecordBatch* batch,
        Pod5FileReader* file,
        const std::string& path,
        const std::unordered_map<int, std::vector<DataLoader::ReadSortInfo>>& reads_by_channel,
        const std::unordered_map<std::string, size_t>& read_id_to_index) {
    utils::set_thread_name("process_pod5");
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
    if (pod5_format_read_id(read_data.read_id, read_id_tmp) != POD5_OK) {
        spdlog::error("Failed to format read id");
    }
    std::string read_id_str(read_id_tmp);

    auto options = at::TensorOptions().dtype(at::kShort);
    auto samples = at::empty(read_data.num_samples, options);

    if (pod5_get_read_complete_signal(file, batch, row, read_data.num_samples,
                                      samples.data_ptr<int16_t>()) != POD5_OK) {
        spdlog::error("Failed to get read {} signal: {}", row, pod5_get_error_string());
    }

    auto new_read = std::make_unique<SimplexRead>();
    new_read->read_common.raw_data = samples;
    new_read->read_common.sample_rate = run_sample_rate;

    auto start_time_ms = run_acquisition_start_time_ms +
                         ((read_data.start_sample * 1000) /
                          (uint64_t)run_sample_rate);  // TODO check if this cast is needed
    auto start_time = utils::get_string_timestamp_from_unix_time(start_time_ms);
    new_read->run_acquisition_start_time_ms = run_acquisition_start_time_ms;
    new_read->read_common.start_time_ms = start_time_ms;
    new_read->scaling = read_data.calibration_scale;
    new_read->offset = read_data.calibration_offset;
    new_read->read_common.read_id = std::move(read_id_str);
    new_read->read_common.num_trimmed_samples = 0;
    new_read->read_common.attributes.read_number = read_data.read_number;
    new_read->read_common.attributes.fast5_filename =
            std::filesystem::path(path.c_str()).filename().string();
    new_read->read_common.attributes.mux = read_data.well;
    new_read->read_common.attributes.num_samples = read_data.num_samples;
    new_read->read_common.attributes.channel_number = read_data.channel;
    new_read->read_common.attributes.start_time = start_time;
    new_read->read_common.run_id = run_info_data->acquisition_id;
    new_read->start_sample = read_data.start_sample;
    new_read->end_sample = read_data.start_sample + read_data.num_samples;
    new_read->read_common.flowcell_id = run_info_data->flow_cell_id;
    new_read->read_common.flow_cell_product_code = run_info_data->flow_cell_product_code;
    new_read->read_common.position_id = run_info_data->sequencer_position;
    new_read->read_common.experiment_id = run_info_data->experiment_name;
    new_read->read_common.is_duplex = false;

    // Get the condition_info from the run_info_data to determine if the sequencing kit
    // used has a rapid adapter and which one.
    const auto condition_info = models::ConditionInfo(get_chemistry_key(run_info_data));
    new_read->read_common.rapid_chemistry = condition_info.rapid_chemistry();
    new_read->read_common.chemistry = condition_info.chemistry();

    pod5_end_reason_t end_reason_value{POD5_END_REASON_UNKNOWN};
    char end_reason_string_value[200];
    size_t end_reason_string_value_size = sizeof(end_reason_string_value);

    pod5_error_t pod5_ret =
            pod5_get_end_reason(batch, read_data.end_reason, &end_reason_value,
                                end_reason_string_value, &end_reason_string_value_size);
    if (pod5_ret != POD5_OK) {
        spdlog::error("Failed to get read end_reason {} {}", row, pod5_get_error_string());
    } else if (end_reason_value == POD5_END_REASON_UNBLOCK_MUX_CHANGE ||
               end_reason_value == POD5_END_REASON_MUX_CHANGE) {
        new_read->read_common.attributes.is_end_reason_mux_change = true;
    }

    // Determine the time sorted predecessor of the read
    // if that information is available (primarily used for offline
    // duplex runs).
    if (reads_by_channel.find(read_data.channel) != reads_by_channel.end()) {
        auto& read_id = new_read->read_common.read_id;
        const auto& v = reads_by_channel.at(read_data.channel);
        auto read_id_iter = v.begin() + read_id_to_index.at(read_id);

        if (read_id_iter != v.begin()) {
            new_read->prev_read = std::prev(read_id_iter)->read_id;
        }
        if (std::next(read_id_iter) != v.end()) {
            new_read->next_read = std::next(read_id_iter)->read_id;
        }
    }

    if (pod5_free_run_info(run_info_data) != POD5_OK) {
        spdlog::error("Failed to free run info");
    }

    // dump_to_stderr(new_read);

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
    if (pod5_format_read_id(read_data.read_id, read_id_tmp) != POD5_OK) {
        spdlog::error("Failed to format read id");
    }

    std::string read_id_str(read_id_tmp);
    bool read_in_ignore_list = ignored_read_ids.find(read_id_str) != ignored_read_ids.end();
    bool read_in_read_list =
            !allowed_read_ids || (allowed_read_ids->find(read_id_str) != allowed_read_ids->end());
    if (!read_in_ignore_list && read_in_read_list) {
        return true;
    }
    return false;
}

}  // namespace

void Pod5Destructor::operator()(Pod5FileReader_t* pod5) { pod5_close_and_free_reader(pod5); }

void DataLoader::load_reads(const std::filesystem::path& path,
                            bool recursive_file_loading,
                            ReadOrder traversal_order) {
    if (!std::filesystem::exists(path)) {
        spdlog::error("Requested input path {} does not exist!", path.string());
        return;
    }

    auto iterate_directory = [&](const auto& iterator) {
        switch (traversal_order) {
        case ReadOrder::BY_CHANNEL:{
            std::string slow5_file_path = "";
            int slow5_file_count = 0;
            slow5_file_t *sp = NULL;
            for (const auto& entry : iterator) {
                std::string ext = std::filesystem::path(entry).extension().string();
                if (ext == ".slow5" || ext == ".blow5") {
                    slow5_file_path = entry.path();
                    slow5_file_count++;
                }
            }

            if(slow5_file_count > 1){
                throw std::runtime_error("Please provide a single BLOW5 file path for duplex calling. Multiple files are not supported");
            } 
            if(slow5_file_count == 1){
                sp = slow5_open(slow5_file_path.c_str(), "r");
                if (sp == NULL) {
                    throw std::runtime_error("Error in opening SLOW5/BLOW5 file");
                }
                int ret = slow5_idx_load(sp);
                if(ret<0){
                    throw std::runtime_error("Error in loading index for SLOW5/BLOW5 file");
                }
            }

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
                if (m_reads_by_channel.find(channel) != m_reads_by_channel.end()) {
                    // Sort the read ids within a channel by its mux
                    // and start time.
                    spdlog::debug("Sort channel {}", channel);
                    auto& reads = m_reads_by_channel.at(channel);
                    std::sort(reads.begin(), reads.end(), [](ReadSortInfo& a, ReadSortInfo& b) {
                        if (a.mux != b.mux) {
                            return a.mux < b.mux;
                        } else {
                            return a.read_number < b.read_number;
                        }
                    });
                    // Once sorted, create a hash table from read id
                    // to index in the sorted list to quickly fetch the
                    // read location and its neighbors.
                    for (size_t i = 0; i < reads.size(); i++) {
                        m_read_id_to_index[reads[i].read_id] = i;
                    }
                    spdlog::debug("Sorted channel {}", channel);
                }
                if(slow5_file_count == 1){
                    auto& channel_to_read_ids =
                            m_file_channel_read_order_map.at(slow5_file_path);
                    auto& read_ids = channel_to_read_ids[channel];
                    if (!read_ids.empty()) {
                        load_slow5_reads_from_file_by_read_ids(sp, read_ids);
                    }
                }else{
                    for (const auto& entry : iterator) {
                        if (m_loaded_read_count == m_max_reads) {
                            break;
                        }
                        auto entry_path = std::filesystem::path(entry);
                        std::string ext = entry_path.extension().string();
                        std::transform(ext.begin(), ext.end(), ext.begin(),
                                    [](unsigned char c) { return std::tolower(c); });
                        if (ext == ".fast5") {
                            throw std::runtime_error(
                                    "Traversing reads by channel is only available for POD5. "
                                    "Encountered FAST5 at " +
                                    entry_path.string());
                        } else if (ext == ".pod5") {
                            auto& channel_to_read_ids =
                                    m_file_channel_read_order_map.at(entry_path.string());
                            auto& read_ids = channel_to_read_ids[channel];
                            if (!read_ids.empty()) {
                                load_pod5_reads_from_file_by_read_ids(entry_path.string(), read_ids);
                            }
                        }
                    }
                }
                
                // Erase sorted list as it's not needed anymore.
                m_reads_by_channel.erase(channel);
            }
            if(slow5_file_count == 1){
                slow5_idx_unload(sp);
                slow5_close(sp);
            }
            break;
        }
        case ReadOrder::UNRESTRICTED:
            for (const auto& entry : iterator) {
                if (m_loaded_read_count == m_max_reads) {
                    break;
                }
                spdlog::debug("Load reads from file {}", entry.path().string());
                std::string ext = std::filesystem::path(entry).extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(),
                               [](unsigned char c) { return std::tolower(c); });
                if (ext == ".fast5") {
                    load_fast5_reads_from_file(entry.path().string());
                } else if (ext == ".pod5") {
                    load_pod5_reads_from_file(entry.path().string());
                } else if (ext == ".slow5" || ext == ".blow5") {
                    load_slow5_reads_from_file(entry.path().string());
                }
            }
            break;
        default:
            throw std::runtime_error("Unsupported traversal order detected: " +
                                     dorado::to_string(traversal_order));
        }
    };

    auto filtered_entries =
            filter_fast5_for_mixed_datasets(fetch_directory_entries(path, recursive_file_loading));
    iterate_directory(filtered_entries);
}

int DataLoader::get_num_reads(const std::filesystem::path& data_path,
                              std::optional<std::unordered_set<std::string>> read_list,
                              const std::unordered_set<std::string>& ignore_read_list,
                              bool recursive_file_loading) {
    if(read_list){
        spdlog::debug("> args to get_num_reads: {}", data_path.string());
        spdlog::debug("> args to get_num_reads: {}",ignore_read_list.size());
        spdlog::debug("> args to get_num_reads: {}",recursive_file_loading);
    }
    
    return 0;

}

void DataLoader::load_read_channels(const std::filesystem::path& data_path,
                                    bool recursive_file_loading) {
    auto iterate_directory = [&](const auto& iterator) {
        for (const auto& entry : iterator) {
            auto file_path = std::filesystem::path(entry);
            std::string ext = file_path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            if (ext == ".pod5") {
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
                        channel_to_read_id[channel].push_back(read_id);

                        char read_id_tmp[POD5_READ_ID_LEN];
                        if (pod5_format_read_id(read_data.read_id, read_id_tmp) != POD5_OK) {
                            spdlog::error("Failed to format read id");
                        }
                        std::string rid(read_id_tmp);
                        m_reads_by_channel[channel].push_back(
                                {rid, read_data.well, read_data.read_number});
                    }

                    if (pod5_free_read_batch(batch) != POD5_OK) {
                        spdlog::error("Failed to release batch");
                    }
                }
                if (pod5_close_and_free_reader(file) != POD5_OK) {
                    spdlog::error("Failed to close and free POD5 reader");
                }
            } else if (ext == ".slow5" || ext == ".blow5") {
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
                int ret_batch=0;
                int ret=0;

                slow5_mt_t *mt = slow5_init_mt(slow5_threads,sp);
                slow5_batch_t *read_batch = slow5_init_batch(slow5_batchsize);

                while((ret_batch = slow5_get_next_batch(mt,read_batch,slow5_batchsize)) > 0){
                    rec = read_batch->slow5_rec;
                    for(int i=0;i<ret_batch;i++){
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
                            uint8_t  arr[16] = {0};
                            ret = sscanf(rec[i]->read_id, "%2hhx%2hhx%2hhx%2hhx-%2hhx%hhx-%2hhx%2hhx-%2hhx%2hhx-%2hhx%2hhx%2hhx%2hhx%2hhx%2hhx",
                                            &arr[0], &arr[1], &arr[2], &arr[3], &arr[4], &arr[5], &arr[6], &arr[7],
                                            &arr[8], &arr[9], &arr[10], &arr[11], &arr[12], &arr[13], &arr[14], &arr[15]);
                            if(ret !=16){
                                fprintf(stderr,"Parsing uuid failed. Return val %d\n",ret);
                                exit(1);
                            }

                            ReadID read_id;
                            std::memcpy(read_id.data(), arr, POD5_READ_ID_SIZE);
                            channel_to_read_id[channel].push_back(std::move(read_id));

                            std::string rid(rec[i]->read_id);
                            ret = 0;
                            uint32_t mux = slow5_aux_get_uint8(rec[i], "start_mux", &ret);
                            if (ret != 0) {
                                throw std::runtime_error("Error in getting auxiliary attribute 'start_mux' from the file.");
                            }
                            ret = 0;
                            int32_t read_number = slow5_aux_get_int32(rec[i], "read_number", &ret);
                            if (ret != 0) {
                                throw std::runtime_error("Error in getting auxiliary attribute 'read_number' from the file.");
                            }
                            m_reads_by_channel[channel].push_back(
                                    {rid, (int32_t)mux, (uint32_t)read_number});

                        }
                    }
                    if(ret_batch<slow5_batchsize){ //this indicates nothing left to read //need to handle errors
                        break;
                    }
                }
                slow5_free_batch(read_batch);
                slow5_free_mt(mt);
                slow5_close(sp);
            }
        }
    };

    iterate_directory(fetch_directory_entries(data_path, recursive_file_loading));
}

std::unordered_map<std::string, ReadGroup> DataLoader::load_read_groups(
        const std::filesystem::path& data_path,
        std::string model_name,
        std::string modbase_model_names,
        bool recursive_file_loading) {
    std::unordered_map<std::string, ReadGroup> read_groups;

    auto iterate_directory = [&](const auto& iterator) {
        for (const auto& entry : iterator) {
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
                        std::string position_id = run_info_data->sequencer_position;
                        std::string experiment_id = run_info_data->experiment_name;

                        if (pod5_free_run_info(run_info_data) != POD5_OK) {
                            spdlog::error("Failed to free run info");
                        }

                        std::string id = std::string(run_id).append("_").append(model_name);
                        read_groups[id] = ReadGroup{
                                std::move(run_id),
                                model_name,
                                modbase_model_names,
                                std::move(flowcell_id),
                                std::move(device_id),
                                utils::get_string_timestamp_from_unix_time(exp_start_time_ms),
                                std::move(sample_id),
                                std::move(position_id),
                                std::move(experiment_id),
                        };
                    }
                    if (pod5_close_and_free_reader(file) != POD5_OK) {
                        spdlog::error("Failed to close and free POD5 reader");
                    }
                }
            } else if (ext == ".slow5" || ext == ".blow5") {
                slow5_file_t *sp = slow5_open(entry.path().string().c_str(),"r");
                if(sp==NULL){
                    throw std::runtime_error("Error in opening file");
                }
                int64_t read_group_count = sp->header->num_read_groups;
                for(int64_t j=0; j<read_group_count; j++){
                    char* run_id_c = slow5_hdr_get("run_id", j, sp->header);
                    std::string run_id = "";
                    if(!run_id_c){
                        fprintf(stderr,"No run_id found in %s. (%s)\n", entry.path().string().c_str(), "DataLoader::load_read_groups");
                        throw std::runtime_error("No run_id found");
                    }
                    run_id = std::string(run_id_c);

                    char* flow_cell_id_c = slow5_hdr_get("flow_cell_id", j, sp->header);
                    std::string flow_cell_id = "";
                    if(!flow_cell_id_c){
                        fprintf(stderr,"No flowcell_id found in %s. (%s)\n", entry.path().string().c_str(), "DataLoader::load_read_groups");
                        throw std::runtime_error("No flowcell_id found");
                    }
                    flow_cell_id = std::string(flow_cell_id_c);
                    
                    char* device_id_c = slow5_hdr_get("host_product_serial_number", j, sp->header); //pod5 has started to use confusing variable name. device_id and sequencer_position refer to the same in the blow5 and pod5 file specifications.
                    std::string device_id = "";
                    if(!device_id_c){
                            device_id_c = slow5_hdr_get("system_name", j, sp->header);
                            if(!device_id_c) {
                                throw std::runtime_error("Niether host_product_serial_number nor system_name found");
                            }
                    }
                    device_id = std::string(device_id_c);

                    char* experiment_id_c = slow5_hdr_get("experiment_name", j, sp->header);
                    std::string experiment_id = "";
                    if(experiment_id_c){
                        experiment_id = std::string(experiment_id_c);
                    }

                    char* exp_start_time_ms_c = slow5_hdr_get("acquisition_start_time", j, sp->header);
                    std::string exp_start_time_ms = "";
                    if(!exp_start_time_ms_c){
                        exp_start_time_ms_c = slow5_hdr_get("exp_start_time", j, sp->header);
                        if(!exp_start_time_ms_c) {
                            throw std::runtime_error("Neither acquisition_start_time nor exp_start_time found");
                        }
                    }
                    exp_start_time_ms = std::string(exp_start_time_ms_c);
                    // std::cerr << "exp_start_time_ms: " << exp_start_time_ms << std::endl;
                    
                    char* sample_id_c = slow5_hdr_get("sample_id", j, sp->header);
                    std::string sample_id = "";
                    if(!sample_id_c){
                        fprintf(stderr,"No sample_id found in %s. (%s)\n", entry.path().string().c_str(), "DataLoader::load_read_groups");
                        throw std::runtime_error("No sample_id found");
                    }
                    sample_id = std::string(sample_id_c);
                    

                    char* position_id_c = slow5_hdr_get("sequencer_position", j, sp->header);
                    std::string position_id = "";
                    if(position_id_c){
                        position_id = std::string(position_id_c);
                    }

                    std::string id = std::string(run_id).append("_").append(model_name);
                    read_groups[id] = ReadGroup{
                            std::move(run_id),
                            model_name,
                            modbase_model_names,
                            std::move(flow_cell_id),
                            std::move(device_id),
                            utils::get_string_timestamp_from_unix_time(utils::get_unix_time_from_string_timestamp(exp_start_time_ms)),
                            std::move(sample_id),
                            std::move(position_id),
                            std::move(experiment_id),
                    };
                }
                slow5_close(sp);
            }
        }
    };

    iterate_directory(fetch_directory_entries(data_path, recursive_file_loading));

    return read_groups;
}

bool DataLoader::is_read_data_present(const std::filesystem::path& data_path,
                                      bool recursive_file_loading) {
    auto check_directory = [&](const auto& iterator) {
        for (const auto& entry : iterator) {
            std::string ext = std::filesystem::path(entry).extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            if (ext == ".pod5" || ext == ".fast5" || ext == ".slow5" || ext == ".blow5") {
                return true;
            }
        }
        return false;
    };

    return check_directory(fetch_directory_entries(data_path, recursive_file_loading));
}

uint16_t DataLoader::get_sample_rate(const std::filesystem::path& data_path,
                                     bool recursive_file_loading) {
    std::optional<uint16_t> sample_rate = std::nullopt;

    auto iterate_directory = [&](const auto& iterator) {
        for (const auto& entry : iterator) {
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
                    auto free_pod5 = [&]() {
                        if (pod5_close_and_free_reader(file) != POD5_OK) {
                            spdlog::error("Failed to close and free POD5 reader for file {}",
                                          file_path.c_str());
                        }
                    };

                    auto post = utils::PostCondition(free_pod5);

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
            } else if (ext == ".fast5") {
                H5Easy::File file(entry.path().string(), H5Easy::File::ReadOnly);
                HighFive::Group reads = file.getGroup("/");
                int num_reads = int(reads.getNumberObjects());

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
            } else if (ext == ".slow5" || ext == ".blow5") {
                slow5_file_t *sp = slow5_open(file_path.c_str(),"r");
                if(sp==NULL){
                    fprintf(stderr,"Error in opening file\n");
                    exit(EXIT_FAILURE);
                }
                slow5_rec_t *rec = NULL;
                int ret=0;
                while((ret = slow5_get_next(&rec,sp)) >= 0){
                    sample_rate = rec->sampling_rate;
                    break;
                }
                slow5_rec_free(rec);
                slow5_close(sp);
            }

            // Break out of loop if sample rate is found.
            if (sample_rate) {
                break;
            }
        }
    };

    iterate_directory(fetch_directory_entries(data_path, recursive_file_loading));

    if (sample_rate) {
        return *sample_rate;
    } else {
        throw std::runtime_error("Unable to determine sample rate for data.");
    }
}

std::set<models::ChemistryKey> DataLoader::get_sequencing_chemistries(
        const std::filesystem::path& data_path,
        bool recursive_file_loading) {
    std::set<models::ChemistryKey> chemistries;

    auto iterate_directory = [&](const auto& iterator) {
        for (const auto& entry : iterator) {
            std::string ext = std::filesystem::path(entry).extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            auto file_path = entry.path().string();
            if (ext == ".fast5") {
                throw std::runtime_error("Cannot automate model selection using fast5 files");
                return;
            }

            if (ext == ".pod5") {
                pod5_init();
                // Open the file ready for walking:
                Pod5FileReader_t* file = pod5_open_file(file_path.c_str());

                if (!file) {
                    spdlog::error("Failed to open file {}: {}", file_path.c_str(),
                                  pod5_get_error_string());
                } else {
                    auto free_pod5 = [&]() {
                        if (pod5_close_and_free_reader(file) != POD5_OK) {
                            spdlog::error("Failed to close and free POD5 reader for file {}",
                                          file_path.c_str());
                        }
                    };

                    auto post = utils::PostCondition(free_pod5);

                    // First get the run info count
                    run_info_index_t run_info_count;
                    if (pod5_get_file_run_info_count(file, &run_info_count) != POD5_OK) {
                        spdlog::error("Failed to fetch POD5 run info count for file {} : {}",
                                      file_path.c_str(), pod5_get_error_string());

                        continue;
                    }

                    for (run_info_index_t ri_idx = 0; ri_idx < run_info_count; ri_idx++) {
                        RunInfoDictData_t* run_info_data;
                        if (pod5_get_file_run_info(file, ri_idx, &run_info_data) != POD5_OK) {
                            spdlog::error(
                                    "Failed to fetch POD5 run info dict for file {} and run info "
                                    "index {}: {}",
                                    file_path.c_str(), ri_idx, pod5_get_error_string());
                        } else {
                            const auto chemistry_key = get_chemistry_key(run_info_data);
                            spdlog::trace("POD5: {} {}", file_path.c_str(), to_string(chemistry_key));
                            chemistries.insert(chemistry_key);
                        }
                        if (pod5_free_run_info(run_info_data) != POD5_OK) {
                            spdlog::error(
                                    "Failed to free POD5 run info for file {} and run info index: "
                                    "{}",
                                    file_path.c_str(), ri_idx);
                        }
                    }
                };
            } else if (ext == ".slow5" || ext == ".blow5") {
                slow5_file_t *sp = slow5_open(file_path.c_str(),"r");
                if(sp==NULL){
                    fprintf(stderr,"Error in opening file\n");
                    exit(EXIT_FAILURE);
                }
                int64_t read_group_count = sp->header->num_read_groups;
                for(int64_t j=0; j<read_group_count; j++){
                    char* flow_cell_product_code_c = slow5_hdr_get("flow_cell_product_code", j, sp->header);
                    std::string flow_cell_product_code = "";
                    if(!flow_cell_product_code_c){
                        fprintf(stderr,"No flow_cell_product_code found in %s. (%s)\n", file_path.c_str(), "DataLoader::get_sequencing_chemistries");
                        exit(EXIT_FAILURE);
                    } else{
                        flow_cell_product_code = std::string(flow_cell_product_code_c);
                    }
                    char* sequencing_kit_c = slow5_hdr_get("sequencing_kit", j, sp->header);
                    std::string sequencing_kit = "";
                    if(!sequencing_kit_c){
                        fprintf(stderr,"No sequencing_kit found in %s. (%s)\n", file_path.c_str(), "DataLoader::get_sequencing_chemistries");
                        exit(EXIT_FAILURE);
                    } else{
                        sequencing_kit = std::string(sequencing_kit_c);
                    }

                    slow5_rec_t *rec = NULL;
                    int ret=0;
                    int sample_rate = 0;
                    while((ret = slow5_get_next(&rec,sp)) >= 0){
                        sample_rate = rec->sampling_rate;
                        break;
                    }
                    slow5_rec_free(rec);

                    RunInfoDictData_t run_info_data;
                    run_info_data.flow_cell_product_code = flow_cell_product_code.c_str();
                    run_info_data.sequencing_kit = sequencing_kit.c_str();
                    run_info_data.sample_rate = sample_rate;

                    const auto chemistry_key = get_chemistry_key(&run_info_data);
                    spdlog::trace("BLOW5/SLOW5: {} {}", file_path.c_str(), to_string(chemistry_key));
                    chemistries.insert(chemistry_key);
                }
                slow5_close(sp);
            }
        }
    };

    iterate_directory(fetch_directory_entries(data_path, recursive_file_loading));
    return chemistries;
}

models::Chemistry DataLoader::get_unique_sequencing_chemisty(const std::string& data,
                                                             bool recursive_file_loading) {
    std::set<models::ChemistryKey> data_chemistries =
            get_sequencing_chemistries(data, recursive_file_loading);

    if (data_chemistries.empty()) {
        throw std::runtime_error(
                "Failed to determine sequencing chemistry from data. Please select a model by "
                "path");
    }

    std::set<models::Chemistry> found;
    for (const auto& dc : data_chemistries) {
        const auto chemistry = models::get_chemistry(dc);
        if (chemistry == models::Chemistry::UNKNOWN) {
            spdlog::error("No supported chemistry found for {}", to_string(dc));
            spdlog::error(
                    "This is typically seen when using prototype kits. Please download an "
                    "appropriate model for your data and select it by model path");

            throw std::runtime_error("Could not resolve chemistry from data: Unknown chemistry");
        }
        found.insert(chemistry);
    }
    if (found.empty()) {
        throw std::runtime_error("Could not resolve chemistry from data: No data");
    }
    if (found.size() > 1) {
        spdlog::error("Multiple sequencing chemistries found in data");
        for (auto f : found) {
            spdlog::error("Found: {}", to_string(f));
        }

        throw std::runtime_error("Could not uniquely resolve chemistry from inhomogeneous data");
    }
    return *std::begin(found);
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

    auto free_pod5 = [&]() {
        if (pod5_close_and_free_reader(file) != POD5_OK) {
            spdlog::error("Failed to close and free POD5 reader for file {}", path.c_str());
        }
    };

    auto post = utils::PostCondition(free_pod5);

    std::vector<uint8_t> read_id_array(POD5_READ_ID_SIZE * read_ids.size());
    for (size_t i = 0; i < read_ids.size(); i++) {
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

    // Create static threadpool so it is reused across calls to this function.
    static cxxpool::thread_pool pool{m_num_worker_threads};

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

        std::vector<std::future<SimplexReadPtr>> futures;
        for (std::size_t row_idx = 0; row_idx < traversal_batch_counts[batch_index]; row_idx++) {
            uint32_t row = traversal_batch_rows[row_idx + row_offset];

            if (can_process_pod5_row(batch, row, m_allowed_read_ids, m_ignored_read_ids)) {
                futures.push_back(pool.push(process_pod5_thread_fn, row, batch, file,
                                            std::cref(path), std::cref(m_reads_by_channel),
                                            std::cref(m_read_id_to_index)));
            }
        }

        for (auto& v : futures) {
            auto read = v.get();
            initialise_read(read->read_common);
            check_read(read);
            m_pipeline.push_message(std::move(read));
            m_loaded_read_count++;
        }

        if (pod5_free_read_batch(batch) != POD5_OK) {
            spdlog::error("Failed to release batch");
        }

        row_offset += traversal_batch_counts[batch_index];
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

        std::vector<std::future<SimplexReadPtr>> futures;

        for (std::size_t row = 0; row < batch_row_count; ++row) {
            // TODO - check the read ID here, for each one, only send the row if it is in the list of ones we care about

            if (can_process_pod5_row(batch, int(row), m_allowed_read_ids, m_ignored_read_ids)) {
                futures.push_back(pool.push(process_pod5_thread_fn, row, batch, file,
                                            std::cref(path), std::cref(m_reads_by_channel),
                                            std::cref(m_read_id_to_index)));
            }
        }

        for (auto& v : futures) {
            auto read = v.get();
            initialise_read(read->read_common);
            check_read(read);
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
    int num_reads = int(reads.getNumberObjects());

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
        if (ds.getDataType().string() != "Integer16") {
            throw std::runtime_error("Invalid FAST5 Signal data type of " +
                                     ds.getDataType().string());
        }

        auto options = at::TensorOptions().dtype(at::kShort);
        auto samples = at::empty(ds.getElementCount(), options);
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
        std::string exp_start_time = get_string_attribute(tracking_id_group, "exp_start_time");
        std::string flow_cell_id = get_string_attribute(tracking_id_group, "flow_cell_id");
        std::string flow_cell_product_code =
                get_string_attribute(tracking_id_group, "flow_cell_product_code");
        std::string device_id = get_string_attribute(tracking_id_group, "device_id");
        std::string group_protocol_id =
                get_string_attribute(tracking_id_group, "group_protocol_id");

        auto start_time_str = utils::adjust_time(exp_start_time,
                                                 static_cast<uint32_t>(start_time / sampling_rate));

        auto new_read = std::make_unique<SimplexRead>();
        new_read->read_common.sample_rate = uint64_t(sampling_rate);
        new_read->read_common.raw_data = samples;
        new_read->digitisation = digitisation;
        new_read->range = range;
        new_read->offset = offset;
        new_read->scaling = range / digitisation;
        new_read->read_common.read_id = read_id;
        new_read->read_common.num_trimmed_samples = 0;
        new_read->read_common.attributes.mux = mux;
        new_read->read_common.attributes.read_number = read_number;
        new_read->read_common.attributes.channel_number = channel_number;
        new_read->read_common.attributes.start_time = start_time_str;
        new_read->read_common.attributes.fast5_filename = fast5_filename;
        new_read->read_common.flowcell_id = flow_cell_id;
        new_read->read_common.flow_cell_product_code = flow_cell_product_code;
        new_read->read_common.position_id = device_id;
        new_read->read_common.experiment_id = group_protocol_id;
        new_read->read_common.is_duplex = false;

        if (!m_allowed_read_ids || (m_allowed_read_ids->find(new_read->read_common.read_id) !=
                                    m_allowed_read_ids->end())) {
            initialise_read(new_read->read_common);
            m_pipeline.push_message(std::move(new_read));
            m_loaded_read_count++;
        }
    }
}

void DataLoader::initialise_read(ReadCommon& read_common) const {
    for (const auto& initialiser : m_read_initialisers) {
        initialiser(read_common);
    }
}

void DataLoader::check_read(const SimplexReadPtr& read) {
    if (read->read_common.chemistry == models::Chemistry::UNKNOWN &&
        m_log_unknown_chemistry.exchange(false)) {
        spdlog::warn(
                "Could not determine sequencing Chemistry from read data - "
                "some features might be disabled");
    }
}

SimplexReadPtr create_read(slow5_file_t *sp, slow5_rec_t * rec, std::string m_device_, const std::unordered_map<int, std::vector<DataLoader::ReadSortInfo>>& reads_by_channel,
        const std::unordered_map<std::string, size_t>& read_id_to_index){
    std::vector<int16_t> tmp(rec->raw_signal, rec->raw_signal + rec->len_raw_signal);
    char* run_id_c = slow5_hdr_get("run_id", rec->read_group, sp->header);
    std::string run_id = "";
    if(!run_id_c){
        fprintf(stderr,"No run_id found in %s. (%s)\n", sp->meta.pathname, "create_read");
        exit(EXIT_FAILURE);
    } else{
        run_id = std::string(run_id_c);
    }
    char* flow_cell_id_c = slow5_hdr_get("flow_cell_id", rec->read_group, sp->header);
    std::string flow_cell_id = "";
    if(!flow_cell_id_c){
        fprintf(stderr,"No flowcell_id found in %s. (%s)\n", sp->meta.pathname, "create_read");
        exit(EXIT_FAILURE);
    } else{
        flow_cell_id = std::string(flow_cell_id_c);
    }
    char* flow_cell_product_code_c = slow5_hdr_get("flow_cell_product_code", rec->read_group, sp->header);
    std::string flow_cell_product_code = "";
    if(!flow_cell_product_code_c){
        fprintf(stderr,"No flow_cell_product_code found in %s. (%s)\n", sp->meta.pathname, "create_read");
        exit(EXIT_FAILURE);
    } else{
        flow_cell_product_code = std::string(flow_cell_product_code_c);
    }
    char* position_id_c = slow5_hdr_get("sequencer_position", rec->read_group, sp->header);
    std::string position_id = "";
    if(position_id_c){
        position_id = std::string(position_id_c);
        // fprintf(stderr,"No sequencer_position found in %s. (%s)\n", sp->meta.pathname, "create_read");
        // exit(EXIT_FAILURE);
    }

    char* experiment_id_c = slow5_hdr_get("experiment_name", rec->read_group, sp->header);
    std::string experiment_id = "";
    if(experiment_id_c){
        experiment_id = std::string(experiment_id_c);
        // fprintf(stderr,"No experiment_name found in %s. (%s)\n", sp->meta.pathname, "create_read");
        // exit(EXIT_FAILURE);
    }

    char* sequencing_kit_c = slow5_hdr_get("sequencing_kit", rec->read_group, sp->header);
    std::string sequencing_kit = "";
    if(!sequencing_kit_c){
        fprintf(stderr,"No sequencing_kit found in %s. (%s)\n", sp->meta.pathname, "create_read");
        exit(EXIT_FAILURE);
    } else{
        sequencing_kit = std::string(sequencing_kit_c);
    }
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

    char* exp_start_time_ms_c = slow5_hdr_get("acquisition_start_time", rec->read_group, sp->header);
    std::string exp_start_time_ms = "";
    if(!exp_start_time_ms_c){
        exp_start_time_ms_c = slow5_hdr_get("exp_start_time", rec->read_group, sp->header);
        if(!exp_start_time_ms_c) {
            throw std::runtime_error("Neither acquisition_start_time nor exp_start_time found");
        }
    }
    exp_start_time_ms = std::string(exp_start_time_ms_c);
    auto run_acquisition_start_time_ms = utils::get_unix_time_from_string_timestamp(exp_start_time_ms);
    // std::cerr << "run_acquisition_start_time_ms: " << run_acquisition_start_time_ms << std::endl;
    auto start_time_ms = run_acquisition_start_time_ms + ((start_time * 1000) /(uint64_t)rec->sampling_rate);
    auto start_time_str = utils::get_string_timestamp_from_unix_time(start_time_ms);

    auto new_read = std::make_unique<SimplexRead>();

    auto options = at::TensorOptions().dtype(at::kShort);
    new_read->read_common.raw_data = at::from_blob(tmp.data(), tmp.size(), options).clone().to(m_device_);
    new_read->read_common.sample_rate = rec->sampling_rate;
    new_read->run_acquisition_start_time_ms = run_acquisition_start_time_ms;
    new_read->read_common.start_time_ms = start_time_ms;
    new_read->scaling = rec->range / rec->digitisation;
    new_read->offset = rec->offset;
    new_read->read_common.read_id = std::string(rec->read_id);
    new_read->read_common.num_trimmed_samples = 0;
    new_read->read_common.attributes.read_number = read_number;
    new_read->read_common.attributes.fast5_filename = std::string(sp->meta.pathname);
    new_read->read_common.attributes.mux = mux;
    new_read->read_common.attributes.num_samples = rec->len_raw_signal;
    new_read->read_common.attributes.channel_number = channel_number;
    new_read->read_common.attributes.start_time = start_time_str;
    new_read->read_common.run_id = run_id;
    new_read->start_sample = start_time;
    new_read->end_sample = start_time + rec->len_raw_signal;
    new_read->read_common.flowcell_id = flow_cell_id;
    new_read->read_common.flow_cell_product_code = flow_cell_product_code;
    new_read->read_common.position_id = position_id;
    new_read->read_common.experiment_id = experiment_id;
    new_read->read_common.is_duplex = false;

    // Get the condition_info from the run_info_data to determine if the sequencing kit
    // used has a rapid adapter and which one.

    RunInfoDictData_t run_info_data;
    run_info_data.flow_cell_product_code = flow_cell_product_code.c_str();
    run_info_data.sequencing_kit = sequencing_kit.c_str();
    run_info_data.sample_rate = rec->sampling_rate;

    const auto condition_info = models::ConditionInfo(get_chemistry_key(&run_info_data));
    new_read->read_common.rapid_chemistry = condition_info.rapid_chemistry();
    new_read->read_common.chemistry = condition_info.chemistry();

    uint8_t end_reason = slow5_aux_get_enum(rec,"end_reason",&ret);
    if(ret!=0){
        fprintf(stderr,"Error in getting auxiliary attribute from the file. Error code %d\n",ret);
        exit(EXIT_FAILURE);
    }

    uint8_t num_label = 0;
    char **labels = slow5_get_aux_enum_labels(sp->header, "end_reason", &num_label);
    if(labels==NULL){
        fprintf(stderr,"Error in getting list of enum labels\n");
        exit(EXIT_FAILURE);
    }

    // for (int i = 0; i < num_label; ++i) {
    //     if (labels[i] != nullptr) { // Check for null pointers
    //         std::cerr << "Label " << i << ": " << labels[i] << std::endl;
    //     } else {
    //         std::cerr << "Label " << i << ": (null)" << std::endl;
    //     }
    // }
    
    // if(end_reason != SLOW5_ENUM_NULL){
    //     printf("end_reason = %s\n", labels[end_reason]);
    // } else{
    //     printf("end_reason is missing for the record\n");
    // }



    if(end_reason != SLOW5_ENUM_NULL){
        // pod5_end_reason_t end_reason_value = static_cast<pod5_end_reason_t>(end_reason);
        // if (end_reason_value == POD5_END_REASON_UNBLOCK_MUX_CHANGE ||
        //        end_reason_value == POD5_END_REASON_MUX_CHANGE) {
        //     new_read->read_common.attributes.is_end_reason_mux_change = true;
        // }
        if(strcmp(labels[end_reason],"unblock_mux_change")==0 || strcmp(labels[end_reason],"mux_change")==0){
            new_read->read_common.attributes.is_end_reason_mux_change = true;
        }
    } else{
        spdlog::error("Failed to get read end_reason for read {}", new_read->read_common.read_id);
    }


    // Determine the time sorted predecessor of the read
    // if that information is available (primarily used for offline
    // duplex runs).
    if (reads_by_channel.find(channel_number) != reads_by_channel.end()) {
        auto& read_id = new_read->read_common.read_id;
        const auto& v = reads_by_channel.at(channel_number);
        auto read_id_iter = v.begin() + read_id_to_index.at(read_id);

        if (read_id_iter != v.begin()) {
            new_read->prev_read = std::prev(read_id_iter)->read_id;
        }
        if (std::next(read_id_iter) != v.end()) {
            new_read->next_read = std::next(read_id_iter)->read_id;
        }
    }

    // dump_to_stderr(new_read);

    return new_read;


}

void create_read_data(core_t *core, db_t *db, int32_t i) {
    //
    struct slow5_rec *rec = NULL;
    if (slow5_rec_depress_parse(&db->mem_records[i], &db->mem_bytes[i], NULL, &rec, core->fp) != 0) {
        exit(EXIT_FAILURE);
    } else {
        free(db->mem_records[i]);
    }
    auto new_read = create_read(core->fp, rec, core->m_device_, core->reads_by_channel, core->read_id_to_index);
    //
    // db->read_data_ptrs[i] = new_read;
    db->read_data_ptrs[i] = std::move(new_read);
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
        core_t core = {0};
        core.num_thread = (num_threads > record_count) ? record_count : num_threads;
        if (record_count == 0) {
            core.num_thread = 1;
        }
        core.fp = sp;
        core.m_device_ = m_device;
        core.reads_by_channel = std::cref(m_reads_by_channel);
        core.read_id_to_index = std::cref(m_read_id_to_index);


        db.n_batch = record_count;
        db.read_data_ptrs = std::vector<dorado::SimplexReadPtr>(record_count);
        work_db(&core, &db, create_read_data);

        for (int64_t i = 0; i < record_count; i++) {
            if (!m_allowed_read_ids ||
                (m_allowed_read_ids->find(db.read_data_ptrs[i]->read_common.read_id) != m_allowed_read_ids->end())) {
                initialise_read(db.read_data_ptrs[i]->read_common);
                check_read(db.read_data_ptrs[i]);
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
    slow5_close(sp);
}
void DataLoader::load_slow5_reads_from_file_by_read_ids(slow5_file_t *sp, const std::vector<ReadID>& read_ids) {

    int ret = 0;
    slow5_rec_t **rec = NULL;

    size_t num_rid = read_ids.size();
    size_t read_count = 0;
    char **rid = (char**)malloc(sizeof(char*)*slow5_batchsize);

    while(1){
        int local_batch_size = ((size_t)slow5_batchsize > (num_rid - read_count)) ? num_rid - read_count : (size_t)slow5_batchsize;
        for(size_t i=0; i<(size_t)local_batch_size; i++){
            char read_id_tmp[POD5_READ_ID_LEN];
            if (pod5_format_read_id(read_ids[read_count].data(), read_id_tmp) != POD5_OK) {
                spdlog::error("Failed to format read id");
            }
            std::string read_s(read_id_tmp);
            
            rid[i] = strdup(read_s.c_str());
            read_count++;
        }
        ret = slow5_get_batch_lazy(&rec, sp, rid, local_batch_size, slow5_threads);
        assert(ret==local_batch_size);
        for(int i=0;i<ret;i++){
            if (!m_allowed_read_ids ||
                (m_allowed_read_ids->find(std::string(rec[i]->read_id)) != m_allowed_read_ids->end())) {
                auto new_read = create_read(sp, rec[i], m_device, std::cref(m_reads_by_channel), std::cref(m_read_id_to_index));
                spdlog::debug("read_id queued: {}",rec[i]->read_id);
                // m_pipeline.push_message(new_read);
                initialise_read(new_read->read_common);
                check_read(new_read);
                m_pipeline.push_message(std::move(new_read));
                m_loaded_read_count++;
            }
        }
        slow5_free_batch_lazy(&rec,ret);
        for(int i=0; i<local_batch_size; i++){
            free(rid[i]);
        }
        if(local_batch_size < slow5_batchsize){
            break;
        }
    }
    free(rid);
    
}



DataLoader::DataLoader(Pipeline& pipeline,
                       const std::string& device,
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
          m_ignored_read_ids(std::move(read_ignore_list)) {
    slow5_threads = slow5_threads_;
    slow5_batchsize = slow5_batchsize_;
    m_max_reads = max_reads == 0 ? std::numeric_limits<decltype(m_max_reads)>::max() : max_reads;
    assert(m_num_worker_threads > 0);
    static std::once_flag vbz_init_flag;
    std::call_once(vbz_init_flag, vbz_register);
}

stats::NamedStats DataLoader::sample_stats() const {
    return stats::NamedStats{{"loaded_read_count", static_cast<double>(m_loaded_read_count)}};
}
}  // namespace dorado
