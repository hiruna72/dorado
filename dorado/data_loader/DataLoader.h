#pragma once
#include <string>
#include <unordered_map>
#include <unordered_set>

struct ReadGroup {
    std::string run_id;
    std::string basecalling_model;
    std::string flowcell_id;
    std::string device_id;
    std::string exp_start_time;
    std::string sample_id;
};

namespace dorado {

class ReadSink;

class DataLoader {
public:
    DataLoader(ReadSink& read_sink,
               const std::string& device,
               size_t num_worker_threads,
               size_t max_reads = 0,
               std::unordered_set<std::string> read_list = std::unordered_set<std::string>(),
               int32_t slow5_threads = 8,
               int64_t slow5_batchsize = 4000);
    void load_reads(const std::string& path);

    static std::unordered_map<std::string, ReadGroup> load_read_groups(std::string data_path,
                                                                       std::string model_path);

private:
#ifdef USE_FAST5
    void load_fast5_reads_from_file(const std::string& path);
#endif
#ifdef USE_POD5
    void load_pod5_reads_from_file(const std::string& path);
#endif
    void load_slow5_reads_from_file(const std::string& path);
    ReadSink& m_read_sink;  // Where should the loaded reads go?
    size_t m_loaded_read_count{0};
    std::string m_device;
    int32_t slow5_threads{8};
    int64_t slow5_batchsize{4000};
    size_t m_num_worker_threads{1};
    size_t m_max_reads{0};
    std::unordered_set<std::string> m_allowed_read_ids;
};

}  // namespace dorado
