#pragma once
#include "HtsReader.h"
#include "ReadPipeline.h"
#include "utils/bam_utils.h"

#include <map>
#include <memory>
#include <string>

namespace dorado {
// Duplex caller node receives a map of template_id to complement_id (typically generated from a pairs file),
// and a map of `read_id` to `dorado::Read` object. It then performs duplex calling and pushes `dorado::Read`
// objects to its output queue.
class BaseSpaceDuplexCallerNode : public MessageSink {
public:
    BaseSpaceDuplexCallerNode(std::map<std::string, std::string> template_complement_map,
                              read_map reads,
                              size_t threads);
    ~BaseSpaceDuplexCallerNode() { terminate_impl(); }
    std::string get_name() const override { return "BaseSpaceDuplexCallerNode"; }
    void terminate() override { terminate_impl(); }

private:
    void terminate_impl();
    void worker_thread();
    void basespace(const std::string& template_read_id, const std::string& complement_read_id);

    size_t m_num_worker_threads{1};
    std::unique_ptr<std::thread> m_worker_thread;
    std::map<std::string, std::string> m_template_complement_map;
    read_map m_reads;
};
}  // namespace dorado
