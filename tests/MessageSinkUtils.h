#pragma once

#include "data_loader/DataLoader.h"
#include "read_pipeline/ReadPipeline.h"

#include <memory>
#include <vector>

class MessageSinkToVector : public dorado::MessageSink {
public:
    MessageSinkToVector(size_t max_messages, std::vector<dorado::Message>& messages)
            : MessageSink(max_messages, 0), m_messages(messages) {
        start_threads();
    }
    ~MessageSinkToVector() { terminate_impl(); }
    void terminate(const dorado::FlushOptions&) override { terminate_impl(); }
    void restart() override {
        start_input_queue();
        start_threads();
    }

private:
    void start_threads() {
        m_worker_thread = std::make_unique<std::thread>(
                std::thread(&MessageSinkToVector::worker_thread, this));
    }

    void terminate_impl() {
        terminate_input_queue();
        if (m_worker_thread && m_worker_thread->joinable()) {
            m_worker_thread->join();
        }
        m_worker_thread.reset();
    }

    std::unique_ptr<std::thread> m_worker_thread;
    std::vector<dorado::Message>& m_messages;

    void worker_thread() {
        dorado::Message message;
        while (get_input_message(message)) {
            m_messages.push_back(std::move(message));
        }
    }
};

template <class T>
std::vector<T> ConvertMessages(std::vector<dorado::Message>&& messages) {
    std::vector<T> converted_messages;
    for (auto& message : messages) {
        converted_messages.push_back(std::get<T>(std::move(message)));
    }
    messages.clear();
    return converted_messages;
}

inline size_t CountSinkReads(const std::string& data_path,
                             const std::string& device,
                             size_t num_worker_threads,
                             size_t max_reads,
                             std::optional<std::unordered_set<std::string>> read_list,
                             std::unordered_set<std::string> read_ignore_list) {
    dorado::PipelineDescriptor pipeline_desc;
    std::vector<dorado::Message> messages;
    pipeline_desc.add_node<MessageSinkToVector>({}, 100, messages);
    auto pipeline = dorado::Pipeline::create(std::move(pipeline_desc), nullptr);

    dorado::DataLoader loader(*pipeline, device, num_worker_threads, max_reads,
                              std::move(read_list), std::move(read_ignore_list));
    loader.load_reads(data_path, false, dorado::ReadOrder::UNRESTRICTED);
    pipeline.reset();
    return messages.size();
}