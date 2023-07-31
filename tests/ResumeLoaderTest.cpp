#include "MessageSinkUtils.h"
#include "TestUtils.h"
#include "read_pipeline/ResumeLoaderNode.h"

#include <catch2/catch.hpp>

#define TEST_GROUP "[read_pipeline][ResumeLoaderNode]"

namespace fs = std::filesystem;

TEST_CASE(TEST_GROUP) {
    std::vector<dorado::Message> messages;
    MessageSinkToVector sink(100, messages);
    fs::path aligner_test_dir = fs::path(get_data_dir("aligner_test"));
    auto sam = aligner_test_dir / "basecall.sam";

    dorado::ResumeLoaderNode loader(sink, sam.string());
    loader.copy_completed_reads();
    sink.terminate();
    CHECK(messages.size() == 1);
    auto read_ids = loader.get_processed_read_ids();
    CHECK(read_ids.count("002bd127-db82-436f-b828-28567c3d505d") == 1);
}
