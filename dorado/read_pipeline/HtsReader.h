#pragma once
#include "htslib/sam.h"
#include "read_pipeline/ReadPipeline.h"
#include "utils/stats.h"
#include "utils/types.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace dorado {

using read_map = std::unordered_map<std::string, std::shared_ptr<Read>>;

class Pipeline;

class HtsReader {
public:
    HtsReader(const std::string& filename);
    ~HtsReader();
    bool read();
    void read(Pipeline& pipeline, int max_reads = -1);
    template <typename T>
    T get_tag(std::string tagname);
    bool has_tag(std::string tagname);

    char* format{nullptr};
    bool is_aligned{false};
    BamPtr record{nullptr};
    sam_hdr_t* header{nullptr};

private:
    htsFile* m_file{nullptr};
};

template <typename T>
T HtsReader::get_tag(std::string tagname) {
    T tag_value;
    uint8_t* tag = bam_aux_get(record.get(), tagname.c_str());

    if (!tag) {
        return tag_value;
    }
    if constexpr (std::is_integral_v<T>) {
        tag_value = bam_aux2i(tag);
    } else if constexpr (std::is_floating_point_v<T>) {
        tag_value = bam_aux2f(tag);
    } else {
        tag_value = bam_aux2Z(tag);
    }

    return tag_value;
}

/**
 * @brief Reads a SAM/BAM/CRAM file and returns a map of read IDs to Read objects.
 *
 * This function opens a SAM/BAM/CRAM file specified by the input filename parameter,
 * reads the alignments, and creates a map that associates read IDs with their
 * corresponding Read objects. The Read objects contain the read ID, sequence,
 * and quality string.
 *
 * @param filename The input BAM file path as a string.
 * @param read_ids A set of read_ids to filter on.
 * @return A map with read IDs as keys and shared pointers to Read objects as values.
 *
 * @note The caller is responsible for managing the memory of the returned map.
 * @note The input BAM file must be properly formatted and readable.
 */
read_map read_bam(const std::string& filename, const std::unordered_set<std::string>& read_ids);

/**
 * @brief Reads an HTS file format (SAM/BAM/FASTX/etc) and returns a set of read ids.
 *
 * This function opens the HTS file using the htslib APIs and iterates through
 * all records. When an unreadable record is encountered, the iteration is stopped
 * and all read ids seen so far are returned.
 *
 * @param filename The path to the input HTS file.
 * @return An unordered set with read ids.
 */
std::unordered_set<std::string> fetch_read_ids(const std::string& filename);

}  // namespace dorado
