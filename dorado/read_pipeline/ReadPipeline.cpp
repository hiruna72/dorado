#include "ReadPipeline.h"

#include "modbase/ModBaseContext.h"
#include "stereo_features.h"
#include "utils/bam_utils.h"
#include "utils/sequence_utils.h"

#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <unordered_map>

using namespace std::chrono_literals;

namespace dorado {

std::string ReadCommon::generate_read_group() const {
    std::string read_group;
    if (!run_id.empty()) {
        read_group = run_id + '_';
        if (model_name.empty()) {
            read_group += "unknown";
        } else {
            read_group += model_name;
        }
        if (!barcode.empty() && barcode != "unclassified") {
            read_group += '_' + barcode;
        }
    }
    return read_group;
}

void ReadCommon::generate_read_tags(bam1_t *aln, bool emit_moves, bool is_duplex_parent) const {
    int qs = static_cast<int>(std::round(calculate_mean_qscore()));
    bam_aux_append(aln, "qs", 'i', sizeof(qs), (uint8_t *)&qs);

    float du = (float)(get_raw_data_samples() + num_trimmed_samples) / (float)sample_rate;
    bam_aux_append(aln, "du", 'f', sizeof(du), (uint8_t *)&du);

    int ns = get_raw_data_samples() + num_trimmed_samples;
    bam_aux_append(aln, "ns", 'i', sizeof(ns), (uint8_t *)&ns);

    int ts = num_trimmed_samples;
    bam_aux_append(aln, "ts", 'i', sizeof(ts), (uint8_t *)&ts);

    int mx = attributes.mux;
    bam_aux_append(aln, "mx", 'i', sizeof(mx), (uint8_t *)&mx);

    int ch = attributes.channel_number;
    bam_aux_append(aln, "ch", 'i', sizeof(ch), (uint8_t *)&ch);

    bam_aux_append(aln, "st", 'Z', attributes.start_time.length() + 1,
                   (uint8_t *)attributes.start_time.c_str());

    // For reads which are the result of read splitting, the read number will be set to -1
    int rn = attributes.read_number;
    bam_aux_append(aln, "rn", 'i', sizeof(rn), (uint8_t *)&rn);

    bam_aux_append(aln, "fn", 'Z', attributes.fast5_filename.length() + 1,
                   (uint8_t *)attributes.fast5_filename.c_str());

    float sm = shift;
    bam_aux_append(aln, "sm", 'f', sizeof(sm), (uint8_t *)&sm);

    float sd = scale;
    bam_aux_append(aln, "sd", 'f', sizeof(sd), (uint8_t *)&sd);

    bam_aux_append(aln, "sv", 'Z', scaling_method.size() + 1, (uint8_t *)scaling_method.c_str());

    int32_t dx = (is_duplex_parent ? -1 : 0);
    bam_aux_append(aln, "dx", 'i', sizeof(dx), (uint8_t *)&dx);

    auto rg = generate_read_group();
    if (!rg.empty()) {
        bam_aux_append(aln, "RG", 'Z', rg.length() + 1, (uint8_t *)rg.c_str());
    }

    if (!parent_read_id.empty()) {
        bam_aux_append(aln, "pi", 'Z', parent_read_id.size() + 1,
                       (uint8_t *)parent_read_id.c_str());
        // For split reads, also store the start coordinate of the new read
        // in the original signal.
        bam_aux_append(aln, "sp", 'i', sizeof(split_point), (uint8_t *)&split_point);
    }

    if (emit_moves) {
        std::vector<uint8_t> m(moves.size() + 1, 0);
        m[0] = model_stride;

        for (size_t idx = 0; idx < moves.size(); idx++) {
            m[idx + 1] = static_cast<uint8_t>(moves[idx]);
        }

        bam_aux_update_array(aln, "mv", 'c', m.size(), (uint8_t *)m.data());
    }

    if (rna_poly_tail_length >= 0) {
        bam_aux_append(aln, "pt", 'i', sizeof(rna_poly_tail_length),
                       (uint8_t *)&rna_poly_tail_length);
    }
}

void ReadCommon::generate_duplex_read_tags(bam1_t *aln) const {
    int qs = static_cast<int>(std::round(calculate_mean_qscore()));
    bam_aux_append(aln, "qs", 'i', sizeof(qs), (uint8_t *)&qs);
    uint32_t duplex = 1;
    bam_aux_append(aln, "dx", 'i', sizeof(duplex), (uint8_t *)&duplex);

    int mx = attributes.mux;
    bam_aux_append(aln, "mx", 'i', sizeof(mx), (uint8_t *)&mx);

    int ch = attributes.channel_number;
    bam_aux_append(aln, "ch", 'i', sizeof(ch), (uint8_t *)&ch);

    bam_aux_append(aln, "st", 'Z', attributes.start_time.length() + 1,
                   (uint8_t *)attributes.start_time.c_str());

    auto rg = generate_read_group();
    if (!rg.empty()) {
        bam_aux_append(aln, "RG", 'Z', rg.length() + 1, (uint8_t *)rg.c_str());
    }

    if (!parent_read_id.empty()) {
        bam_aux_append(aln, "pi", 'Z', parent_read_id.size() + 1,
                       (uint8_t *)parent_read_id.c_str());
    }
}

void ReadCommon::generate_modbase_tags(bam1_t *aln, uint8_t threshold) const {
    if (!mod_base_info) {
        return;
    }

    const size_t num_channels = mod_base_info->alphabet.size();
    const std::string cardinal_bases = "ACGT";
    char current_cardinal = 0;
    if (seq.length() * num_channels != base_mod_probs.size()) {
        throw std::runtime_error(
                "Mismatch between base_mod_probs size and sequence length * num channels in "
                "modbase_alphabet!");
    }

    std::string modbase_string = "";
    std::vector<uint8_t> modbase_prob;

    // Create a mask indicating which bases are modified.
    std::unordered_map<char, bool> base_has_context = {
            {'A', false}, {'C', false}, {'G', false}, {'T', false}};
    utils::ModBaseContext context_handler;
    if (!mod_base_info->context.empty()) {
        if (!context_handler.decode(mod_base_info->context)) {
            throw std::runtime_error("Invalid base modification context string.");
        }
        for (auto base : cardinal_bases) {
            if (context_handler.motif(base).size() > 1) {
                // If the context is just the single base, then this is equivalent to no context.
                base_has_context[base] = true;
            }
        }
    }
    auto modbase_mask = context_handler.get_sequence_mask(seq);
    context_handler.update_mask(modbase_mask, seq, mod_base_info->alphabet, base_mod_probs,
                                threshold);

    // Iterate over the provided alphabet and find all the channels we need to write out
    for (size_t channel_idx = 0; channel_idx < num_channels; channel_idx++) {
        if (cardinal_bases.find(mod_base_info->alphabet[channel_idx]) != std::string::npos) {
            // A cardinal base
            current_cardinal = mod_base_info->alphabet[channel_idx][0];
        } else {
            // A modification on the previous cardinal base
            std::string bam_name = mod_base_info->alphabet[channel_idx];
            if (!utils::validate_bam_tag_code(bam_name)) {
                return;
            }

            // Write out the results we found
            modbase_string += std::string(1, current_cardinal) + "+" + bam_name;
            modbase_string += base_has_context[current_cardinal] ? "?" : ".";
            int skipped_bases = 0;
            for (size_t base_idx = 0; base_idx < seq.size(); base_idx++) {
                if (seq[base_idx] == current_cardinal) {
                    if (modbase_mask[base_idx]) {
                        modbase_string += "," + std::to_string(skipped_bases);
                        skipped_bases = 0;
                        modbase_prob.push_back(
                                base_mod_probs[base_idx * num_channels + channel_idx]);
                    } else {
                        // Skip this base
                        skipped_bases++;
                    }
                }
            }
            modbase_string += ";";
        }
    }

    bam_aux_append(aln, "MM", 'Z', modbase_string.length() + 1, (uint8_t *)modbase_string.c_str());
    bam_aux_update_array(aln, "ML", 'C', modbase_prob.size(), (uint8_t *)modbase_prob.data());
}

float ReadCommon::calculate_mean_qscore() const {
    // If Q-score start position is greater than the
    // read length, then calculate mean Q-score from the
    // start of the read.
    if (qstring.length() <= mean_qscore_start_pos) {
        return utils::mean_qscore_from_qstring(qstring, 0);
    }
    return utils::mean_qscore_from_qstring(qstring, mean_qscore_start_pos);
}

std::vector<BamPtr> ReadCommon::extract_sam_lines(bool emit_moves,
                                                  uint8_t modbase_threshold,
                                                  bool is_duplex_parent) const {
    if (read_id.empty()) {
        throw std::runtime_error("Empty read_name string provided");
    }
    if (seq.size() != qstring.size()) {
        throw std::runtime_error("Sequence and qscore do not match size for read id " + read_id);
    }
    if (seq.empty()) {
        throw std::runtime_error("Empty sequence and qstring provided for read id " + read_id);
    }

    std::vector<BamPtr> alns;

    bam1_t *aln = bam_init1();
    uint32_t flags = 4;              // 4 = UNMAPPED
    std::string ref_seq = "*";       // UNMAPPED
    int leftmost_pos = -1;           // UNMAPPED - will be written as 0
    int map_q = 0;                   // UNMAPPED
    std::string cigar_string = "*";  // UNMAPPED
    std::string r_next = "*";
    int next_pos = -1;  // UNMAPPED - will be written as 0
    size_t template_length = seq.size();

    // Convert string qscore to phred vector.
    std::vector<uint8_t> qscore;
    std::transform(qstring.begin(), qstring.end(), std::back_inserter(qscore),
                   [](char c) { return (uint8_t)(c)-33; });

    bam_set1(aln, read_id.length(), read_id.c_str(), flags, -1, leftmost_pos, map_q, 0, nullptr, -1,
             next_pos, 0, seq.length(), seq.c_str(), (char *)qscore.data(), 0);

    if (!barcode.empty() && barcode != "unclassified") {
        bam_aux_append(aln, "BC", 'Z', barcode.length() + 1, (uint8_t *)barcode.c_str());
    }

    if (is_duplex) {
        generate_duplex_read_tags(aln);
    } else {
        generate_read_tags(aln, emit_moves, is_duplex_parent);
    }
    generate_modbase_tags(aln, modbase_threshold);
    alns.push_back(BamPtr(aln));

    return alns;
}

ReadCommon &get_read_common_data(const Message &message) {
    if (!is_read_message(message)) {
        throw std::invalid_argument("Message is not a read");
    } else {
        if (std::holds_alternative<SimplexReadPtr>(message)) {
            return std::get<SimplexReadPtr>(message)->read_common;
        } else {
            return std::get<DuplexReadPtr>(message)->read_common;
        }
    }
}

void materialise_read_raw_data(Message &message) {
    if (std::holds_alternative<DuplexReadPtr>(message)) {
        auto &duplex_read = *std::get<DuplexReadPtr>(message);
        duplex_read.read_common.raw_data =
                GenerateStereoFeatures(duplex_read.stereo_feature_inputs);
    }
}

ReadPair::ReadData ReadPair::ReadData::from_read(const SimplexRead &read,
                                                 uint64_t seq_start,
                                                 uint64_t seq_end) {
    ReadData data;
    data.read_common = read.read_common;
    data.seq_start = seq_start;
    data.seq_end = seq_end;
    return data;
}

MessageSink::MessageSink(size_t max_messages) : m_work_queue(max_messages) {}

void MessageSink::push_message_internal(Message &&message) {
    const auto status = m_work_queue.try_push(std::move(message));
    // try_push will fail if the sink has been told to terminate.
    // We do not expect to be pushing reads from this source if that is the case.
    assert(status == utils::AsyncQueueStatus::Success);
}

// Depth first search that establishes a topological ordering for node destruction.
// Returns true if a cycle is found.
bool Pipeline::DFS(const std::vector<PipelineDescriptor::NodeDescriptor> &node_descriptors,
                   NodeHandle node_handle,
                   std::vector<DFSState> &dfs_state,
                   std::vector<NodeHandle> &source_to_sink_order) {
    auto &node_state = dfs_state.at(node_handle);
    if (node_state == DFSState::Visited) {
        // Already reached this node via another route.
        return false;
    }
    if (node_state == DFSState::Visiting) {
        // Back edge => cycle.
        return true;
    }
    node_state = DFSState::Visiting;
    auto sink_handles = node_descriptors.at(node_handle).sink_handles;
    for (auto sink_handle : sink_handles) {
        if (DFS(node_descriptors, sink_handle, dfs_state, source_to_sink_order)) {
            return true;
        }
    }
    node_state = DFSState::Visited;
    source_to_sink_order.push_back(node_handle);
    return false;
}

std::unique_ptr<Pipeline> Pipeline::create(
        PipelineDescriptor &&descriptor,
        std::vector<dorado::stats::StatsReporter> *const stats_reporters,
        stats::NamedStats *const final_stats) {
    // Find a source node, i.e. one that is not the sink of any other node.
    // There should be exactly 1 one for a valid pipeline.
    const auto node_count = descriptor.m_node_descriptors.size();
    std::vector<bool> is_sink(node_count, false);
    for (auto &[desc_node, sink_handles] : descriptor.m_node_descriptors) {
        for (auto sink_handle : sink_handles) {
            is_sink.at(sink_handle) = true;
        }
    }
    const auto num_sources = std::count(is_sink.cbegin(), is_sink.cend(), false);
    if (num_sources != 1) {
        spdlog::error("There must be exactly 1 source node.  {} were present.", num_sources);
        return nullptr;
    }
    auto source_it = std::find(is_sink.cbegin(), is_sink.cend(), false);
    auto source_node = std::distance(is_sink.cbegin(), source_it);

    // Perform a depth first search from the source to determine the
    // source-to-sink destruction order.  At the same time, cycles are detected.
    std::vector<NodeHandle> source_to_sink_order;
    std::vector<DFSState> dfs_state(node_count, DFSState::Unvisited);
    const bool has_cycle =
            DFS(descriptor.m_node_descriptors, source_node, dfs_state, source_to_sink_order);
    if (has_cycle) {
        spdlog::error("Graph has cycle");
        return nullptr;
    }
    std::reverse(source_to_sink_order.begin(), source_to_sink_order.end());
    // If the graph is fully connected then we should have visited all nodes.
    assert(std::all_of(dfs_state.cbegin(), dfs_state.cend(),
                       [](DFSState v) { return v == DFSState::Visited; }));
    assert(source_to_sink_order.size() == descriptor.m_node_descriptors.size());

    return std::unique_ptr<Pipeline>(
            new Pipeline(std::move(descriptor), source_to_sink_order, stats_reporters));
}

Pipeline::Pipeline(PipelineDescriptor &&descriptor,
                   std::vector<NodeHandle> source_to_sink_order,
                   std::vector<dorado::stats::StatsReporter> *const stats_reporters)
        : m_source_to_sink_order(std::move(source_to_sink_order)) {
    for (auto &[desc_node, _] : descriptor.m_node_descriptors) {
        m_nodes.push_back(std::move(desc_node));
        if (stats_reporters) {
            stats_reporters->push_back(stats::make_stats_reporter(*m_nodes.back()));
        }
    }

    for (size_t i = 0; i < m_nodes.size(); ++i) {
        auto &node = m_nodes.at(i);
        const auto &sink_handles = descriptor.m_node_descriptors.at(i).sink_handles;
        for (const auto sink_handle : sink_handles)
            node->add_sink(dynamic_cast<MessageSink &>(*m_nodes.at(sink_handle)));
    }
}

void MessageSink::add_sink(MessageSink &sink) { m_sinks.push_back(std::ref(sink)); }

void Pipeline::push_message(Message &&message) {
    assert(!m_nodes.empty());
    const auto source_node_index = m_source_to_sink_order.front();
    dynamic_cast<MessageSink &>(*m_nodes.at(source_node_index)).push_message(std::move(message));
}

stats::NamedStats Pipeline::terminate(const FlushOptions &flush_options) {
    stats::NamedStats final_stats;
    // Nodes must be terminated in source to sink order to ensure all in flight
    // processing is completed, and sources still have valid sinks as they finish
    // work.
    for (auto handle : m_source_to_sink_order) {
        auto &node = m_nodes.at(handle);
        node->terminate(flush_options);
        auto node_stats = node->sample_stats();
        const auto node_name = node->get_name();
        for (const auto &[name, value] : node_stats) {
            final_stats[node_name + "." + name] = value;
        }
    }
    return final_stats;
}

void Pipeline::restart() {
    // The order in which we restart nodes shouldn't matter, so
    // we go source to sink.
    for (auto handle : m_source_to_sink_order) {
        m_nodes.at(handle)->restart();
    }
}

Pipeline::~Pipeline() {
    for (auto handle : m_source_to_sink_order) {
        auto &node = m_nodes.at(handle);
        node.reset();
    }
}

// Free functions
bool is_read_message(const Message &message) {
    return std::holds_alternative<SimplexReadPtr>(message) ||
           std::holds_alternative<DuplexReadPtr>(message);
}

uint64_t SimplexRead::get_end_time_ms() const {
    return read_common.start_time_ms +
           ((end_sample - start_sample) * 1000) /
                   read_common.sample_rate;  //TODO get rid of the trimmed thing?
}

}  // namespace dorado
