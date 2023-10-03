#include "Version.h"
#include "read_pipeline/HtsReader.h"
#include "utils/bam_utils.h"
#include "utils/log_utils.h"

#include <argparse.hpp>
#include <date/date.h>
#include <date/tz.h>
#include <spdlog/spdlog.h>

#include <cctype>
#include <csignal>
#include <filesystem>

namespace dorado {

volatile sig_atomic_t interrupt = 0;

// todo: move to time_utils after !273
double time_difference_seconds(const std::string &timestamp1, const std::string &timestamp2) {
    using namespace date;
    using namespace std::chrono;
    try {
        std::istringstream ss1(timestamp1);
        std::istringstream ss2(timestamp2);
        sys_time<microseconds> time1, time2;
        ss1 >> parse("%FT%T%Ez", time1);
        ss2 >> parse("%FT%T%Ez", time2);
        // If parsing with timezone offset failed, try parsing with 'Z' format
        if (ss1.fail()) {
            ss1.clear();
            ss1.str(timestamp1);
            ss1 >> parse("%FT%TZ", time1);
        }
        if (ss2.fail()) {
            ss2.clear();
            ss2.str(timestamp2);
            ss2 >> parse("%FT%TZ", time2);
        }
        duration<double> diff = time1 - time2;
        return diff.count();
    } catch (const std::exception &e) {
        throw std::runtime_error("Failed to parse timestamps");
    }
}

int summary(int argc, char *argv[]) {
    utils::InitLogging();

    argparse::ArgumentParser parser("dorado", DORADO_VERSION, argparse::default_arguments::help);
    parser.add_argument("reads").help("SAM/BAM file produced by dorado basecaller.");
    parser.add_argument("-s", "--separator").default_value(std::string("\t"));
    parser.add_argument("-v", "--verbose").default_value(false).implicit_value(true);

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception &e) {
        std::ostringstream parser_stream;
        parser_stream << parser;
        spdlog::error("{}\n{}", e.what(), parser_stream.str());
        std::exit(1);
    }

    if (parser.get<bool>("--verbose")) {
        utils::SetDebugLogging();
    }

    std::vector<std::string> header = {
            "filename",
            "read_id",
            "run_id",
            "channel",
            "mux",
            "start_time",
            "duration",
            "template_start",
            "template_duration",
            "sequence_length_template",
            "mean_qscore_template",
    };

    std::vector<std::string> aligned_header = {
            "alignment_genome",         "alignment_genome_start",    "alignment_genome_end",
            "alignment_strand_start",   "alignment_strand_end",      "alignment_direction",
            "alignment_length",         "alignment_num_aligned",     "alignment_num_correct",
            "alignment_num_insertions", "alignment_num_deletions",   "alignment_num_substitutions",
            "alignment_mapq",           "alignment_strand_coverage", "alignment_identity",
            "alignment_accuracy"};

    auto reads(parser.get<std::string>("reads"));
    auto separator(parser.get<std::string>("separator"));

    HtsReader reader(reads);

    auto read_group_exp_start_time = utils::get_read_group_info(reader.header, "DT");

    spdlog::debug("> input fmt: {} aligned: {}", reader.format, reader.is_aligned);
#ifndef _WIN32
    std::signal(SIGPIPE, [](int signum) { interrupt = 1; });
#endif
    std::signal(SIGINT, [](int signum) { interrupt = 1; });

    for (int col = 0; col < header.size() - 1; col++) {
        std::cout << header[col] << separator;
    }
    std::cout << header[header.size() - 1];

    if (reader.is_aligned) {
        for (int col = 0; col < aligned_header.size() - 1; col++) {
            std::cout << separator << aligned_header[col];
        }
        std::cout << separator << aligned_header[aligned_header.size() - 1];
    }

    std::cout << '\n';

    while (reader.read() && !interrupt) {
        if (reader.record->core.flag & (BAM_FSECONDARY | BAM_FSUPPLEMENTARY)) {
            continue;
        }

        auto rg_value = reader.get_tag<std::string>("RG");
        if (rg_value.length() == 0) {
            spdlog::error("> Cannot generate sequencing summary for files with no RG tags");
            return 1;
        }
        auto rg_split = rg_value.find("_");
        auto run_id = rg_value.substr(0, rg_split);
        auto model = rg_value.substr(rg_split + 1, rg_value.length());

        auto filename = reader.get_tag<std::string>("f5");
        if (filename.empty()) {
            filename = reader.get_tag<std::string>("fn");
        }
        auto read_id = bam_get_qname(reader.record);
        auto channel = reader.get_tag<int>("ch");
        auto mux = reader.get_tag<int>("mx");

        auto start_time_dt = reader.get_tag<std::string>("st");
        auto duration = reader.get_tag<float>("du");

        auto seqlen = reader.record->core.l_qseq;
        auto mean_qscore = reader.get_tag<int>("qs");

        auto num_samples = reader.get_tag<int>("ns");
        auto trim_samples = reader.get_tag<int>("ts");

        float sample_rate = num_samples / duration;
        float template_duration = (num_samples - trim_samples) / sample_rate;
        auto exp_start_dt = read_group_exp_start_time.at(rg_value);
        auto start_time = time_difference_seconds(start_time_dt, exp_start_dt);
        auto template_start_time = start_time + (duration - template_duration);

        std::cout << filename << separator << read_id << separator << run_id << separator << channel
                  << separator << mux << separator << start_time << separator << duration
                  << separator << template_start_time << separator << template_duration << separator
                  << seqlen << separator << mean_qscore;

        if (reader.is_aligned) {
            int32_t query_start = 0;
            int32_t query_end = 0;
            std::string alignment_genome = "*";
            int32_t alignment_genome_start = -1;
            int32_t alignment_genome_end = -1;
            int32_t alignment_strand_start = -1;
            int32_t alignment_strand_end = -1;
            std::string alignment_direction = "*";
            int32_t alignment_length = 0;
            int32_t alignment_mapq = 0;
            int alignment_num_aligned = 0;
            int alignment_num_correct = 0;
            int alignment_num_insertions = 0;
            int alignment_num_deletions = 0;
            int alignment_num_substitutions = 0;
            float strand_coverage = 0.0;
            float alignment_identity = 0.0;
            float alignment_accurary = 0.0;

            if (!(reader.record->core.flag & BAM_FUNMAP)) {
                alignment_mapq = static_cast<int>(reader.record->core.qual);
                alignment_genome = reader.header->target_name[reader.record->core.tid];

                alignment_genome_start = reader.record->core.pos;
                alignment_genome_end = bam_endpos(reader.record.get());
                alignment_direction = bam_is_rev(reader.record) ? "-" : "+";

                auto alignment_counts = utils::get_alignment_op_counts(reader.record.get());
                alignment_num_aligned = alignment_counts.matches;
                alignment_num_correct = alignment_counts.matches - alignment_counts.substitutions;
                alignment_num_insertions = alignment_counts.insertions;
                alignment_num_deletions = alignment_counts.deletions;
                alignment_num_substitutions = alignment_counts.substitutions;
                alignment_length = alignment_counts.matches + alignment_counts.insertions +
                                   alignment_counts.deletions;
                alignment_strand_start = alignment_counts.softclip_start;
                alignment_strand_end = seqlen - alignment_counts.softclip_end;

                strand_coverage = (alignment_strand_end - alignment_strand_start) /
                                  static_cast<float>(seqlen);
                alignment_identity =
                        alignment_num_correct / static_cast<float>(alignment_counts.matches);
                alignment_accurary = alignment_num_correct / static_cast<float>(alignment_length);
            }

            std::cout << separator << alignment_genome << separator << alignment_genome_start
                      << separator << alignment_genome_end << separator << alignment_strand_start
                      << separator << alignment_strand_end << separator << alignment_direction
                      << separator << alignment_length << separator << alignment_num_aligned
                      << separator << alignment_num_correct << separator << alignment_num_insertions
                      << separator << alignment_num_deletions << separator
                      << alignment_num_substitutions << separator << alignment_mapq << separator
                      << strand_coverage << separator << alignment_identity << separator
                      << alignment_accurary;
        }

        std::cout << '\n';
    }

    return 0;
}

}  // namespace dorado
