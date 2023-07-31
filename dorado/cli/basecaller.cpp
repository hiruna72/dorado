#include "Version.h"
#include "data_loader/DataLoader.h"
#include "nn/CRFModel.h"
#include "nn/ModBaseRunner.h"
#include "nn/ModelRunner.h"
#include "nn/Runners.h"
#include "read_pipeline/AlignerNode.h"
#include "read_pipeline/BasecallerNode.h"
#include "read_pipeline/HtsReader.h"
#include "read_pipeline/HtsWriter.h"
#include "read_pipeline/ModBaseCallerNode.h"
#include "read_pipeline/ProgressTracker.h"
#include "read_pipeline/ReadFilterNode.h"
#include "read_pipeline/ReadToBamTypeNode.h"
#include "read_pipeline/ResumeLoaderNode.h"
#include "read_pipeline/ScalerNode.h"
#include "utils/bam_utils.h"
#include "utils/basecaller_utils.h"
#include "utils/cli_utils.h"
#include "utils/log_utils.h"
#include "utils/models.h"
#include "utils/parameters.h"
#include "utils/stats.h"

#include <argparse.hpp>
#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>

namespace dorado {

using dorado::utils::default_parameters;
using namespace std::chrono_literals;

void setup(std::vector<std::string> args,
           const std::filesystem::path& model_path,
           const std::string& data_path,
           const std::string& remora_models,
           const std::string& device,
           const std::string& ref,
           size_t chunk_size,
           size_t overlap,
           size_t batch_size,
           size_t num_runners,
           size_t remora_batch_size,
           size_t num_remora_threads,
           float methylation_threshold_pct,
           HtsWriter::OutputMode output_mode,
           bool emit_moves,
           size_t max_reads,
           size_t min_qscore,
           std::string read_list_file_path,
           int32_t slow5_threads,
           int64_t slow5_batchsize,
           bool recursive_file_loading,
           int kmer_size,
           int window_size,
           uint64_t mm2_index_batch_size,
           bool skip_model_compatibility_check,
           const std::string& dump_stats_file,
           const std::string& dump_stats_filter,
           const std::string& resume_from_file,
           argparse::ArgumentParser& resume_parser) {
    torch::set_num_threads(1);

    // create modbase runners first so basecall runners can pick batch sizes based on available memory
    auto remora_runners = create_modbase_runners(
            remora_models, device, default_parameters.remora_runners_per_caller, remora_batch_size);

    auto model_config = dorado::load_crf_model_config(model_path);
    auto [runners, num_devices] =
            create_basecall_runners(model_config, device, num_runners, batch_size, chunk_size);

    // verify that all runners are using the same stride, in case we allow multiple models in future
    auto model_stride = runners.front()->model_stride();
    auto adjusted_chunk_size = runners.front()->chunk_size();
    auto adjusted_overlap = (overlap / model_stride) * model_stride;

    if (overlap != adjusted_overlap) {
        spdlog::debug("- adjusted overlap to match model stride: {} -> {}", overlap,
                      adjusted_overlap);
        overlap = adjusted_overlap;
    }

    if (!remora_runners.empty() && output_mode == HtsWriter::OutputMode::FASTQ) {
        throw std::runtime_error("Modified base models cannot be used with FASTQ output");
    }

    if (!ref.empty() && output_mode == HtsWriter::OutputMode::FASTQ) {
        throw std::runtime_error("Alignment to reference cannot be used with FASTQ output.");
    }

    std::string model_name = std::filesystem::canonical(model_path).filename().string();
    auto read_groups = DataLoader::load_read_groups(data_path, model_name, recursive_file_loading);

    auto read_list = utils::load_read_list(read_list_file_path);

    // Check sample rate of model vs data.
    auto data_sample_rate = DataLoader::get_sample_rate(data_path, recursive_file_loading);
    auto model_sample_rate = get_model_sample_rate(model_path);
    if (!skip_model_compatibility_check &&
        !sample_rates_compatible(data_sample_rate, model_sample_rate)) {
        std::stringstream err;
        err << "Sample rate for model (" << model_sample_rate << ") and data (" << data_sample_rate
            << ") are not compatible.";
        throw std::runtime_error(err.str());
    }

    size_t num_reads = DataLoader::get_num_reads(
            data_path, read_list, {} /*reads_already_processed*/, recursive_file_loading);
    num_reads = max_reads == 0 ? num_reads : std::min(num_reads, max_reads);

    bool rna = utils::is_rna_model(model_path), duplex = false;

    auto const thread_allocations = utils::default_thread_allocations(
            num_devices, !remora_runners.empty() ? num_remora_threads : 0);

    std::unique_ptr<sam_hdr_t, void (*)(sam_hdr_t*)> hdr(sam_hdr_init(), sam_hdr_destroy);
    utils::add_pg_hdr(hdr.get(), args);
    utils::add_rg_hdr(hdr.get(), read_groups);

    PipelineDescriptor pipeline_desc;
    auto hts_writer = PipelineDescriptor::InvalidNodeHandle;
    auto aligner = PipelineDescriptor::InvalidNodeHandle;
    auto converted_reads_sink = PipelineDescriptor::InvalidNodeHandle;
    std::unordered_set<std::string> reads_already_processed;
    // TODO -- refactor to avoid repeated code here.
    if (ref.empty()) {
        hts_writer = pipeline_desc.add_node<HtsWriter>(
                {}, "-", output_mode, thread_allocations.writer_threads, num_reads);
        converted_reads_sink = hts_writer;
    } else {
        hts_writer = pipeline_desc.add_node<HtsWriter>(
                {}, "-", output_mode, thread_allocations.writer_threads, num_reads);
        aligner = pipeline_desc.add_node<Aligner>({hts_writer}, ref, kmer_size, window_size,
                                                  mm2_index_batch_size,
                                                  thread_allocations.aligner_threads);
        converted_reads_sink = aligner;
    }
    auto read_converter = pipeline_desc.add_node<ReadToBamType>(
            {converted_reads_sink}, emit_moves, rna, thread_allocations.read_converter_threads,
            methylation_threshold_pct);
    auto read_filter_node = pipeline_desc.add_node<ReadFilterNode>(
            {read_converter}, min_qscore, default_parameters.min_sequence_length,
            std::unordered_set<std::string>{}, thread_allocations.read_filter_threads);

    auto basecaller_node_sink = read_filter_node;

    if (!remora_runners.empty()) {
        auto mod_base_caller_node = pipeline_desc.add_node<ModBaseCallerNode>(
                {read_filter_node}, std::move(remora_runners),
                thread_allocations.remora_threads * num_devices, model_stride, remora_batch_size);
        basecaller_node_sink = mod_base_caller_node;
    }
    const int kBatchTimeoutMS = 100;
    auto basecaller_node = pipeline_desc.add_node<BasecallerNode>(
            {basecaller_node_sink}, std::move(runners), overlap, kBatchTimeoutMS, model_name, 1000,
            "BasecallerNode", false, get_model_mean_qscore_start_pos(model_config));

    auto scaler_node =
            pipeline_desc.add_node<ScalerNode>({basecaller_node}, model_config.signal_norm_params,
                                               thread_allocations.scaler_node_threads);

    // Create the Pipeline from our description.
    std::vector<dorado::stats::StatsReporter> stats_reporters;
    auto pipeline = Pipeline::create(std::move(pipeline_desc), &stats_reporters);
    if (pipeline == nullptr) {
        spdlog::error("Failed to create pipeline");
        std::exit(EXIT_FAILURE);
    }

    // At present, header output file header writing relies on direct node method calls
    // rather than the pipeline framework.
    auto& hts_writer_ref = dynamic_cast<HtsWriter&>(pipeline->get_node_ref(hts_writer));
    if (!ref.empty()) {
        const auto& aligner_ref = dynamic_cast<Aligner&>(pipeline->get_node_ref(aligner));
        utils::add_sq_hdr(hdr.get(), aligner_ref.get_sequence_records_for_header());
    }
    hts_writer_ref.set_and_write_header(hdr.get());

    if (!resume_from_file.empty()) {
        spdlog::info("> Inspecting resume file...");
        // Turn off warning logging as header info is fetched.
        auto initial_hts_log_level = hts_get_log_level();
        hts_set_log_level(HTS_LOG_OFF);
        auto pg_keys = utils::extract_pg_keys_from_hdr(resume_from_file, {"CL"});
        hts_set_log_level(initial_hts_log_level);

        auto tokens = utils::extract_token_from_cli(pg_keys["CL"]);
        // First token is the dorado binary name. Remove that because the
        // sub parser only knows about the `basecaller` command.
        tokens.erase(tokens.begin());
        resume_parser.parse_args(tokens);
        auto resume_model_name =
                utils::extract_model_from_model_path(resume_parser.get<std::string>("model"));
        if (model_name != resume_model_name) {
            throw std::runtime_error(
                    "Resume only works if the same model is used. Resume model was " +
                    resume_model_name + " and current model is " + model_name);
        }
        // Resume functionality injects reads directly into the writer node.
        ResumeLoaderNode resume_loader(hts_writer_ref, resume_from_file);
        resume_loader.copy_completed_reads();
        reads_already_processed = resume_loader.get_processed_read_ids();
    }

    std::vector<dorado::stats::StatsCallable> stats_callables;
    ProgressTracker tracker(num_reads, duplex);
    stats_callables.push_back(
            [&tracker](const stats::NamedStats& stats) { tracker.update_progress_bar(stats); });
    constexpr auto kStatsPeriod = 100ms;
    auto stats_sampler = std::make_unique<dorado::stats::StatsSampler>(
            kStatsPeriod, stats_reporters, stats_callables);

    DataLoader loader(*pipeline, "cpu", thread_allocations.loader_threads, max_reads, read_list,
                      reads_already_processed, slow5_threads, slow5_batchsize);

    // Run pipeline.
    loader.load_reads(data_path, recursive_file_loading);

    // Wait for the pipeline to complete.  When it does, we collect
    // final stats to allow accurate summarisation.
    auto final_stats = pipeline->terminate();

    // Stop the stats sampler thread before tearing down any pipeline objects.
    stats_sampler->terminate();

    // Then update progress tracking one more time from this thread, to
    // allow accurate summarisation.
    tracker.update_progress_bar(final_stats);
    tracker.summarize();
    if (!dump_stats_file.empty()) {
        std::ofstream stats_file(dump_stats_file);
        stats_sampler->dump_stats(stats_file,
                                  dump_stats_filter.empty()
                                          ? std::nullopt
                                          : std::optional<std::regex>(dump_stats_filter));
    }
}

int basecaller(int argc, char* argv[]) {
    utils::InitLogging();

    argparse::ArgumentParser parser("dorado", DORADO_VERSION, argparse::default_arguments::help);

    parser.add_argument("model").help("the basecaller model to run.");

    parser.add_argument("data").help("the data directory.");

    parser.add_argument("-v", "--verbose").default_value(false).implicit_value(true);

    parser.add_argument("-x", "--device")
            .help("device string in format \"cuda:0,...,N\", \"cuda:all\", \"metal\", \"cpu\" "
                  "etc..")
            .default_value(default_parameters.device);

    parser.add_argument("-l", "--read-ids")
            .help("A file with a newline-delimited list of reads to basecall. If not provided, all "
                  "reads will be basecalled")
            .default_value(std::string(""));

    parser.add_argument("--resume-from")
            .help("Resume basecalling from the given HTS file. Fully written read records are not "
                  "processed again.")
            .default_value(std::string(""));

    parser.add_argument("-n", "--max-reads").default_value(0).scan<'i', int>();

    parser.add_argument("--min-qscore").default_value(0).scan<'i', int>();

    parser.add_argument("-b", "--batchsize")
            .default_value(default_parameters.batchsize)
            .scan<'i', int>()
            .help("if 0 an optimal batchsize will be selected. batchsizes are rounded to the "
                  "closest multiple of 64.");

    parser.add_argument("-c", "--chunksize")
            .default_value(default_parameters.chunksize)
            .scan<'i', int>();

    parser.add_argument("-o", "--overlap")
            .default_value(default_parameters.overlap)
            .scan<'i', int>();

    parser.add_argument("-r", "--recursive")
            .default_value(false)
            .implicit_value(true)
            .help("Recursively scan through directories to load FAST5 and POD5 files");

    parser.add_argument("--slow5_threads")
            .default_value(default_parameters.slow5_threads)
            .scan<'i', int32_t>();

    parser.add_argument("--slow5_batchsize")
            .default_value(default_parameters.slow5_batchsize)
            .scan<'i', int64_t>();

    parser.add_argument("--modified-bases")
            .nargs(argparse::nargs_pattern::at_least_one)
            .action([](const std::string& value) {
                if (std::find(modified::mods.begin(), modified::mods.end(), value) ==
                    modified::mods.end()) {
                    spdlog::error(
                            "'{}' is not a supported modification please select from {}", value,
                            std::accumulate(std::next(modified::mods.begin()), modified::mods.end(),
                                            modified::mods[0], [](std::string a, std::string b) {
                                                return a + ", " + b;
                                            }));
                    std::exit(EXIT_FAILURE);
                }
                return value;
            });

    parser.add_argument("--modified-bases-models")
            .default_value(std::string())
            .help("a comma separated list of modified base models");

    parser.add_argument("--modified-bases-threshold")
            .default_value(default_parameters.methylation_threshold)
            .scan<'f', float>()
            .help("the minimum predicted methylation probability for a modified base to be emitted "
                  "in an all-context model, [0, 1]");

    parser.add_argument("--emit-fastq")
            .help("Output in fastq format.")
            .default_value(false)
            .implicit_value(true);
    parser.add_argument("--emit-sam")
            .help("Output in SAM format.")
            .default_value(false)
            .implicit_value(true);

    parser.add_argument("--emit-moves").default_value(false).implicit_value(true);

    parser.add_argument("--reference")
            .help("Path to reference for alignment.")
            .default_value(std::string(""));
    parser.add_argument("-k")
            .help("k-mer size for alignment with minimap2 (maximum 28).")
            .default_value(15)
            .scan<'i', int>();
    parser.add_argument("-w")
            .help("minimizer window size for alignment with minimap2.")
            .default_value(10)
            .scan<'i', int>();
    parser.add_argument("-I").help("minimap2 index batch size.").default_value(std::string("16G"));

    argparse::ArgumentParser internal_parser;

    // Create a copy of the parser to use if the resume feature is enabled. Needed
    // to parse the model used for the file being resumed from. Note that this copy
    // needs to be made __before__ the parser is used.
    argparse::ArgumentParser resume_parser = parser;

    try {
        auto remaining_args = parser.parse_known_args(argc, argv);
        internal_parser = utils::parse_internal_options(remaining_args);
    } catch (const std::exception& e) {
        std::ostringstream parser_stream;
        parser_stream << parser;
        spdlog::error("{}\n{}", e.what(), parser_stream.str());
        std::exit(1);
    }

    std::vector<std::string> args(argv, argv + argc);

    if (parser.get<bool>("--verbose")) {
        utils::SetDebugLogging();
    }

    auto model = parser.get<std::string>("model");
    auto mod_bases = parser.get<std::vector<std::string>>("--modified-bases");
    auto mod_bases_models = parser.get<std::string>("--modified-bases-models");

    if (mod_bases.size() && !mod_bases_models.empty()) {
        spdlog::error(
                "only one of --modified-bases or --modified-bases-models should be specified.");
        std::exit(EXIT_FAILURE);
    } else if (mod_bases.size()) {
        std::vector<std::string> m;
        std::transform(mod_bases.begin(), mod_bases.end(), std::back_inserter(m),
                       [&model](std::string m) { return utils::get_modification_model(model, m); });

        mod_bases_models =
                std::accumulate(std::next(m.begin()), m.end(), m[0],
                                [](std::string a, std::string b) { return a + "," + b; });
    }

    auto methylation_threshold = parser.get<float>("--modified-bases-threshold");
    if (methylation_threshold < 0.f || methylation_threshold > 1.f) {
        spdlog::error("--modified-bases-threshold must be between 0 and 1.");
        std::exit(EXIT_FAILURE);
    }

    auto output_mode = HtsWriter::OutputMode::BAM;

    auto emit_fastq = parser.get<bool>("--emit-fastq");
    auto emit_sam = parser.get<bool>("--emit-sam");

    if (emit_fastq && emit_sam) {
        throw std::runtime_error("Only one of --emit-{fastq, sam} can be set (or none).");
    }

    if (emit_fastq) {
        output_mode = HtsWriter::OutputMode::FASTQ;
    } else if (emit_sam || utils::is_fd_tty(stdout)) {
        output_mode = HtsWriter::OutputMode::SAM;
    } else if (utils::is_fd_pipe(stdout)) {
        output_mode = HtsWriter::OutputMode::UBAM;
    }

    spdlog::info("> Creating basecall pipeline");

    try {
        setup(args, model, parser.get<std::string>("data"), mod_bases_models,
              parser.get<std::string>("-x"), parser.get<std::string>("--reference"),
              parser.get<int>("-c"), parser.get<int>("-o"), parser.get<int>("-b"),
              default_parameters.num_runners, default_parameters.remora_batchsize,
              default_parameters.remora_threads, methylation_threshold, output_mode,
              parser.get<bool>("--emit-moves"), parser.get<int>("--max-reads"),
              parser.get<int>("--min-qscore"), parser.get<std::string>("--read-ids"), parser.get<int32_t>("--slow5_threads"), parser.get<int64_t>("--slow5_batchsize"),
              parser.get<bool>("--recursive"), parser.get<int>("k"), parser.get<int>("w"),
              utils::parse_string_to_size(parser.get<std::string>("I")),
              internal_parser.get<bool>("--skip-model-compatibility-check"),
              internal_parser.get<std::string>("--dump_stats_file"),
              internal_parser.get<std::string>("--dump_stats_filter"),
              parser.get<std::string>("--resume-from"), resume_parser);
    } catch (const std::exception& e) {
        spdlog::error("{}", e.what());
        return 1;
    }

    spdlog::info("> Finished");
    return 0;
}

}  // namespace dorado
