#include "alignment/minimap2_args.h"
#include "api/pipeline_creation.h"
#include "api/runner_creation.h"
#include "basecall/CRFModelConfig.h"
#include "basecall_output_args.h"
#include "cli/cli_utils.h"
#include "cli/model_resolution.h"
#include "data_loader/DataLoader.h"
#include "demux/adapter_info.h"
#include "demux/barcoding_info.h"
#include "demux/parse_custom_kit.h"
#include "demux/parse_custom_sequences.h"
#include "dorado_version.h"
#include "model_downloader/model_downloader.h"
#include "models/kits.h"
#include "models/model_complex.h"
#include "models/models.h"
#include "poly_tail/poly_tail_calculator_selector.h"
#include "read_pipeline/AdapterDetectorNode.h"
#include "read_pipeline/AlignerNode.h"
#include "read_pipeline/BarcodeClassifierNode.h"
#include "read_pipeline/DefaultClientInfo.h"
#include "read_pipeline/HtsWriter.h"
#include "read_pipeline/PolyACalculatorNode.h"
#include "read_pipeline/ProgressTracker.h"
#include "read_pipeline/ReadFilterNode.h"
#include "read_pipeline/ReadToBamTypeNode.h"
#include "read_pipeline/ResumeLoader.h"
#include "read_pipeline/TrimmerNode.h"
#include "torch_utils/auto_detect_device.h"
#include "utils/SampleSheet.h"
#include "utils/arg_parse_ext.h"
#include "utils/bam_utils.h"
#include "utils/barcode_kits.h"
#include "utils/basecaller_utils.h"
#include "utils/string_utils.h"

#include <argparse.hpp>
#include <cxxpool.h>

#include <string>
#if DORADO_CUDA_BUILD
#include "torch_utils/cuda_utils.h"
#endif
#include "torch_utils/torch_utils.h"
#include "utils/fs_utils.h"
#include "utils/log_utils.h"
#include "utils/parameters.h"
#include "utils/stats.h"
#include "utils/sys_stats.h"
#include "utils/tty_utils.h"

#include <htslib/sam.h>
#include <spdlog/spdlog.h>
#include <torch/utils.h>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <thread>

using dorado::utils::default_parameters;
using OutputMode = dorado::utils::HtsFile::OutputMode;
using namespace std::chrono_literals;
using namespace dorado::models;
using namespace dorado::model_resolution;
namespace fs = std::filesystem;

namespace dorado {

namespace {

const barcode_kits::KitInfo& get_barcode_kit_info(const std::string& kit_name) {
    const auto kit_info = barcode_kits::get_kit_info(kit_name);
    if (!kit_info) {
        spdlog::error(
                "{} is not a valid barcode kit name. Please run the help "
                "command to find out available barcode kits.",
                kit_name);
        std::exit(EXIT_FAILURE);
    }
    return *kit_info;
}

std::pair<std::string, barcode_kits::KitInfo> get_custom_barcode_kit_info(
        const std::string& custom_kit_file) {
    auto custom_kit_info = demux::parse_custom_arrangement(custom_kit_file);
    if (!custom_kit_info) {
        spdlog::error("Unable to load custom barcode arrangement file: {}", custom_kit_file);
        std::exit(EXIT_FAILURE);
    }
    return *custom_kit_info;
}

void set_basecaller_params(const argparse::ArgumentParser& arg,
                           basecall::CRFModelConfig& model_config,
                           const std::string& device) {
    model_config.basecaller.update(basecall::BasecallerParams::Priority::CLI_ARG,
                                   cli::get_optional_argument<int>("--chunksize", arg),
                                   cli::get_optional_argument<int>("--overlap", arg),
                                   cli::get_optional_argument<int>("--batchsize", arg));

    if (device == "cpu" && model_config.basecaller.batch_size() == 0) {
        // Force the batch size to 128
        // TODO: This is tuned for LSTM models - investigate Tx
        model_config.basecaller.set_batch_size(128);
    }
#if DORADO_METAL_BUILD
    else if (device == "metal" && model_config.is_tx_model() &&
             model_config.basecaller.batch_size() == 0) {
        model_config.basecaller.set_batch_size(32);
    }
#endif

    model_config.normalise_basecaller_params();
}

}  // namespace

void set_dorado_basecaller_args(utils::arg_parse::ArgParser& parser, int& verbosity) {
    parser.visible.add_argument("model").help(
            "Model selection {fast,hac,sup}@v{version} for automatic model selection including "
            "modbases, or path to existing model directory.");
    parser.visible.add_argument("data").help("The data directory or file (POD5/FAST5/SLOW5/BLOW5 format).");
    {
        // Default "Optional arguments" group
        parser.visible.add_argument("-v", "--verbose")
                .default_value(false)
                .implicit_value(true)
                .nargs(0)
                .action([&](const auto&) { ++verbosity; })
                .append();
        cli::add_device_arg(parser);
        parser.visible.add_argument("--models-directory")
                .default_value(std::string("."))
                .help("Optional directory to search for existing models or download new models "
                      "into.");
        parser.visible.add_argument("--bed-file")
                .help("Optional bed-file. If specified, overlaps between the alignments and "
                      "bed-file entries will be counted, and recorded in BAM output using the 'bh' "
                      "read tag.")
                .default_value(std::string(""));
    }
    {
        parser.visible.add_group("Input data arguments");
        parser.visible.add_argument("-r", "--recursive")
                .default_value(false)
                .implicit_value(true)
                .help("Recursively scan through directories to load FAST5 and POD5 and SLOW5/BLOW5 files.");

        parser.visible.add_argument("--slow5_threads")
            .default_value(default_parameters.slow5_threads)
            .scan<'i', int32_t>()
            .help("Number of slow5 threads used to read SLOW5/BLOW5");

        parser.visible.add_argument("--slow5_batchsize")
            .default_value(default_parameters.slow5_batchsize)
            .scan<'i', int64_t>()
            .help("Batchsize used to read SLOW5/BLOW5");

        // parser.visible.add_argument("-l", "--read-ids")
        //         .help("A file with a newline-delimited list of reads to basecall. If not provided, "
        //               "all reads will be basecalled.")
        //         .default_value(std::string(""));
        parser.visible.add_argument("-n", "--max-reads")
                .help("Limit the number of reads to be basecalled.")
                .default_value(0)
                .scan<'i', int>();
        parser.visible.add_argument("--resume-from")
                .help("Resume basecalling from the given HTS file. Fully written read records are "
                      "not processed again.")
                .default_value(std::string(""));
    }
    {
        parser.visible.add_group("Output arguments");
        parser.visible.add_argument("--min-qscore")
                .help("Discard reads with mean Q-score below this threshold.")
                .default_value(0)
                .scan<'i', int>();
        parser.visible.add_argument("--emit-moves")
                .help("Write the move table to the 'mv' tag.")
                .default_value(false)
                .implicit_value(true);
        cli::add_basecaller_output_arguments(parser);
    }
    {
        parser.visible.add_group("Alignment arguments");
        parser.visible.add_argument("--reference")
                .help("Path to reference for alignment.")
                .default_value(std::string(""));
        alignment::mm2::add_options_string_arg(parser);
    }
    {
        const std::string mods_codes = utils::join(models::modified_model_variants(), ", ");
        parser.visible.add_group("Modified model arguments");
        parser.visible.add_argument("--modified-bases")
                .help("A space separated list of modified base codes. Choose from: " + mods_codes +
                      ".")
                .nargs(argparse::nargs_pattern::at_least_one)
                .action([mods_codes](const std::string& value) {
                    const auto& mods = models::modified_model_variants();
                    if (std::find(mods.begin(), mods.end(), value) == mods.end()) {
                        spdlog::error("'{}' is not a supported modification please select from {}",
                                      value, mods_codes);
                        std::exit(EXIT_FAILURE);
                    }
                    return value;
                });
        parser.visible.add_argument("--modified-bases-models")
                .default_value(std::string())
                .help("A comma separated list of modified base model paths.");
        parser.visible.add_argument("--modified-bases-threshold")
                .default_value(default_parameters.methylation_threshold)
                .scan<'f', float>()
                .help("The minimum predicted methylation probability for a modified base to be "
                      "emitted in an all-context model, [0, 1].");
    }
    {
        parser.visible.add_group("Barcoding arguments");
        parser.visible.add_argument("--kit-name")
                .help("Enable barcoding with the provided kit name. Choose from: " +
                      dorado::barcode_kits::barcode_kits_list_str() + ".")
                .default_value(std::string{});
        parser.visible.add_argument("--sample-sheet")
                .help("Path to the sample sheet to use.")
                .default_value(std::string(""));
        parser.visible.add_argument("--barcode-both-ends")
                .help("Require both ends of a read to be barcoded for a double ended barcode.")
                .default_value(false)
                .implicit_value(true);
        parser.visible.add_argument("--barcode-arrangement")
                .help("Path to file with custom barcode arrangement.");
        parser.visible.add_argument("--barcode-sequences")
                .help("Path to file with custom barcode sequences.");
        parser.visible.add_argument("--primer-sequences")
                .help("Path to file with custom primer sequences.")
                .default_value(std::nullopt);
    }
    {
        parser.visible.add_group("Trimming arguments");
        parser.visible.add_argument("--no-trim")
                .help("Skip trimming of barcodes, adapters, and primers. If option is not chosen, "
                      "trimming of all three is enabled.")
                .default_value(false)
                .implicit_value(true);
        parser.visible.add_argument("--trim")
                .help("Specify what to trim. Options are 'none', 'all', 'adapters', and 'primers'. "
                      "Default behaviour is to trim all detected adapters, primers, or barcodes. "
                      "Choose 'adapters' to just trim adapters. The 'primers' choice will trim "
                      "adapters and primers, but not barcodes. The 'none' choice is equivelent to "
                      "using --no-trim. Note that this only applies to DNA. "
                      "RNA adapters are always trimmed.")
                .default_value(std::string(""));
        parser.hidden.add_argument("--rna-adapters")
                .help("Force use of RNA adapters.")
                .implicit_value(true)
                .default_value(false);
    }
    {
        parser.visible.add_group("Poly(A) arguments");
        parser.visible.add_argument("--estimate-poly-a")
                .help("Estimate poly(A)/poly(T) tail lengths (beta feature). Primarily meant for "
                      "cDNA and dRNA use cases.")
                .default_value(false)
                .implicit_value(true);
        parser.visible.add_argument("--poly-a-config")
                .help("Configuration file for poly(A) estimation to change default behaviours")
                .default_value(std::string(""));
    }
    {
        parser.visible.add_group("Advanced arguments");
        parser.visible.add_argument("-b", "--batchsize")
                .help("The number of chunks in a batch. If 0 an optimal batchsize will be "
                      "selected.")
                .default_value(default_parameters.batchsize)
                .scan<'i', int>();
        parser.visible.add_argument("-c", "--chunksize")
                .help("The number of samples in a chunk.")
                .default_value(default_parameters.chunksize)
                .scan<'i', int>();
        parser.visible.add_argument("-o", "--overlap")
                .help("The number of samples overlapping neighbouring chunks.")
                .default_value(default_parameters.overlap)
                .scan<'i', int>();
    }
    cli::add_internal_arguments(parser);
}

void setup(const std::vector<std::string>& args,
           const basecall::CRFModelConfig& model_config,
           const std::string& data_path,
           const std::vector<fs::path>& remora_models,
           const std::string& device,
           const std::string& ref,
           const std::string& bed,
           size_t num_runners,
           size_t remora_batch_size,
           size_t num_remora_threads,
           float methylation_threshold_pct,
           std::unique_ptr<utils::HtsFile> hts_file,
           bool emit_moves,
           size_t max_reads,
           size_t min_qscore,
           const std::string& read_list_file_path,
           int32_t slow5_threads,
           int64_t slow5_batchsize,
           bool recursive_file_loading,
           const alignment::Minimap2Options& aligner_options,
           bool skip_model_compatibility_check,
           const std::string& dump_stats_file,
           const std::string& dump_stats_filter,
           bool run_batchsize_benchmarks,
           bool emit_batchsize_benchmarks,
           const std::string& resume_from_file,
           bool estimate_poly_a,
           const std::string& polya_config,
           const ModelComplex& model_complex,
           std::shared_ptr<const dorado::demux::BarcodingInfo> barcoding_info,
           std::shared_ptr<const dorado::demux::AdapterInfo> adapter_info,
           std::unique_ptr<const utils::SampleSheet> sample_sheet) {
    spdlog::debug(model_config.to_string());
    const std::string model_name = models::extract_model_name_from_path(model_config.model_path);
    const std::string modbase_model_names = models::extract_model_names_from_paths(remora_models);

    if (!DataLoader::is_read_data_present(data_path, recursive_file_loading)) {
        std::string err = "No POD5 or FAST5 data found in path: " + data_path;
        throw std::runtime_error(err);
    }

    auto read_list = utils::load_read_list(read_list_file_path);
    size_t num_reads = DataLoader::get_num_reads(
            data_path, read_list, {} /*reads_already_processed*/, recursive_file_loading);
    // if (num_reads == 0) {
    //     spdlog::error("No POD5 or FAST5 reads found in path: " + data_path);
    //     std::exit(EXIT_FAILURE);
    // }
    num_reads = max_reads == 0 ? num_reads : std::min(num_reads, max_reads);

    // Sampling rate is checked by ModelComplexSearch when a complex is given, only test for a path
    if (model_complex.is_path() && !skip_model_compatibility_check) {
        const auto model_sample_rate = model_config.sample_rate < 0
                                               ? get_sample_rate_by_model_name(model_name)
                                               : model_config.sample_rate;
        bool inspect_ok = true;
        models::SamplingRate data_sample_rate = 0;
        try {
            data_sample_rate = DataLoader::get_sample_rate(data_path, recursive_file_loading);
        } catch (const std::exception& e) {
            inspect_ok = false;
            spdlog::warn(
                    "Could not check that model sampling rate and data sampling rate match. "
                    "Proceed with caution. Reason: {}",
                    e.what());
        }
        if (inspect_ok) {
            check_sampling_rates_compatible(model_sample_rate, data_sample_rate);
        }
    }

    if (is_rna_model(model_config)) {
        spdlog::info(
                " - BAM format does not support `U`, so RNA output files will include `T` instead "
                "of `U` for all file types.");
    }

    const bool enable_aligner = !ref.empty();

#if DORADO_CUDA_BUILD
    auto initial_device_info = utils::get_cuda_device_info(device, false);
#endif

    // create modbase runners first so basecall runners can pick batch sizes based on available memory
    auto remora_runners = api::create_modbase_runners(
            remora_models, device, default_parameters.mod_base_runners_per_caller,
            remora_batch_size);

    std::vector<basecall::RunnerPtr> runners;
    size_t num_devices = 0;
#if DORADO_CUDA_BUILD
    if (device != "cpu") {
        // Iterate over the separate devices to create the basecall runners.
        // We may have multiple GPUs with different amounts of free memory left after the modbase runners were created.
        // This allows us to set a different memory_limit_fraction in case we have a heterogeneous GPU setup
        auto updated_device_info = utils::get_cuda_device_info(device, false);
        std::vector<std::pair<std::string, float>> gpu_fractions;
        for (size_t i = 0; i < updated_device_info.size(); ++i) {
            auto device_id = "cuda:" + std::to_string(updated_device_info[i].device_id);
            auto fraction = static_cast<float>(updated_device_info[i].free_mem) /
                            static_cast<float>(initial_device_info[i].free_mem);
            gpu_fractions.push_back(std::make_pair(device_id, fraction));
        }

        cxxpool::thread_pool pool{gpu_fractions.size()};
        struct BasecallerRunners {
            std::vector<dorado::basecall::RunnerPtr> runners;
            size_t num_devices{};
        };

        std::vector<std::future<BasecallerRunners>> futures;
        auto create_runners = [&](const std::string& device_id, float fraction) {
            BasecallerRunners basecaller_runners;
            std::tie(basecaller_runners.runners, basecaller_runners.num_devices) =
                    api::create_basecall_runners(
                            {model_config, device_id, fraction, api::PipelineType::simplex, 0.f,
                             run_batchsize_benchmarks, emit_batchsize_benchmarks},
                            num_runners, 0);
            return basecaller_runners;
        };

        futures.reserve(gpu_fractions.size());
        for (const auto& [device_id, fraction] : gpu_fractions) {
            futures.push_back(pool.push(create_runners, std::cref(device_id), fraction));
        }

        for (auto& future : futures) {
            auto data = future.get();
            runners.insert(runners.end(), std::make_move_iterator(data.runners.begin()),
                           std::make_move_iterator(data.runners.end()));
            num_devices += data.num_devices;
        }

        if (num_devices == 0) {
            throw std::runtime_error("CUDA device requested but no devices found.");
        }
    } else
#endif
    {
        std::tie(runners, num_devices) = api::create_basecall_runners(
                {model_config, device, 1.f, api::PipelineType::simplex, 0.f,
                 run_batchsize_benchmarks, emit_batchsize_benchmarks},
                num_runners, 0);
    }

    auto read_groups = DataLoader::load_read_groups(data_path, model_name, modbase_model_names,
                                                    recursive_file_loading);

    const bool adapter_trimming_enabled =
            (adapter_info && (adapter_info->trim_adapters || adapter_info->trim_primers));
    const auto thread_allocations = utils::default_thread_allocations(
            int(num_devices), !remora_runners.empty() ? int(num_remora_threads) : 0, enable_aligner,
            barcoding_info != nullptr, adapter_trimming_enabled);

    SamHdrPtr hdr(sam_hdr_init());
    cli::add_pg_hdr(hdr.get(), "basecaller", args, device);

    if (barcoding_info) {
        std::unordered_map<std::string, std::string> custom_barcodes{};
        if (barcoding_info->custom_seqs) {
            custom_barcodes = demux::parse_custom_sequences(*barcoding_info->custom_seqs);
        }
        if (barcoding_info->custom_kit) {
            auto [kit_name, kit_info] = get_custom_barcode_kit_info(*barcoding_info->custom_kit);
            utils::add_rg_headers_with_barcode_kit(hdr.get(), read_groups, kit_name, kit_info,
                                                   custom_barcodes, sample_sheet.get());
        } else {
            const auto& kit_info = get_barcode_kit_info(barcoding_info->kit_name);
            utils::add_rg_headers_with_barcode_kit(hdr.get(), read_groups, barcoding_info->kit_name,
                                                   kit_info, custom_barcodes, sample_sheet.get());
        }
    } else {
        utils::add_rg_headers(hdr.get(), read_groups);
    }

    hts_file->set_num_threads(thread_allocations.writer_threads);

    PipelineDescriptor pipeline_desc;
    std::string gpu_names{};
#if DORADO_CUDA_BUILD
    gpu_names = utils::get_cuda_gpu_names(device);
#endif
    auto hts_writer = pipeline_desc.add_node<HtsWriter>({}, *hts_file, gpu_names);
    auto aligner = PipelineDescriptor::InvalidNodeHandle;
    auto current_sink_node = hts_writer;
    if (enable_aligner) {
        auto index_file_access = std::make_shared<alignment::IndexFileAccess>();
        auto bed_file_access = std::make_shared<alignment::BedFileAccess>();
        if (!bed.empty()) {
            if (!bed_file_access->load_bedfile(bed)) {
                throw std::runtime_error("Could not load bed-file " + bed);
            }
        }
        aligner = pipeline_desc.add_node<AlignerNode>({current_sink_node}, index_file_access,
                                                      bed_file_access, ref, bed, aligner_options,
                                                      thread_allocations.aligner_threads);
        current_sink_node = aligner;
    }
    current_sink_node = pipeline_desc.add_node<ReadToBamTypeNode>(
            {current_sink_node}, emit_moves, thread_allocations.read_converter_threads,
            methylation_threshold_pct, std::move(sample_sheet), 1000);
    if ((barcoding_info && barcoding_info->trim) || adapter_trimming_enabled) {
        current_sink_node = pipeline_desc.add_node<TrimmerNode>({current_sink_node}, 1);
    }

    const bool is_rna_adapter = is_rna_model(model_config) &&
                                (adapter_info->rna_adapters ||
                                 (barcoding_info && (!barcoding_info->kit_name.empty() ||
                                                     barcoding_info->custom_kit.has_value())));

    auto client_info = std::make_shared<DefaultClientInfo>();
    client_info->contexts().register_context<const demux::AdapterInfo>(std::move(adapter_info));

    if (estimate_poly_a) {
        auto poly_tail_calc_selector =
                std::make_shared<const poly_tail::PolyTailCalculatorSelector>(
                        polya_config, is_rna_model(model_config), is_rna_adapter);
        client_info->contexts().register_context<const poly_tail::PolyTailCalculatorSelector>(
                std::move(poly_tail_calc_selector));
        current_sink_node = pipeline_desc.add_node<PolyACalculatorNode>(
                {current_sink_node}, std::thread::hardware_concurrency(), 1000);
    }
    if (barcoding_info) {
        client_info->contexts().register_context<const demux::BarcodingInfo>(
                std::move(barcoding_info));
        current_sink_node = pipeline_desc.add_node<BarcodeClassifierNode>(
                {current_sink_node}, thread_allocations.barcoder_threads);
    }
    if (adapter_trimming_enabled) {
        current_sink_node = pipeline_desc.add_node<AdapterDetectorNode>(
                {current_sink_node}, thread_allocations.adapter_threads);
    }

    current_sink_node = pipeline_desc.add_node<ReadFilterNode>(
            {current_sink_node}, min_qscore, default_parameters.min_sequence_length,
            std::unordered_set<std::string>{}, thread_allocations.read_filter_threads);

    auto mean_qscore_start_pos = model_config.mean_qscore_start_pos;

    api::create_simplex_pipeline(
            pipeline_desc, std::move(runners), std::move(remora_runners), mean_qscore_start_pos,
            thread_allocations.scaler_node_threads, true /* Enable read splitting */,
            thread_allocations.splitter_node_threads, thread_allocations.remora_threads,
            current_sink_node, PipelineDescriptor::InvalidNodeHandle);

    // Create the Pipeline from our description.
    std::vector<dorado::stats::StatsReporter> stats_reporters{dorado::stats::sys_stats_report};
    auto pipeline = Pipeline::create(std::move(pipeline_desc), &stats_reporters);
    if (pipeline == nullptr) {
        spdlog::error("Failed to create pipeline");
        std::exit(EXIT_FAILURE);
    }

    // At present, header output file header writing relies on direct node method calls
    // rather than the pipeline framework.
    auto& hts_writer_ref = dynamic_cast<HtsWriter&>(pipeline->get_node_ref(hts_writer));
    if (enable_aligner) {
        const auto& aligner_ref = dynamic_cast<AlignerNode&>(pipeline->get_node_ref(aligner));
        utils::add_sq_hdr(hdr.get(), aligner_ref.get_sequence_records_for_header());
    }
    hts_file->set_header(hdr.get());

    std::unordered_set<std::string> reads_already_processed;
    if (!resume_from_file.empty()) {
        spdlog::info("> Inspecting resume file...");
        // Turn off warning logging as header info is fetched.
        auto initial_hts_log_level = hts_get_log_level();
        hts_set_log_level(HTS_LOG_OFF);
        auto pg_keys =
                utils::extract_pg_keys_from_hdr(resume_from_file, {"CL"}, "ID", "basecaller");
        hts_set_log_level(initial_hts_log_level);

        auto tokens = cli::extract_token_from_cli(pg_keys["CL"]);
        // First token is the dorado binary name. Remove that because the
        // sub parser only knows about the `basecaller` command.
        tokens.erase(tokens.begin());

        // Create a new basecaller parser to parse the resumed basecaller CLI string
        utils::arg_parse::ArgParser resume_parser("dorado");
        int verbosity = 0;
        set_dorado_basecaller_args(resume_parser, verbosity);
        resume_parser.visible.parse_known_args(tokens);

        const std::string model_arg = resume_parser.visible.get<std::string>("model");
        const ModelComplex resume_model_complex = ModelComplexParser::parse(model_arg);

        if (resume_model_complex.is_path()) {
            // If the model selection is a path, check it exists and matches
            const auto resume_model_name =
                    models::extract_model_name_from_path(fs::path(model_arg));
            if (model_name != resume_model_name) {
                throw std::runtime_error(
                        "Resume only works if the same model is used. Resume model was " +
                        resume_model_name + " and current model is " + model_name);
            }
        } else if (resume_model_complex != model_complex) {
            throw std::runtime_error(
                    "Resume only works if the same model is used. Resume model complex was " +
                    resume_model_complex.raw + " and current model is " + model_complex.raw);
        }

        // Resume functionality injects reads directly into the writer node.
        ResumeLoader resume_loader(hts_writer_ref, resume_from_file);
        resume_loader.copy_completed_reads();
        reads_already_processed = resume_loader.get_processed_read_ids();
    }

    // If we're doing alignment, post-processing takes longer due to bam file sorting.
    float post_processing_percentage = (hts_file->finalise_is_noop() || ref.empty()) ? 0.0f : 0.5f;

    ProgressTracker tracker(int(num_reads), false, post_processing_percentage);
    tracker.set_description("Basecalling");

    std::vector<dorado::stats::StatsCallable> stats_callables;
    stats_callables.push_back(
            [&tracker](const stats::NamedStats& stats) { tracker.update_progress_bar(stats); });
    constexpr auto kStatsPeriod = 100ms;
    const size_t max_stats_records = static_cast<size_t>(dump_stats_file.empty() ? 0 : 100000);
    auto stats_sampler = std::make_unique<dorado::stats::StatsSampler>(
            kStatsPeriod, stats_reporters, stats_callables, max_stats_records);

    DataLoader loader(*pipeline, "cpu", thread_allocations.loader_threads, max_reads, read_list,
                      reads_already_processed, slow5_threads, slow5_batchsize);

    auto func = [client_info](ReadCommon& read) { read.client_info = client_info; };
    loader.add_read_initialiser(func);

    // Run pipeline.
    loader.load_reads(data_path, recursive_file_loading, ReadOrder::UNRESTRICTED);

    // Wait for the pipeline to complete.  When it does, we collect
    // final stats to allow accurate summarisation.
    auto final_stats = pipeline->terminate(DefaultFlushOptions());

    // Stop the stats sampler thread before tearing down any pipeline objects.
    // Then update progress tracking one more time from this thread, to
    // allow accurate summarisation.
    stats_sampler->terminate();
    tracker.update_progress_bar(final_stats);

    // Report progress during output file finalisation.
    tracker.set_description("Sorting output files");
    hts_file->finalise([&](size_t progress) {
        tracker.update_post_processing_progress(static_cast<float>(progress));
    });

    // Give the user a nice summary.
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
    utils::set_torch_allocator_max_split_size();
    utils::make_torch_deterministic();
    torch::set_num_threads(1);

    utils::arg_parse::ArgParser parser("dorado");
    int verbosity = 0;
    set_dorado_basecaller_args(parser, verbosity);

    std::vector<std::string> args_excluding_mm2_opts{};
    auto mm2_option_string = alignment::mm2::extract_options_string_arg({argv, argv + argc},
                                                                        args_excluding_mm2_opts);

    try {
        utils::arg_parse::parse(parser, args_excluding_mm2_opts);
    } catch (const std::exception& e) {
        std::ostringstream parser_stream;
        parser_stream << parser.visible;
        spdlog::error("{}\n{}", e.what(), parser_stream.str());
        return EXIT_FAILURE;
    }

    std::vector<std::string> args(argv, argv + argc);

    if (parser.visible.get<bool>("--verbose")) {
        utils::SetVerboseLogging(static_cast<dorado::utils::VerboseLogLevel>(verbosity));
    }

    const auto model_arg = parser.visible.get<std::string>("model");
    const auto data = parser.visible.get<std::string>("data");
    const auto recursive = parser.visible.get<bool>("--recursive");
    const auto mod_bases = parser.visible.get<std::vector<std::string>>("--modified-bases");
    const auto mod_bases_models = parser.visible.get<std::string>("--modified-bases-models");

    const ModelComplex model_complex = parse_model_argument(model_arg);

    if (!mods_model_arguments_valid(model_complex, mod_bases, mod_bases_models)) {
        return EXIT_FAILURE;
    }

    auto methylation_threshold = parser.visible.get<float>("--modified-bases-threshold");
    if (methylation_threshold < 0.f || methylation_threshold > 1.f) {
        spdlog::error("--modified-bases-threshold must be between 0 and 1.");
        return EXIT_FAILURE;
    }

    if (parser.visible.get<std::string>("--reference").empty() &&
        !parser.visible.get<std::string>("--bed-file").empty()) {
        spdlog::error("--bed-file cannot be used without --reference.");
        return EXIT_FAILURE;
    }

    auto device{parser.visible.get<std::string>("-x")};
    if (!cli::validate_device_string(device)) {
        return EXIT_FAILURE;
    }
    if (device == cli::AUTO_DETECT_DEVICE) {
        device = utils::get_auto_detected_device();
    }

    auto hts_file = cli::extract_hts_file(parser);
    if (!hts_file) {
        return EXIT_FAILURE;
    }
    if (hts_file->get_output_mode() == OutputMode::FASTQ) {
        if (model_complex.has_mods_variant() || !mod_bases.empty() || !mod_bases_models.empty()) {
            spdlog::error(
                    "--emit-fastq cannot be used with modbase models as FASTQ cannot store modbase "
                    "results.");
            return EXIT_FAILURE;
        }
    }

    bool no_trim_barcodes = false, no_trim_primers = false, no_trim_adapters = false;
    auto trim_options = parser.visible.get<std::string>("--trim");
    if (parser.visible.get<bool>("--no-trim")) {
        if (!trim_options.empty()) {
            spdlog::error("Only one of --no-trim and --trim can be used.");
            return EXIT_FAILURE;
        }
        no_trim_barcodes = no_trim_primers = no_trim_adapters = true;
    }
    if (trim_options == "none") {
        no_trim_barcodes = no_trim_primers = no_trim_adapters = true;
    } else if (trim_options == "primers") {
        no_trim_barcodes = true;
    } else if (trim_options == "adapters") {
        no_trim_barcodes = no_trim_primers = true;
    } else if (!trim_options.empty() && trim_options != "all") {
        spdlog::error("Unsupported --trim value '{}'.", trim_options);
        return EXIT_FAILURE;
    }

    std::string polya_config = "";
    if (parser.visible.get<bool>("--estimate-poly-a")) {
        polya_config = parser.visible.get<std::string>("--poly-a-config");
    }

    if (parser.visible.is_used("--kit-name") && parser.visible.is_used("--barcode-arrangement")) {
        spdlog::error(
                "--kit-name and --barcode-arrangement cannot be used together. Please provide only "
                "one.");
        return EXIT_FAILURE;
    }

    std::shared_ptr<demux::BarcodingInfo> barcoding_info{};
    std::unique_ptr<const utils::SampleSheet> sample_sheet{};
    if (parser.visible.is_used("--kit-name") || parser.visible.is_used("--barcode-arrangement")) {
        barcoding_info = std::make_shared<demux::BarcodingInfo>();
        barcoding_info->kit_name = parser.visible.get<std::string>("--kit-name");
        barcoding_info->barcode_both_ends = parser.visible.get<bool>("--barcode-both-ends");
        barcoding_info->trim = !no_trim_barcodes;
        barcoding_info->custom_kit = parser.visible.present<std::string>("--barcode-arrangement");
        barcoding_info->custom_seqs = parser.visible.present<std::string>("--barcode-sequences");

        auto barcode_sample_sheet = parser.visible.get<std::string>("--sample-sheet");
        if (!barcode_sample_sheet.empty()) {
            sample_sheet = std::make_unique<const utils::SampleSheet>(barcode_sample_sheet, false);
            barcoding_info->allowed_barcodes = sample_sheet->get_barcode_values();
        }
    }

    std::optional<std::string> custom_primer_file = std::nullopt;
    if (parser.visible.is_used("--primer-sequences")) {
        custom_primer_file = parser.visible.get<std::string>("--primer-sequences");
    }

    auto adapter_info = std::make_shared<demux::AdapterInfo>();
    adapter_info->trim_adapters = !no_trim_adapters;
    adapter_info->trim_primers = !no_trim_primers;
    adapter_info->custom_seqs = custom_primer_file;
    adapter_info->rna_adapters = parser.hidden.get<bool>("--rna-adapters");

    fs::path model_path;
    std::vector<fs::path> mods_model_paths;

    const auto model_directory = model_resolution::get_models_directory(parser.visible);
    model_downloader::ModelDownloader downloader(model_directory);

    if (model_complex.is_path()) {
        model_path = fs::path(model_arg);

        if (!fs::exists(model_path)) {
            spdlog::error(
                    "Model path does not exist at: '{}' - Please download the model or use a model "
                    "complex",
                    model_arg);
            return EXIT_FAILURE;
        }

        mods_model_paths = model_resolution::get_non_complex_mods_models(
                model_path, mod_bases, mod_bases_models, downloader);
    } else {
        const auto chemistry = DataLoader::get_unique_sequencing_chemisty(data, recursive);
        const auto model_search = models::ModelComplexSearch(model_complex, chemistry, true);
        try {
            model_path = downloader.get(model_search.simplex(), "simplex");
            if (model_complex.has_mods_variant()) {
                // Get mods models from complex - we assert above that there's only one method
                mods_model_paths = downloader.get(model_search.mods(), "mods");
            } else {
                // Get mods models from args
                mods_model_paths = model_resolution::get_non_complex_mods_models(
                        model_path, mod_bases, mod_bases_models, downloader);
            }
        } catch (std::exception& e) {
            spdlog::error(e.what());
            utils::clean_temporary_models(downloader.temporary_models());
            return EXIT_FAILURE;
        }
    }

    auto model_config = basecall::load_crf_model_config(model_path);
    set_basecaller_params(parser.visible, model_config, device);

    spdlog::info("> Creating basecall pipeline");

    std::string err_msg{};
    auto minimap_options = alignment::mm2::try_parse_options(mm2_option_string, err_msg);
    if (!minimap_options) {
        spdlog::error("{}\n{}", err_msg, alignment::mm2::get_help_message());
        return EXIT_FAILURE;
    }

    // Force on running of batchsize benchmarks if emission is on
    bool run_batchsize_benchmarks = parser.hidden.get<bool>("--emit-batchsize-benchmarks") ||
                                    parser.hidden.get<bool>("--run-batchsize-benchmarks");

    try {
        setup(args, model_config, data, mods_model_paths, device,
              parser.visible.get<std::string>("--reference"),
              parser.visible.get<std::string>("--bed-file"), default_parameters.num_runners,
              default_parameters.remora_batchsize, default_parameters.remora_threads,
              methylation_threshold, std::move(hts_file), parser.visible.get<bool>("--emit-moves"),
              parser.visible.get<int>("--max-reads"), parser.visible.get<int>("--min-qscore"),
              "", parser.visible.get<int32_t>("--slow5_threads"), parser.visible.get<int64_t>("--slow5_batchsize"), recursive, *minimap_options,
              parser.hidden.get<bool>("--skip-model-compatibility-check"),
              parser.hidden.get<std::string>("--dump_stats_file"),
              parser.hidden.get<std::string>("--dump_stats_filter"), run_batchsize_benchmarks,
              parser.hidden.get<bool>("--emit-batchsize-benchmarks"),
              parser.visible.get<std::string>("--resume-from"),
              parser.visible.get<bool>("--estimate-poly-a"), polya_config, model_complex,
              std::move(barcoding_info), std::move(adapter_info), std::move(sample_sheet));
    } catch (const std::exception& e) {
        spdlog::error("{}", e.what());
        utils::clean_temporary_models(downloader.temporary_models());
        return EXIT_FAILURE;
    }

    utils::clean_temporary_models(downloader.temporary_models());
    spdlog::info("> Finished");
    return EXIT_SUCCESS;
}

}  // namespace dorado
