#include "TestUtils.h"
#include "read_pipeline/HtsReader.h"
#include "utils/bam_utils.h"
#include "utils/barcode_kits.h"

#include <catch2/catch.hpp>
#include <htslib/sam.h>

#include <filesystem>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#define TEST_GROUP "[bam_utils]"

namespace fs = std::filesystem;
using namespace dorado;

namespace {
class WrappedKString {
    kstring_t m_str = KS_INITIALIZE;

public:
    WrappedKString() { m_str = utils::allocate_kstring(); }
    ~WrappedKString() { ks_free(&m_str); }

    kstring_t *get() { return &m_str; }
};
}  // namespace

TEST_CASE("BamUtilsTest: fetch keys from PG header", TEST_GROUP) {
    fs::path aligner_test_dir = fs::path(get_data_dir("aligner_test"));
    auto sam = aligner_test_dir / "basecall.sam";

    auto keys = utils::extract_pg_keys_from_hdr(sam.string(), {"PN", "CL", "VN"});
    CHECK(keys["PN"] == "dorado");
    CHECK(keys["VN"] == "0.5.0+5fa4de73+dirty");
    CHECK(keys["CL"] ==
          "dorado basecaller dna_r9.4.1_e8_hac@v3.3 ./tests/data/pod5 -x cpu --modified-bases "
          "5mCG --emit-sam");
}

TEST_CASE("BamUtilsTest: add_rg_hdr read group headers", TEST_GROUP) {
    auto has_read_group_header = [](sam_hdr_t *ptr, const char *id) {
        return sam_hdr_line_index(ptr, "RG", id) >= 0;
    };
    WrappedKString barcode_kstring;
    auto get_barcode_tag = [&barcode_kstring](sam_hdr_t *ptr,
                                              const char *id) -> std::optional<std::string> {
        if (sam_hdr_find_tag_id(ptr, "RG", "ID", id, "BC", barcode_kstring.get()) != 0) {
            return std::nullopt;
        }
        std::string tag(ks_str(barcode_kstring.get()), ks_len(barcode_kstring.get()));
        return tag;
    };

    SECTION("No read groups generate no headers") {
        dorado::SamHdrPtr sam_header(sam_hdr_init());
        CHECK(sam_hdr_count_lines(sam_header.get(), "RG") == 0);
        dorado::utils::add_rg_hdr(sam_header.get(), {}, {}, nullptr);
        CHECK(sam_hdr_count_lines(sam_header.get(), "RG") == 0);
    }

    const std::unordered_map<std::string, dorado::ReadGroup> read_groups{
            {"id_0",
             {"run_0", "basecalling_model_0", "modbase_model_0", "flowcell_0", "device_0",
              "exp_start_0", "sample_0", "", ""}},
            {"id_1",
             {"run_1", "basecalling_model_1", "modbase_model_1", "flowcell_1", "device_1",
              "exp_start_1", "sample_1", "", ""}},
    };

    SECTION("Read groups") {
        dorado::SamHdrPtr sam_header(sam_hdr_init());
        dorado::utils::add_rg_hdr(sam_header.get(), read_groups, {}, nullptr);

        // Check the IDs of the groups are all there.
        CHECK(sam_hdr_count_lines(sam_header.get(), "RG") == int(read_groups.size()));
        for (auto &&[id, read_group] : read_groups) {
            CHECK(has_read_group_header(sam_header.get(), id.c_str()));
            // None of the read groups should have a barcode.
            CHECK(get_barcode_tag(sam_header.get(), id.c_str()) == std::nullopt);
        }
    }

    // Pick some of the barcode kits (randomly chosen indices).
    const auto &kit_infos = dorado::barcode_kits::get_kit_infos();
    const std::vector<std::string> barcode_kits{
            std::next(kit_infos.begin(), 1)->first,
            std::next(kit_infos.begin(), 7)->first,
    };

    SECTION("Read groups with barcodes") {
        dorado::SamHdrPtr sam_header(sam_hdr_init());
        dorado::utils::add_rg_hdr(sam_header.get(), read_groups, barcode_kits, nullptr);

        // Check the IDs of the groups are all there.
        size_t total_barcodes = 0;
        for (const auto &kit_name : barcode_kits) {
            total_barcodes += kit_infos.at(kit_name).barcodes.size();
        }
        const size_t total_groups = read_groups.size() * (total_barcodes + 1);
        CHECK(sam_hdr_count_lines(sam_header.get(), "RG") == int(total_groups));

        // Check that the IDs match the expected format.
        const auto &barcode_seqs = dorado::barcode_kits::get_barcodes();
        for (auto &&[id, read_group] : read_groups) {
            CHECK(has_read_group_header(sam_header.get(), id.c_str()));
            CHECK(get_barcode_tag(sam_header.get(), id.c_str()) == std::nullopt);

            // The headers with barcodes should contain those barcodes.
            for (const auto &kit_name : barcode_kits) {
                const auto &kit_info = kit_infos.at(kit_name);
                for (const auto &barcode_name : kit_info.barcodes) {
                    const auto full_id = id + "_" +
                                         dorado::barcode_kits::generate_standard_barcode_name(
                                                 kit_name, barcode_name);
                    const auto &barcode_seq = barcode_seqs.at(barcode_name);
                    CHECK(has_read_group_header(sam_header.get(), full_id.c_str()));
                    CHECK(get_barcode_tag(sam_header.get(), full_id.c_str()) == barcode_seq);
                }
            }
        }
    }

    SECTION("Read groups with unknown barcode kit") {
        dorado::SamHdrPtr sam_header(sam_hdr_init());
        CHECK_THROWS(dorado::utils::add_rg_hdr(sam_header.get(), read_groups, {"blah"}, nullptr));
    }
}

TEST_CASE("BamUtilsTest: Test bam extraction helpers", TEST_GROUP) {
    fs::path bam_utils_test_dir = fs::path(get_data_dir("bam_utils"));
    auto sam = bam_utils_test_dir / "test.sam";

    HtsReader reader(sam.string(), std::nullopt);
    REQUIRE(reader.read());  // Parse first and only record.
    auto record = reader.record.get();

    SECTION("Test sequence extraction") {
        std::string seq = utils::extract_sequence(record);
        CHECK(seq ==
              "AATAAACCGAAGACAATTTAGAAGCCAGCGAGGTATGTGCGTCTACTTCGTTCGGTTATGCGAAGCCGATATAACCTGCAGGAC"
              "AACACAACATTTCCACTGTTTTCGTTCATTCGTAAACGCTTTCGCGTTCATCACACTCAACCATAGGCTTTAGCCAGAACGTTA"
              "TGAACCCCAGCGACTTCCAGAACGGCGCGCGTGCCACCACCGGCGATGATACCGGTTCCTTCGGAAGCCGGCTGCATGAATACG"
              "CGAGAACCCGTGTGAACACCTTTAACAGGGTGTTGCAGAGTGCCGTTGCTGCGGCACGATAGTTAAGTCGTATTGCTGAAGCGA"
              "CACTGTCCATCGCTTTCTGGATGGCT");
    }

    SECTION("Test quality extraction") {
        const std::string qual =
                "%$%&%$####%'%%$&'(1/...022.+%%%%%%$$%%&%$%%%&&+)()./"
                "0%$$'&'&'%$###$&&&'*(()()%%%%(%%'))(('''3222276<BAAABE:+''&)**%(/"
                "''(:322**(*,,++&+++/1)(&&(006=B??@AKLK=<==HHHHHFFCBB@??>==943323/-.'56::71.//"
                "0933))%&%&))*1739:666455116/"
                "0,(%%&(*-55EBEB>@;??>>@BBDC?><<98-,,BGHEGFFGIIJFFDBB;6AJ>===KB:::<70/"
                "..--,++,))+*)&&'*-,+*)))(%%&'&''%%%$&%$###$%%$$%'%%$$+1.--.7969....*)))";
        auto qual_vector = utils::extract_quality(record);
        CHECK(qual_vector.size() == qual.length());
        for (size_t i = 0; i < qual.length(); i++) {
            CHECK(qual[i] == qual_vector[i] + 33);
        }
    }

    SECTION("Test move table extraction") {
        auto [stride, move_table] = utils::extract_move_table(record);
        int seqlen = record->core.l_qseq;
        REQUIRE(!move_table.empty());
        CHECK(stride == 6);
        CHECK(seqlen == std::accumulate(move_table.begin(), move_table.end(), 0));
    }

    SECTION("Test mod base info extraction") {
        auto [modbase_str, modbase_probs] = utils::extract_modbase_info(record);
        const std::vector<int8_t> expected_modbase_probs = {5, 1};
        CHECK(modbase_str == "C+h?,1;C+m?,1;");
        CHECK(modbase_probs.size() == expected_modbase_probs.size());
        for (size_t i = 0; i < expected_modbase_probs.size(); i++) {
            CHECK(modbase_probs[i] == expected_modbase_probs[i]);
        }
    }
}

TEST_CASE("BamUtilsTest: cigar2str utility", TEST_GROUP) {
    const std::string cigar = "12S17M1D296M2D21M1D3M2D10M1I320M1D2237M41S";
    size_t m = 0;
    uint32_t *a_cigar = NULL;
    char *end = NULL;
    int n_cigar = int(sam_parse_cigar(cigar.c_str(), &end, &a_cigar, &m));
    std::string converted_str = utils::cigar2str(n_cigar, a_cigar);
    CHECK(cigar == converted_str);

    if (a_cigar) {
        hts_free(a_cigar);
    }
}

TEST_CASE("BamUtilsTest: Test trim CIGAR", TEST_GROUP) {
    const std::string cigar = "12S17M1D296M2D21M1D3M2D10M1I320M1D2237M41S";
    size_t m = 0;
    uint32_t *a_cigar = NULL;
    char *end = NULL;
    int n_cigar = int(sam_parse_cigar(cigar.c_str(), &end, &a_cigar, &m));
    const uint32_t qlen = uint32_t(bam_cigar2qlen(n_cigar, a_cigar));

    SECTION("Trim nothing") {
        auto ops = utils::trim_cigar(n_cigar, a_cigar, {0, qlen});
        std::string converted_str = utils::cigar2str(uint32_t(ops.size()), ops.data());
        CHECK(converted_str == "12S17M1D296M2D21M1D3M2D10M1I320M1D2237M41S");
    }

    SECTION("Trim from first op") {
        auto ops = utils::trim_cigar(n_cigar, a_cigar, {1, qlen});
        std::string converted_str = utils::cigar2str(uint32_t(ops.size()), ops.data());
        CHECK(converted_str == "11S17M1D296M2D21M1D3M2D10M1I320M1D2237M41S");
    }

    SECTION("Trim entire first op") {
        auto ops = utils::trim_cigar(n_cigar, a_cigar, {12, qlen});
        std::string converted_str = utils::cigar2str(uint32_t(ops.size()), ops.data());
        CHECK(converted_str == "17M1D296M2D21M1D3M2D10M1I320M1D2237M41S");
    }

    SECTION("Trim several ops from the front") {
        auto ops = utils::trim_cigar(n_cigar, a_cigar, {29, qlen});
        std::string converted_str = utils::cigar2str(uint32_t(ops.size()), ops.data());
        CHECK(converted_str == "296M2D21M1D3M2D10M1I320M1D2237M41S");
    }

    SECTION("Trim from last op") {
        auto ops = utils::trim_cigar(n_cigar, a_cigar, {0, qlen - 20});
        std::string converted_str = utils::cigar2str(uint32_t(ops.size()), ops.data());
        CHECK(converted_str == "12S17M1D296M2D21M1D3M2D10M1I320M1D2237M21S");
    }

    SECTION("Trim entire last op") {
        auto ops = utils::trim_cigar(n_cigar, a_cigar, {0, qlen - 41});
        std::string converted_str = utils::cigar2str(uint32_t(ops.size()), ops.data());
        CHECK(converted_str == "12S17M1D296M2D21M1D3M2D10M1I320M1D2237M");
    }

    SECTION("Trim several ops from the end") {
        auto ops = utils::trim_cigar(n_cigar, a_cigar, {0, qlen - 2278});
        std::string converted_str = utils::cigar2str(uint32_t(ops.size()), ops.data());
        CHECK(converted_str == "12S17M1D296M2D21M1D3M2D10M1I320M");
    }

    SECTION("Trim from the middle") {
        auto ops = utils::trim_cigar(n_cigar, a_cigar, {29, qlen - 2278});
        std::string converted_str = utils::cigar2str(uint32_t(ops.size()), ops.data());
        CHECK(converted_str == "296M2D21M1D3M2D10M1I320M");
    }

    if (a_cigar) {
        hts_free(a_cigar);
    }
}

TEST_CASE("BamUtilsTest: Ref positions consumed", TEST_GROUP) {
    const std::string cigar = "12S17M1D296M2D21M1D3M2D10M1I320M1D2237M41S";
    size_t m = 0;
    uint32_t *a_cigar = NULL;
    char *end = NULL;
    int n_cigar = int(sam_parse_cigar(cigar.c_str(), &end, &a_cigar, &m));

    SECTION("No positions consumed") {
        auto pos_consumed = utils::ref_pos_consumed(n_cigar, a_cigar, 0);
        CHECK(pos_consumed == 0);
    }

    SECTION("No positions consumed with soft clipping") {
        auto pos_consumed = utils::ref_pos_consumed(n_cigar, a_cigar, 12);
        CHECK(pos_consumed == 0);
    }

    SECTION("Match positions consumed") {
        auto pos_consumed = utils::ref_pos_consumed(n_cigar, a_cigar, 25);
        CHECK(pos_consumed == 13);
    }

    SECTION("Match and delete positions consumed") {
        auto pos_consumed = utils::ref_pos_consumed(n_cigar, a_cigar, 29);
        CHECK(pos_consumed == 18);
    }

    if (a_cigar) {
        hts_free(a_cigar);
    }
}

TEST_CASE("BamUtilsTest: test sam_hdr_merge on identical headers", TEST_GROUP) {
    std::string header_1 =
            "@HD\tVN:1.6\tSO:unknown\n"
            "@SQ\tSN:Lambda\tLN:48400\n"
            "@PG\tID:aligner\tPN:minimap2\tVN:2.24-r1122\n"
            "@RG\tID:a706823101911eaf79e9538f89284a76cec07945_unknown\tDS:runid="
            "a706823101911eaf79e9538f89284a76cec07945\tPL:ONT\n";

    sam_hdr_t *header_dest = sam_hdr_parse(header_1.size(), header_1.c_str());
    sam_hdr_t *header_src = sam_hdr_dup(header_dest);

    std::string error_msg;
    bool result = utils::sam_hdr_merge(header_dest, header_src, error_msg);
    CHECK(result == true);

    std::string result_header = sam_hdr_str(header_dest);
    CHECK(result_header == header_1);
}

TEST_CASE("BamUtilsTest: test sam_hdr_merge on overlapping headers", TEST_GROUP) {
    std::string header_1 =
            "@HD\tVN:1.6\tSO:unknown\n"
            "@SQ\tSN:Lambda\tLN:48400\n"
            "@PG\tID:aligner\tPN:minimap2\tVN:2.24-r1122\n"
            "@RG\tID:a706823101911eaf79e9538f89284a76cec07945_unknown\tDS:runid="
            "a706823101911eaf79e9538f89284a76cec07945\tPL:ONT\n";

    std::string header_2 =
            "@HD\tVN:1.6\tSO:unknown\n"
            "@SQ\tSN:Lambda\tLN:48400\n"
            "@PG\tID:aligner\tPN:minimap2\tVN:2.24-r1122\n"
            "@RG\tID:b106823101911eaf79e9538f89284a76cec0797f_unknown\tDS:runid="
            "b106823101911eaf79e9538f89284a76cec0797f\tPL:ONT\n";

    std::string expected_result =
            "@HD\tVN:1.6\tSO:unknown\n"
            "@SQ\tSN:Lambda\tLN:48400\n"
            "@PG\tID:aligner\tPN:minimap2\tVN:2.24-r1122\n"
            "@RG\tID:a706823101911eaf79e9538f89284a76cec07945_unknown\tDS:runid="
            "a706823101911eaf79e9538f89284a76cec07945\tPL:ONT\n"
            "@RG\tID:b106823101911eaf79e9538f89284a76cec0797f_unknown\tDS:runid="
            "b106823101911eaf79e9538f89284a76cec0797f\tPL:ONT\n";

    sam_hdr_t *header_dest = sam_hdr_parse(header_1.size(), header_1.c_str());
    sam_hdr_t *header_src = sam_hdr_parse(header_2.size(), header_2.c_str());

    std::string error_msg;
    bool result = utils::sam_hdr_merge(header_dest, header_src, error_msg);
    CHECK(result == true);

    std::string result_header = sam_hdr_str(header_dest);
    CHECK(result_header == expected_result);
}

TEST_CASE("BamUtilsTest: test sam_hdr_merge unsets SO tag in HD line", TEST_GROUP) {
    std::string header_1 = "@HD\tVN:1.6\tSO:coordinate\n";
    std::string header_2 = "@HD\tVN:1.6\tSO:queryname\n";
    std::string expected_result = "@HD\tVN:1.6\tSO:unknown\n";

    sam_hdr_t *header_dest = sam_hdr_parse(header_1.size(), header_1.c_str());
    sam_hdr_t *header_src = sam_hdr_parse(header_2.size(), header_2.c_str());

    std::string error_msg;
    bool result = utils::sam_hdr_merge(header_dest, header_src, error_msg);
    CHECK(result == true);

    std::string result_header = sam_hdr_str(header_dest);
    CHECK(result_header == expected_result);
}

TEST_CASE("BamUtilsTest: test sam_hdr_merge refuses to merge incompatible PG", TEST_GROUP) {
    std::string header_1 =
            "@HD\tVN:1.6\tSO:coordinate\n"
            "@PG\tID:aligner\tPN:minimap2\tVN:2.24-r1122\n";
    std::string header_2 =
            "@HD\tVN:1.6\tSO:queryname\n"
            "@PG\tID:aligner\tPN:minimap3\tVN:2.24-r1122\n";

    sam_hdr_t *header_dest = sam_hdr_parse(header_1.size(), header_1.c_str());
    sam_hdr_t *header_src = sam_hdr_parse(header_2.size(), header_2.c_str());

    std::string error_msg;
    bool result = utils::sam_hdr_merge(header_dest, header_src, error_msg);
    CHECK(result == false);
    CHECK(error_msg.size() != 0);
}

TEST_CASE("BamUtilsTest: test sam_hdr_merge refuses to merge incompatible SQ", TEST_GROUP) {
    std::string header_1 =
            "@HD\tVN:1.6\tSO:coordinate\n"
            "@SQ\tSN:Lambda\tLN:48400\n";
    std::string header_2 =
            "@HD\tVN:1.6\tSO:queryname\n"
            "@SQ\tSN:Chicken\tLN:32000000\n";

    sam_hdr_t *header_dest = sam_hdr_parse(header_1.size(), header_1.c_str());
    sam_hdr_t *header_src = sam_hdr_parse(header_2.size(), header_2.c_str());

    std::string error_msg;
    bool result = utils::sam_hdr_merge(header_dest, header_src, error_msg);
    CHECK(result == false);
    CHECK(error_msg.size() != 0);
}
