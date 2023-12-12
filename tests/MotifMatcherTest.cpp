#include "modbase/MotifMatcher.h"

#include "modbase/ModBaseModelConfig.h"

#include <catch2/catch.hpp>

#define TEST_GROUP "[modbase_motif_matcher]"

using std::make_tuple;

namespace {
// clang-format off
const std::string SEQ = "AACCGGTTACGTGGACTGACACTAAA";
//                      "   CG    CG               "
//                      "   CG    CG               "
//                      "  CC     C     C   C  C   "
//                      "AA                        "
//                      "                       AA "
//                      "                        AA"
//                      "       TAC                "
//                      "            DRACH         "
//                      "                DRACH     "
// clang-format on
}  // namespace

TEST_CASE(TEST_GROUP ": test motifs", TEST_GROUP) {
    dorado::modbase::ModBaseModelConfig config;
    auto [motif, motif_offset, expected_results] =
            GENERATE(table<std::string, size_t, std::vector<size_t>>({
                    // clang-format off
                    make_tuple("CG",    0, std::vector<size_t>{3, 9}),                // C in CG
                    make_tuple("CG",    1, std::vector<size_t>{4, 10}),               // G in CG
                    make_tuple("C",     0, std::vector<size_t>{2, 3, 9, 15, 19, 21}), // Any C
                    make_tuple("AA",    1, std::vector<size_t>{1, 24, 25}),           // A following A
                    make_tuple("TAC",   2, std::vector<size_t>{9}),                   // C in TAC
                    make_tuple("DRACH", 2, std::vector<size_t>{14, 18}),              // X=A in [AGT][AG]XC[ACT]
                    // clang-format on
            }));

    CAPTURE(motif);
    CAPTURE(motif_offset);
    config.motif = motif;
    config.motif_offset = motif_offset;

    dorado::modbase::MotifMatcher matcher(config);
    auto hits = matcher.get_motif_hits(SEQ);
    CHECK(hits == expected_results);
}
