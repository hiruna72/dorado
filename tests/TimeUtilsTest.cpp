#include "utils/time_utils.h"

#include <catch2/catch.hpp>

#include <chrono>
#include <limits>

using std::make_tuple;
using namespace std::chrono;
using namespace std::chrono_literals;

#define CUT_TAG "[TimeUtils]"

TEST_CASE(CUT_TAG ": std::string <-> time_t", CUT_TAG) {
    using sys = std::chrono::system_clock;
    using time_point = std::chrono::system_clock::time_point;

    time_t unix_time_ms;
    std::string timestamp;
    SECTION("With timezone HH:MM") {
        std::tie(timestamp, unix_time_ms) = GENERATE(table<std::string, time_t>({
                // clang-format off
                    make_tuple("1970-01-01T00:00:00.000+00:00", sys::to_time_t(time_point(0s)) * 1000),
                    make_tuple("1970-01-02T00:00:00.000+00:00", sys::to_time_t(time_point(24h)) * 1000),
                    make_tuple("1971-01-02T00:00:00.000+00:00", sys::to_time_t(time_point(8784h)) * 1000),
                    make_tuple("1975-01-02T00:00:00.000+00:00", sys::to_time_t(time_point(43848h)) * 1000),
                    make_tuple("1975-01-02T00:00:00.456+00:00", sys::to_time_t(time_point(43848h)) * 1000 + 456),
                // clang-format on
        }));
        CAPTURE(timestamp);

        auto result_str = dorado::utils::get_string_timestamp_from_unix_time(unix_time_ms);
        CHECK(result_str == timestamp);

        auto result_ms = dorado::utils::get_unix_time_from_string_timestamp(timestamp);
        CHECK(result_ms == unix_time_ms);
    }

    SECTION("With timezone Z") {
        std::tie(timestamp, unix_time_ms) = GENERATE(table<std::string, time_t>({
                // clang-format off
                    make_tuple("1970-01-01T00:00:00Z", sys::to_time_t(time_point(0s)) * 1000),
                    make_tuple("1970-01-02T00:00:00Z", sys::to_time_t(time_point(24h)) * 1000),
                    make_tuple("1971-01-02T00:00:00Z", sys::to_time_t(time_point(8784h)) * 1000),
                    make_tuple("1975-01-02T00:00:00Z", sys::to_time_t(time_point(43848h)) * 1000),
                // clang-format on
        }));
        CAPTURE(timestamp);
        auto result_ms = dorado::utils::get_unix_time_from_string_timestamp(timestamp);
        CHECK(result_ms == unix_time_ms);
    }

    SECTION("With microseconds") {
        std::tie(timestamp, unix_time_ms) = GENERATE(table<std::string, time_t>({
                // clang-format off
                    make_tuple("1970-01-01T00:00:00.000000+00:00", sys::to_time_t(time_point(0s)) * 1000),
                    make_tuple("1970-01-02T00:00:00.000101+00:00", sys::to_time_t(time_point(24h)) * 1000),
                    make_tuple("1971-01-02T00:00:00.456000+00:00", sys::to_time_t(time_point(8784h)) * 1000 + 456),
                    make_tuple("1975-01-02T00:00:00.456123+00:00", sys::to_time_t(time_point(43848h)) * 1000 + 456),
                // clang-format on
        }));
        CAPTURE(timestamp);
        auto result_ms = dorado::utils::get_unix_time_from_string_timestamp(timestamp);
        CHECK(result_ms == unix_time_ms);
    }
}

TEST_CASE(CUT_TAG ": adjust_time", CUT_TAG) {
    std::string timestamp;
    uint32_t adjustment;
    std::string adjusted_timestamp;

    std::tie(timestamp, adjustment,
             adjusted_timestamp) = GENERATE(table<std::string, uint32_t, std::string>({
            // clang-format off
                make_tuple("1970-01-01T00:00:00Z",     0, "1970-01-01T00:00:00Z"), // no change, round trip
                make_tuple("1970-01-02T00:00:00Z",     1, "1970-01-02T00:00:01Z"), // add a second
                make_tuple("1971-01-02T00:00:00Z",  3600, "1971-01-02T01:00:00Z"), // add an hour
                make_tuple("1975-01-02T00:00:00Z", 86400, "1975-01-03T00:00:00Z"), // add a day
                make_tuple("1976-02-28T00:00:00Z", 86400, "1976-02-29T00:00:00Z"), // check leap day!
            // clang-format on
    }));
    CAPTURE(timestamp);
    auto result_time_stamp = dorado::utils::adjust_time(timestamp, adjustment);
    CHECK(result_time_stamp == adjusted_timestamp);
}