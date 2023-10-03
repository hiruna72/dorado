#include "Version.h"
#include "cli/cli.h"
#include "minimap.h"
#include "spdlog/cfg/env.h"
#include "utils/cli_utils.h"

#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#ifdef __linux__
extern "C" {
// There's a bug in GLIBC < 2.25 (Bug 11941) which can trigger an assertion/
// seg fault when a dynamically loaded library is dlclose-d twice - once by ld.so and then
// once by the code that opened the DSO in the first place (more details available
// at https://sourceware.org/legacy-ml/libc-alpha/2016-12/msg00859.html). Dorado
// is seemingly running into this issue transitively through some dependent libraries
// (backtraces indicate libcudnn could be a source). The workaround below bypasses
// the dlclose subroutine entirely by making it a no-op. This will cause a memory
// leak as loaded shared libs won't be closed, but in practice this happens at
// teardown anyway so the leak will be subsumed by termination.
// Fix is borrowed from https://mailman.mit.edu/pipermail/cvs-krb5/2019-October/014884.html
#if !__GLIBC_PREREQ(2, 25)
int dlclose(void*) { return 0; };
#endif  // __GLIBC_PREREQ
}
#endif  // __linux__

using entry_ptr = std::function<int(int, char**)>;

namespace {

void usage(const std::vector<std::string> commands) {
    std::cerr << "Usage: dorado [options] subcommand\n\n"
              << "Positional arguments:" << std::endl;

    for (const auto command : commands) {
        std::cerr << command << std::endl;
    }

    std::cerr << "\nOptional arguments:\n"
              << "-h --help               shows help message and exits\n"
              << "-v --version            prints version information and exits\n"
              << "-vv                     prints verbose version information and exits"
              << std::endl;
}

}  // namespace

int main(int argc, char* argv[]) {
    // Load logging settings from environment/command-line.
    spdlog::cfg::load_env_levels();

    const std::map<std::string, entry_ptr> subcommands = {
            {"basecaller", &dorado::basecaller}, {"duplex", &dorado::duplex},
            {"download", &dorado::download},     {"aligner", &dorado::aligner},
            {"summary", &dorado::summary},
    };

    std::vector<std::string> arguments(argv + 1, argv + argc);
    std::vector<std::string> keys;

    for (const auto& [key, _] : subcommands) {
        keys.push_back(key);
    }

    if (arguments.size() == 0) {
        usage(keys);
        return 0;
    }

    const auto subcommand = arguments[0];

    if (subcommand == "-v" || subcommand == "--version") {
        std::cerr << DORADO_VERSION << std::endl;
    } else if (subcommand == "-vv") {
#ifdef __APPLE__
        std::cerr << "dorado:   " << DORADO_VERSION << std::endl;
#else
        std::cerr << "dorado:   " << DORADO_VERSION << "+cu" << CUDA_VERSION << std::endl;
#endif
        std::cerr << "libtorch: " << TORCH_BUILD_VERSION << std::endl;
        std::cerr << "minimap2: " << MM_VERSION << std::endl;

    } else if (subcommand == "-h" || subcommand == "--help") {
        usage(keys);
        return 0;
    } else if (subcommands.find(subcommand) != subcommands.end()) {
        return subcommands.at(subcommand)(--argc, ++argv);
    } else {
        usage(keys);
        return 1;
    }

    return 0;
}
