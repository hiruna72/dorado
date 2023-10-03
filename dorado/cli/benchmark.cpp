#include "../utils/tensor_utils.h"
#include "Version.h"

#include <argparse.hpp>
#include <torch/torch.h>

#include <chrono>
#include <iostream>

namespace dorado {

int benchmark(int argc, char* argv[]) {
    argparse::ArgumentParser parser("dorado", DORADO_VERSION, argparse::default_arguments::help);

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        std::exit(1);
    }

    std::vector<size_t> sizes{1000, 1000, 2000, 3000, 4000, 10000, 100000, 1000000, 10000000};

    for (auto n : sizes) {
        std::cerr << "samples : " << n << std::endl;

        // generate some input
        auto x = torch::randint(0, 2047, n);
        auto q = torch::tensor({0.2, 0.9}, {torch::kFloat32});

        // torch::quantile
        auto start = std::chrono::system_clock::now();
        auto res = torch::quantile(x, q);
        auto end = std::chrono::system_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cerr << "torch:quant  "
                  << " q20=" << res[0].item<int>() << " q90=" << res[1].item<int>() << " "
                  << duration << "us" << std::endl;

        // nth_element
        start = std::chrono::system_clock::now();
        res = utils::quantile(x, q);
        end = std::chrono::system_clock::now();

        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cerr << "nth_element  "
                  << " q20=" << res[0].item<int>() << " q90=" << res[1].item<int>() << " "
                  << duration << "us" << std::endl;

        x = x.to(torch::kInt16);

        // counting
        start = std::chrono::system_clock::now();
        res = utils::quantile_counting(x, q);
        end = std::chrono::system_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cerr << "counting     "
                  << " q20=" << res[0].item<int>() << " q90=" << res[1].item<int>() << " "
                  << duration << "us" << std::endl
                  << std::endl;
    }

    return 0;
}

}  // namespace dorado
