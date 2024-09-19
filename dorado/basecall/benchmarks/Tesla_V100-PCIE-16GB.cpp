#include "Tesla_V100-PCIE-16GB.h"

namespace dorado::basecall {

void AddTesla_V100_PCIE_16GBBenchmarks(
        std::map<std::tuple<std::string, std::string>, std::map<int, float>>& chunk_benchmarks) {
    chunk_benchmarks[{"Tesla V100-PCIE-16GB", "dna_r10.4.1_e8.2_260bps_fast@v4.1.0"}] = {
            {64, 0.033008f},   {128, 0.018264f},  {192, 0.013893f},  {256, 0.011592f},
            {320, 0.009949f},  {512, 0.009526f},  {576, 0.008949f},  {640, 0.008383f},
            {1280, 0.008156f}, {1920, 0.008113f}, {3200, 0.008075f},
    };
    chunk_benchmarks[{"Tesla V100-PCIE-16GB", "dna_r10.4.1_e8.2_260bps_hac@v4.1.0"}] = {
            {64, 2.472848f},   {128, 1.228984f},  {192, 0.831024f},  {256, 0.620844f},
            {320, 0.497040f},  {384, 0.413211f},  {448, 0.356110f},  {512, 0.311516f},
            {576, 0.277952f},  {640, 0.250406f},  {704, 0.227318f},  {768, 0.208715f},
            {832, 0.193426f},  {896, 0.180186f},  {960, 0.167874f},  {1024, 0.157686f},
            {1088, 0.148737f}, {1152, 0.140414f}, {1216, 0.133336f}, {1280, 0.126833f},
            {1344, 0.121375f}, {1408, 0.116934f}, {1472, 0.111261f}, {1536, 0.106864f},
            {1600, 0.102447f}, {1664, 0.098828f}, {1728, 0.095049f}, {1792, 0.091628f},
            {1856, 0.088525f}, {1920, 0.085663f}, {1984, 0.083818f}, {2048, 0.081312f},
            {2112, 0.078835f}, {2176, 0.076630f}, {2240, 0.074661f}, {2304, 0.072528f},
            {2368, 0.070739f}, {2432, 0.068713f}, {2560, 0.068707f}, {2624, 0.066752f},
            {2688, 0.066296f}, {2752, 0.064581f}, {2816, 0.064036f}, {2880, 0.062991f},
            {2944, 0.062411f}, {3008, 0.061156f}, {3072, 0.060919f}, {3136, 0.059779f},
            {3200, 0.058712f}, {3264, 0.058311f}, {3328, 0.057818f},
    };
    chunk_benchmarks[{"Tesla V100-PCIE-16GB", "dna_r10.4.1_e8.2_260bps_sup@v4.1.0"}] = {
            {64, 2.710656f},  {128, 1.310200f}, {192, 0.895232f}, {256, 0.673180f},
            {320, 0.541930f}, {384, 0.454427f}, {448, 0.391719f}, {512, 0.365278f},
            {576, 0.314709f}, {640, 0.292614f},
    };
    chunk_benchmarks[{"Tesla V100-PCIE-16GB", "dna_r10.4.1_e8.2_400bps_fast@v4.1.0"}] = {
            {64, 0.032640f},   {128, 0.018264f},  {192, 0.013893f}, {256, 0.011556f},
            {320, 0.009878f},  {512, 0.009564f},  {576, 0.008923f}, {640, 0.008355f},
            {1280, 0.008154f}, {1920, 0.008115f},
    };
    chunk_benchmarks[{"Tesla V100-PCIE-16GB", "dna_r10.4.1_e8.2_400bps_fast@v4.3.0"}] = {
            {64, 0.033214f},   {128, 0.018640f},  {192, 0.014571f},  {256, 0.011956f},
            {320, 0.010198f},  {512, 0.009914f},  {576, 0.009223f},  {640, 0.008629f},
            {1280, 0.008378f}, {2560, 0.008362f}, {3200, 0.008352f}, {3840, 0.008349f},
    };
    chunk_benchmarks[{"Tesla V100-PCIE-16GB", "dna_r10.4.1_e8.2_400bps_fast@v5.0.0"}] = {
            {64, 0.033393f},   {128, 0.019264f},  {192, 0.014581f}, {256, 0.011904f},
            {320, 0.010224f},  {512, 0.009888f},  {576, 0.009195f}, {640, 0.008614f},
            {1280, 0.008390f}, {1920, 0.008356f},
    };
    chunk_benchmarks[{"Tesla V100-PCIE-16GB", "dna_r10.4.1_e8.2_400bps_hac@v4.1.0"}] = {
            {64, 2.460944f},   {128, 1.230088f},  {192, 0.825797f},  {256, 0.615756f},
            {320, 0.494909f},  {384, 0.413381f},  {448, 0.354823f},  {512, 0.312192f},
            {576, 0.278441f},  {640, 0.250141f},  {704, 0.226385f},  {768, 0.208715f},
            {832, 0.192805f},  {896, 0.179081f},  {960, 0.167306f},  {1024, 0.157269f},
            {1088, 0.148594f}, {1152, 0.140036f}, {1216, 0.133325f}, {1280, 0.126720f},
            {1344, 0.121165f}, {1408, 0.116327f}, {1472, 0.111002f}, {1536, 0.106309f},
            {1600, 0.102182f}, {1664, 0.098460f}, {1728, 0.094569f}, {1792, 0.091461f},
            {1856, 0.088586f}, {1920, 0.085534f}, {1984, 0.083603f}, {2048, 0.080930f},
            {2112, 0.078871f}, {2176, 0.077117f}, {2240, 0.074624f}, {2304, 0.072233f},
            {2368, 0.070550f}, {2432, 0.068954f}, {2496, 0.068410f}, {2624, 0.066924f},
            {2688, 0.066371f}, {2752, 0.064559f}, {2816, 0.064100f}, {2880, 0.063746f},
            {2944, 0.062451f}, {3008, 0.061567f}, {3072, 0.060548f}, {3136, 0.059716f},
            {3200, 0.059100f}, {3264, 0.058342f}, {3328, 0.057606f},
    };
    chunk_benchmarks[{"Tesla V100-PCIE-16GB", "dna_r10.4.1_e8.2_400bps_hac@v4.3.0"}] = {
            {64, 2.565808f},   {128, 1.244464f},  {192, 0.830448f},  {256, 0.626116f},
            {320, 0.500384f},  {384, 0.420379f},  {448, 0.358181f},  {512, 0.315452f},
            {576, 0.283280f},  {640, 0.253094f},  {704, 0.230419f},  {768, 0.210845f},
            {832, 0.194574f},  {896, 0.181417f},  {960, 0.169378f},  {1024, 0.158892f},
            {1088, 0.149592f}, {1152, 0.141689f}, {1216, 0.134835f}, {1280, 0.128138f},
            {1344, 0.122351f}, {1408, 0.117371f}, {1472, 0.112424f}, {1536, 0.107804f},
            {1600, 0.103572f}, {1664, 0.099980f}, {1728, 0.096010f}, {1792, 0.092753f},
            {1856, 0.089917f}, {1920, 0.086886f}, {1984, 0.085004f}, {2048, 0.082211f},
            {2112, 0.079967f}, {2176, 0.077780f}, {2240, 0.075430f}, {2304, 0.073603f},
            {2368, 0.071792f}, {2432, 0.070197f}, {2624, 0.068549f}, {2688, 0.067357f},
            {2752, 0.066170f}, {2816, 0.065419f}, {2880, 0.064058f}, {3008, 0.062681f},
            {3072, 0.062077f}, {3136, 0.060794f}, {3200, 0.060061f}, {3264, 0.059436f},
            {3328, 0.059297f},
    };
    chunk_benchmarks[{"Tesla V100-PCIE-16GB", "dna_r10.4.1_e8.2_400bps_hac@v5.0.0"}] = {
            {64, 2.563456f},   {128, 1.233456f},  {192, 0.825392f},  {256, 0.620040f},
            {320, 0.496931f},  {384, 0.415955f},  {448, 0.353931f},  {512, 0.311602f},
            {576, 0.277310f},  {640, 0.249978f},  {704, 0.227165f},  {768, 0.208837f},
            {832, 0.192655f},  {896, 0.179183f},  {960, 0.168508f},  {1024, 0.157974f},
            {1088, 0.148667f}, {1152, 0.141067f}, {1216, 0.133849f}, {1280, 0.127371f},
            {1344, 0.121810f}, {1408, 0.116284f}, {1472, 0.111308f}, {1536, 0.106900f},
            {1600, 0.102857f}, {1664, 0.099266f}, {1728, 0.095151f}, {1792, 0.092059f},
            {1856, 0.088940f}, {1920, 0.086415f}, {1984, 0.084265f}, {2048, 0.081424f},
            {2112, 0.079213f}, {2176, 0.076897f}, {2240, 0.074896f}, {2304, 0.073152f},
            {2368, 0.071373f}, {2432, 0.069664f}, {2624, 0.068994f}, {2688, 0.067360f},
            {2752, 0.066112f}, {2816, 0.065160f}, {2880, 0.064306f}, {2944, 0.063948f},
            {3008, 0.062478f}, {3072, 0.062228f}, {3136, 0.061226f}, {3200, 0.060825f},
            {3264, 0.059265f}, {3328, 0.059018f},
    };
    chunk_benchmarks[{"Tesla V100-PCIE-16GB", "dna_r10.4.1_e8.2_400bps_sup@v4.1.0"}] = {
            {64, 2.559440f},  {128, 1.267528f}, {192, 0.860581f}, {256, 0.643304f},
            {320, 0.514317f}, {384, 0.431370f}, {448, 0.392418f}, {512, 0.365374f},
            {576, 0.316460f}, {640, 0.291642f},
    };
    chunk_benchmarks[{"Tesla V100-PCIE-16GB", "dna_r10.4.1_e8.2_400bps_sup@v4.3.0"}] = {
            {64, 2.566849f},  {128, 1.282352f}, {192, 0.877371f}, {256, 0.665928f},
            {320, 0.531613f}, {384, 0.449179f}, {448, 0.417143f}, {512, 0.382094f},
            {576, 0.332645f}, {640, 0.309310f},
    };
    chunk_benchmarks[{"Tesla V100-PCIE-16GB", "dna_r10.4.1_e8.2_400bps_sup@v5.0.0"}] = {
            {32, 2.433216f},  {64, 1.196096f},  {96, 0.846432f},  {128, 0.810040f},
            {160, 0.795047f}, {192, 0.774192f}, {224, 0.768334f}, {256, 0.760544f},
            {288, 0.757646f}, {320, 0.756256f}, {352, 0.746833f},
    };
    chunk_benchmarks[{"Tesla V100-PCIE-16GB", "dna_r9.4.1_e8_fast@v3.4"}] = {
            {64, 0.053456f},   {128, 0.029096f},  {192, 0.021189f},  {256, 0.017228f},
            {320, 0.014736f},  {512, 0.013786f},  {576, 0.012850f},  {640, 0.011608f},
            {1216, 0.011483f}, {1280, 0.011118f}, {1920, 0.011066f}, {2560, 0.011040f},
    };
    chunk_benchmarks[{"Tesla V100-PCIE-16GB", "dna_r9.4.1_e8_hac@v3.3"}] = {
            {64, 2.530288f},   {128, 1.238768f},  {192, 0.824683f},  {256, 0.619040f},
            {320, 0.496429f},  {384, 0.415835f},  {448, 0.357424f},  {512, 0.314672f},
            {576, 0.280501f},  {640, 0.251102f},  {704, 0.229865f},  {768, 0.210800f},
            {832, 0.194940f},  {896, 0.181019f},  {960, 0.169823f},  {1024, 0.159549f},
            {1088, 0.150211f}, {1152, 0.142506f}, {1216, 0.135227f}, {1280, 0.128678f},
            {1344, 0.123800f}, {1408, 0.118412f}, {1472, 0.113469f}, {1536, 0.108827f},
            {1600, 0.104529f}, {1664, 0.100779f}, {1728, 0.097246f}, {1792, 0.093554f},
            {1856, 0.090519f}, {1920, 0.088148f}, {1984, 0.085907f}, {2048, 0.083295f},
            {2112, 0.081059f}, {2176, 0.079025f}, {2240, 0.076851f}, {2304, 0.074916f},
            {2368, 0.072896f}, {2432, 0.070935f}, {2496, 0.069687f}, {2624, 0.068036f},
            {2688, 0.067746f}, {2752, 0.066674f}, {2816, 0.065018f}, {2944, 0.063899f},
            {3008, 0.062710f}, {3072, 0.062436f}, {3136, 0.060947f}, {3200, 0.060315f},
            {3264, 0.058961f}, {3328, 0.058891f},
    };
    chunk_benchmarks[{"Tesla V100-PCIE-16GB", "dna_r9.4.1_e8_sup@v3.3"}] = {
            {64, 2.553472f},  {128, 1.274424f}, {192, 0.859131f}, {256, 0.651192f},
            {320, 0.529645f}, {384, 0.445501f}, {448, 0.385337f}, {512, 0.335258f},
            {576, 0.303172f}, {640, 0.275349f}, {704, 0.252688f}, {768, 0.234692f},
    };
    chunk_benchmarks[{"Tesla V100-PCIE-16GB", "rna004_130bps_fast@v5.1.0"}] = {
            {64, 0.033024f},   {128, 0.017664f},  {192, 0.013051f},  {256, 0.010908f},
            {320, 0.009440f},  {512, 0.008920f},  {576, 0.008395f},  {640, 0.007939f},
            {1216, 0.007911f}, {1280, 0.007703f}, {1920, 0.007677f}, {2560, 0.007651f},
    };
    chunk_benchmarks[{"Tesla V100-PCIE-16GB", "rna004_130bps_hac@v5.1.0"}] = {
            {64, 2.556990f},   {128, 1.236336f},  {192, 0.827131f},  {256, 0.617528f},
            {320, 0.496656f},  {384, 0.414024f},  {448, 0.355129f},  {512, 0.311724f},
            {576, 0.277339f},  {640, 0.249936f},  {704, 0.227145f},  {768, 0.209871f},
            {832, 0.193279f},  {896, 0.180063f},  {960, 0.168135f},  {1024, 0.158295f},
            {1088, 0.149001f}, {1152, 0.140993f}, {1216, 0.133739f}, {1280, 0.127777f},
            {1344, 0.121698f}, {1408, 0.116610f}, {1472, 0.111545f}, {1536, 0.107427f},
            {1600, 0.103116f}, {1664, 0.099471f}, {1728, 0.095223f}, {1792, 0.091947f},
            {1856, 0.088721f}, {1920, 0.086442f}, {1984, 0.084057f}, {2048, 0.082209f},
            {2112, 0.079490f}, {2176, 0.077220f}, {2240, 0.075197f}, {2304, 0.073431f},
            {2368, 0.071579f}, {2432, 0.069642f}, {2624, 0.068971f}, {2688, 0.066996f},
            {2752, 0.066476f}, {2816, 0.065511f}, {2880, 0.064447f}, {2944, 0.064337f},
            {3008, 0.062948f}, {3072, 0.062320f}, {3136, 0.061536f}, {3200, 0.060635f},
            {3264, 0.059551f}, {3328, 0.059264f},
    };
    chunk_benchmarks[{"Tesla V100-PCIE-16GB", "rna004_130bps_sup@v5.1.0"}] = {
            {32, 2.328000f},  {64, 1.184304f},  {96, 0.846635f},  {128, 0.810416f},
            {160, 0.795341f}, {192, 0.774459f}, {224, 0.768672f},
    };
}

}  // namespace dorado::basecall
