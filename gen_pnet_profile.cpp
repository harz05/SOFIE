// Regenerate the ParticleNet GPU header with profiling turned on (Options::kProfile).
// Overwrites the emit's header so the driver, compiled with -DPROFILE, prints per-op timings.
//   argv[1] = onnx path, argv[2] = output header basename (run from the dir you want it written in)
#include "SOFIE/RModelParser_ONNX.hxx"
#include "SOFIE/RModel.hxx"
#include <string>

int main(int argc, char **argv) {
   using namespace SOFIE;
   std::string onnx = (argc > 1) ? argv[1] : "test/input_models/particle-net.onnx";
   std::string out  = (argc > 2) ? argv[2] : "particle-net_FromONNX_GPU_ALPAKA.hxx";
   RModelParser_ONNX parser;
   RModel m = parser.Parse(onnx, /*verbose*/ false);
   m.GenerateGPU_ALPAKA(Options::kProfile, /*batchSize*/ -1, /*verbose*/ false);
   m.OutputGenerated(out);
   return 0;
}
