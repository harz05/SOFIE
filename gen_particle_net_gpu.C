// GPU/alpaka codegen for particle-net.onnx. Same as the mentor's CPU generator
// but calls GenerateGPU_ALPAKA instead of Generate. Pure host codegen (no CUDA
// needed): it walks every operator's Generate_GPU_ALPAKA, which throws on an
// unsupported op, so this alone reports which ops still need a GPU path.
//   root -l -b -q gen_particle_net_gpu.C
void gen_particle_net_gpu(bool verbose = 1, int n = 1, int n_sv = 10, int n_pf = 100) {

   TMVA::Experimental::SOFIE::RModelParser_ONNX p;
   auto m = p.Parse("particle-net.onnx", verbose);
   m.PrintInitializedTensors();
   TMVA::Experimental::SOFIE::Options opt = TMVA::Experimental::SOFIE::Options::kDefault;
   m.GenerateGPU_ALPAKA(opt, -1, verbose);
   m.OutputGenerated();

   gSystem->Exec(TString::Format("cp particle_net.hxx particle_net_gpu_%d_%d_%d.hxx", n, n_sv, n_pf));
   gSystem->Exec(TString::Format("cp particle_net.dat particle_net_gpu_%d_%d_%d.dat", n, n_sv, n_pf));
}
