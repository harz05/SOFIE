// GPU port of run_inference_particle_net.C: same structure/inputs, but the
// standalone alpaka GPU Session instead of the ROOT CPU one. TRandom -> std,
// TStopwatch -> chrono, check_mem is the ROOT-free version.

#ifndef NEVTS
#define NEVTS 100
#endif

#include "particle-net_FromONNX_GPU_ALPAKA.hxx"
#include "check_mem_gpu.h"

#include <alpaka/alpaka.hpp>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

void test_particle_net(int nevts = 1000, int n = 1, int n_sv = 10, int n_pf = 100) {
   // Session is a template on the accelerator tag; Acc/Idx/Dim live inside it.
   using Sess  = SOFIE_particle_net::Session<alpaka::TagGpuCudaRt>;
   using Acc   = Sess::Acc;
   using Idx   = Sess::Idx;
   using Dim   = Sess::Dim;
   using Ext1D = alpaka::Vec<Dim, Idx>;
   std::cout << n_sv << ", " << n_pf << std::endl;

   int nprint = nevts / 10;
   if (nprint == 0) nprint = 1;

   std::cout << "creating session..." << std::endl;
   check_mem("initial");

   // ctor order is (weightfile, N, n_pf, n_sv); construct at the same sizes we infer at
   // so the dynamic buffers and cuBLASLt layouts are registered for these dims.
   Sess s("particle-net_FromONNX_GPU_ALPAKA.dat", n, n_pf, n_sv);

   std::vector<float> pfp(n * 2 * n_pf), pff(n * 20 * n_pf), pfm(n * n_pf);
   std::vector<float> svp(n * 2 * n_sv), svf(n * 11 * n_sv), svm(n * n_sv);

   // device buffers reused across events (one Session, many infers)
   alpaka::PlatformCpu hostPlatform{};
   auto hostDev = alpaka::getDevByIdx(hostPlatform, 0u);
   alpaka::Platform<Acc> platform{};
   auto device = alpaka::getDevByIdx(platform, 0u);
   alpaka::Queue<Acc, alpaka::NonBlocking> queue{device};

   auto dev = [&](std::size_t nel){ return alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{nel})); };
   auto dPfp = dev(pfp.size()), dPff = dev(pff.size()), dPfm = dev(pfm.size());
   auto dSvp = dev(svp.size()), dSvf = dev(svf.size()), dSvm = dev(svm.size());
   auto h2d = [&](auto &dbuf, std::vector<float> &v){
      auto hv = alpaka::createView(hostDev, v.data(), Ext1D::all(Idx{v.size()}));
      alpaka::memcpy(queue, dbuf, hv);
   };

   check_mem("before looping");
#ifdef RANDOM
   std::mt19937 rng(111);
   std::uniform_int_distribution<int> point(1, 9), mask(0, 1);
   std::normal_distribution<float> feat(0.f, 1.f);
   std::cout << "using random inputs" << std::endl;
#else
   std::cout << "using deterministic inputs (matches CPU driver #else fills)" << std::endl;
#endif

   auto tstart = std::chrono::high_resolution_clock::now();
   for (int i = 0; i < nevts; i++) {
#ifdef RANDOM
      std::generate(pfp.begin(), pfp.end(), [&]{ return point(rng); });
      std::generate(pff.begin(), pff.end(), [&]{ return feat(rng); });
      std::generate(pfm.begin(), pfm.end(), [&]{ return mask(rng); });
      std::generate(svp.begin(), svp.end(), [&]{ return point(rng); });
      std::generate(svf.begin(), svf.end(), [&]{ return feat(rng); });
      std::generate(svm.begin(), svm.end(), [&]{ return mask(rng); });
#else
      int j = std::min(i, 9);
      std::fill(pfp.begin(), pfp.end(), int(2 + j / 2));
      std::fill(pff.begin(), pff.end(), 0.1 * (j + 1));
      std::fill(pfm.begin(), pfm.end(), 1);
      std::fill(svp.begin(), svp.end(), int(3 + j / 2));
      std::fill(svf.begin(), svf.end(), 0.2 * (j + 1));
      std::fill(svm.begin(), svm.end(), 1);
#endif

      h2d(dPfp, pfp); h2d(dPff, pff); h2d(dPfm, pfm);
      h2d(dSvp, svp); h2d(dSvf, svf); h2d(dSvm, svm);
      alpaka::wait(queue);

      auto result = s.infer(n, n_pf, dPfp, dPff, dPfm, n_sv, dSvp, dSvf, dSvm);
      alpaka::wait(queue);

      if (i % nprint == 0) {
         auto nout = alpaka::getExtentProduct(result);
         auto hOut = alpaka::allocBuf<float, Idx>(hostDev, Ext1D::all(Idx{nout}));
         alpaka::memcpy(queue, hOut, result);
         alpaka::wait(queue);
         float *o = alpaka::getPtrNative(hOut);
         std::cout << "output for i = " << i << " : ";
         for (std::size_t k = 0; k < nout; k++) std::cout << o[k] << " ";
         std::cout << "\t inputs " << pfp[0] << "  " << pff[0] << std::endl;
         check_mem("at current event");
      }
   }
   auto tend = std::chrono::high_resolution_clock::now();
   std::cout << "Total: " << std::chrono::duration<double, std::milli>(tend - tstart).count()
             << " ms for " << nevts << " events\n";
   check_mem("memory at the end");
}

int main() {
   test_particle_net(NEVTS, 1, 10, 100);
}
