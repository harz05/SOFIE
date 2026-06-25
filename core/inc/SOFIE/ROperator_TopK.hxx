#ifndef SOFIE_ROPERATOR_TOPK
#define SOFIE_ROPERATOR_TOPK

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>
#include <cstdlib>
#include <cstring>


namespace SOFIE {

template <typename T>
class ROperator_TopK final : public ROperator {

private:
   int fAttrAxis;
   int fAttrLargest;
   int fAttrSorted;

   size_t fK;
   std::string fNK;
   std::string fNX;
   std::string fNVal;
   std::string fNInd;
   std::vector<size_t> fShapeX;
   std::vector<size_t> fShapeY;
   std::string fType;

public:
   ROperator_TopK() {}
   ROperator_TopK(int attr_axis, int attr_largest, int attr_sorted, std::string nameK, std::string nameX, std::string nameVal, std::string nameInd)
      : fAttrAxis(attr_axis),
        fAttrLargest(attr_largest),
        fAttrSorted(attr_sorted),
        fNK(UTILITY::Clean_name(nameK)),
        fNX(UTILITY::Clean_name(nameX)),
        fNVal(UTILITY::Clean_name(nameVal)),
        fNInd(UTILITY::Clean_name(nameInd)){
            fInputTensorNames = { fNX, fNK };
            fOutputTensorNames = { fNVal, fNInd };
        }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
         ETensorType ret = input[0];
         return {ret, ret};
      }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      if (input.size() != 2) {
         throw std::runtime_error("SOFIE TopK Op Shape Inference needs exactly 2 input tensors");
      }

      auto shape = input[0]; // Shape format: [ m x n x o x p ... ]

      // set the dimension at the specified axis to k  (fAttrAxis is checked before that is in the correct range
      shape[fAttrAxis] = fK; // Modified shape: [ m x n x k x p ... ]
      return {shape, shape};
   }


   void Initialize(RModel& model) override {
      if (model.CheckIfTensorAlreadyExist(fNX) == false) {
         // input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("SOFIE TopK Op Input Tensor is not found in model");
      }
      if (model.CheckIfTensorAlreadyExist(fNK) == false) {
         // input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("SOFIE TopK Op Input Tensor i.e. K is not found in model");
      }

      fShapeX = model.GetTensorShape(fNX);
      auto fShapeK = model.GetTensorShape(fNK);
      auto kptr = static_cast<int64_t *>(model.GetInitializedTensorData(fNK).get());
      fK = *kptr;
      model.SetNotWritableInitializedTensor(fNK);
      fAttrAxis = fAttrAxis < 0 ? fShapeX.size() + fAttrAxis : fAttrAxis;
      if(static_cast<size_t>(fAttrAxis) >=  fShapeX.size()){
         throw
            std::runtime_error("TMVA::SOFIE ONNX TopK op axis = "+ std::to_string(fAttrAxis) +" value exeeds size of tensor " +fNX+" of size "+std::to_string(fShapeX.size())+" .");
      }
      // fK cannot be larger that axis dimension
      fK = std::min(fK, fShapeX[fAttrAxis]);
      // if(fK>fShapeX[fAttrAxis]){
      //    throw
      //       std::runtime_error("TMVA::SOFIE ONNX TopK op k = "+ std::to_string(fK) +" value exeeds value of tensor " +fNX+" of size "+fShapeX.size()+" at axis= "+std::to_string(fAttrAxis)+".");
      // }
      // fShapeX = model.GetTensorShape(fNX); //  [ m x n x o x p ... ]
      // if(k[0]>=fShapeX.size()){
      //    throw
      //       std::runtime_error("TMVA::SOFIE ONNX TopK op k = "+ std::to_string(k[0]) +"value exeeds size of tensor " +fNX+" of size "+fShapeX.size()+" .");
      // }
      // fShapeY.push_back(2);
      // for (auto i : fShapeX)
      //    fShapeY.push_back(i); //  [ 2 x m x n x o x p ... ]
      // size_t axis = fAttrAxis < 0 ? fShapeX.size() + fAttrAxis : fAttrAxis;
      // fShapeY[axis] = k[0]; //  [ 2 x m x n x K x p ... ]
      fShapeY=ShapeInference({fShapeX,fShapeK})[0];

      // for(int i=0;i<fShapeX.size();i++)
      // std::cout<<fShapeX[i]<<" ";
      // std::cout<<"\ny size -> "<<fShapeY.size()<<std::endl;


      model.AddIntermediateTensor(fNVal, model.GetTensorType(fNX), fShapeY);
      // output indices should be an int64 tensor
      model.AddIntermediateTensor(fNInd, ETensorType::INT64, fShapeY);
      fType = ConvertTypeToString(model.GetTensorType(fNX));
   }

   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShapeX.empty()) {
         throw std::runtime_error("SOFIE Operator TopK called to Generate without being initialized first");
      }
      std::stringstream out;
      size_t size = fShapeX.size();
      size_t axis = fAttrAxis < 0 ? size + fAttrAxis : fAttrAxis;
      out << "\n" << SP << "//------ TopK\n";

      size_t length=ConvertShapeToLength(fShapeX);
      auto strideX = UTILITY::ComputeStrideFromShape(fShapeX);
      auto strideY = UTILITY::ComputeStrideFromShape(fShapeX);
      // we perform loop on dimension before sorted axis and after sorted axis
      size_t n_before = (axis>0) ? length/strideX[axis-1] : 1;
      size_t n_after = strideX[axis];
      size_t n_elements = fShapeX[axis]; // number of elements to be sorted

      // }
      out << SP << "{\n"; // to define a separate scope for the operator code
      out << SP << "std::vector<std::pair<float,int64_t>> elements(" << n_elements << ");\n";
      // loop on elements before
      if (n_before > 1) {
         out << SP << "for (size_t i = 0; i < " << n_before << "; i++) {\n";
         out << SP << SP << "size_t xoffset = i*" << strideX[axis-1] << ";\n";
         out << SP << SP << "size_t yoffset = i*" << strideY[axis-1] << ";\n";
         out << SP;
      } else {
         out << SP << "size_t xoffset = 0;\n";
         out << SP << "size_t yoffset = 0;\n";
      }
      if (n_after > 1)
         out << SP << "for (size_t j = 0; j < " << n_after << "; j++) {\n";
      else
         out << SP << "const size_t j = 0;\n";

      // copy elements to be sorted in vector of pair
      out << SP << SP << "for (size_t l = 0; l < " << n_elements << "; l++) {\n";
      out << SP << SP << SP << "elements[l] = std::make_pair(tensor_" << fNX << "[xoffset + " << strideX[axis] << "*l + j], l);\n";
      out << SP << SP << "}\n";

      if (fAttrSorted) {
         if (fAttrLargest) {
            out<<SP<<SP << "std::partial_sort(elements.begin(),elements.begin()+" << fK << ",elements.end()," <<
               "[](std::pair<float,int64_t>a,std::pair<float,int64_t>b){return (a.first!=b.first) ? (a.first>b.first) : a.second < b.second;});\n";

         } else
            out<<SP<<SP << "std::partial_sort(elements.begin(),elements.begin()+" << fK << ",elements.end()," <<
            "[](std::pair<float,int64_t>a,std::pair<float,int64_t>b){return (a.first!=b.first) ? (a.first<b.first) : a.second < b.second;});\n";
      } else
         // in this case we don;t need to return sorted elements, so we keep same order as before
         out<<SP<<SP << "std::partial_sort(elements.begin(),elements.begin()+" << fK << ",elements.end());\n";

      // copy the selected elements in the output
      out << SP << SP << "for (size_t l = 0; l < " << fK << "; l++) {\n";
      out << SP << SP << SP << "tensor_" << fNVal   << "[yoffset + " << strideY[axis] << "*l + j] = elements[l].first;\n";
      out << SP << SP << SP << "tensor_" << fNInd << "[yoffset + " << strideY[axis] << "*l + j] = elements[l].second;\n";
      out << SP << SP << "}\n";
      if (n_after > 1) out << SP << SP << "}\n";
      if (n_before> 1) out << SP << "}\n";
      out << SP << "}\n"; // end operator scope
      return out.str();
   }

   // next power of two >= the axis length (the bitonic network needs a power-of-2 size)
   size_t TopKPaddedAxis() const {
      size_t axis = fAttrAxis < 0 ? fShapeX.size() + fAttrAxis : fAttrAxis;
      size_t n = fShapeX[axis];
      size_t p = 1; while (p < n) p <<= 1;
      return p;
   }
   // threads per block: cover the paddedN/2 comparator pairs, capped at 1024, one warp
   // kernel and launch both call this
   size_t TopKBlockThreads() const {
      size_t pairs = TopKPaddedAxis() / 2;
      size_t bt = (pairs < 1024) ? pairs : 1024;
      if (bt < 32) bt = 32;
      return bt;
   }

   // -------- GPU algorithm selection (codegen time) --------------------------
   // Two GPU paths share the same kernel signature and launch args:
   //   Bitonic    : block-per-row full sort in shared memory. Portable across all
   //                alpaka backends; the default. Bounded by the 48KB shared cap.
   //   WarpSelect : FAISS-style warp-per-row k-selection. Exact, no shared cap, but
   //                warp-cooperative -> CUDA/warp-32 only. Opt-in for benchmarking.
   enum class GpuTopKAlgo { Bitonic, WarpSelect };

   // thread-queue length t (FAISS picks it from K)
   size_t WarpSelectTQ() const {
      if (fK <= 32)  return 2;
      if (fK <= 128) return 3;
      if (fK <= 256) return 4;
      return 8;
   }

   // WarpSelect is opt-in via env (it is CUDA-only, so we never pick it automatically;
   // bitonic stays the portable default and still owns the over-cap case below).
   GpuTopKAlgo SelectGpuAlgo() const {
      if (const char *e = std::getenv("SOFIE_TOPK_GPU_ALGO")) {
         if (std::strcmp(e, "warpselect") == 0) return GpuTopKAlgo::WarpSelect;
         if (std::strcmp(e, "bitonic") == 0)    return GpuTopKAlgo::Bitonic;
      }
      return GpuTopKAlgo::Bitonic;
   }

   // C++ expression for the ONNX best-first comparator: value (OP) then lower index wins
   std::string WS_better(const std::string &va, const std::string &ia,
                         const std::string &vb, const std::string &ib) const {
      std::string OP = fAttrLargest ? ">" : "<";
      return "((" + va + " " + OP + " " + vb + ") || (" + va + " == " + vb +
             " && " + ia + " < " + ib + "))";
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string /*opName*/) override {
      if (fShapeX.empty())
         throw std::runtime_error("SOFIE Operator TopK called to Generate without being initialized first");
      return (SelectGpuAlgo() == GpuTopKAlgo::WarpSelect) ? EmitWarpSelectKernel()
                                                          : EmitBitonicKernel();
   }

   // We have one block per slice. Cache the row in shared memory, bitonic-sort it
   // best-first (indices ride along for the tie-break), then write the first K.
   std::string EmitBitonicKernel() {
      size_t axis = fAttrAxis < 0 ? fShapeX.size() + fAttrAxis : fAttrAxis;
      std::string NE = std::to_string(fShapeX[axis]); // real axis length
      std::string PAD = std::to_string(TopKPaddedAxis()); // next power of two >= NE
      std::string PADH = std::to_string(TopKPaddedAxis() / 2); // number of comparator pairs
      std::string BT = std::to_string(TopKBlockThreads()); // threads per block
      std::string K = std::to_string(fK);
      std::string OP = fAttrLargest ? ">" : "<"; // best-first value comparator
      std::string SENT = fAttrLargest ? "std::numeric_limits<T>::lowest()"
                                      : "std::numeric_limits<T>::max()"; // padded slots never win
      std::string kname = "TopKKernel_" + fNVal;

      // shared-memory budget guard (caches the whole padded row)
      size_t valBytes = (fType == "double" || fType == "int64_t") ? 8 : 4;
      if (TopKPaddedAxis() * (valBytes + 8) > 48u * 1024u)
         throw std::runtime_error("SOFIE TopK GPU: axis length " + NE +
            " too long for shared-memory bitonic top-K (try SOFIE_TOPK_GPU_ALGO=warpselect on CUDA)");

      std::string op;
      op  = "\n//------ TopK_KERNEL_ALPAKA (block-per-row bitonic)\n";
      op += SP + "struct " + kname + " {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
      op += SP + SP + SP + "TAcc const& acc,\n";
      op += SP + SP + SP + "T const* __restrict__ x,\n";
      op += SP + SP + SP + "T* __restrict__ vals,\n";
      op += SP + SP + SP + "int64_t* __restrict__ inds,\n";
      op += SP + SP + SP + "std::size_t const numSlices,\n";
      op += SP + SP + SP + "std::size_t const nAfter,\n";
      op += SP + SP + SP + "std::size_t const strideXAxis,\n";
      op += SP + SP + SP + "std::size_t const strideXBefore,\n";
      op += SP + SP + SP + "std::size_t const strideYAxis,\n";
      op += SP + SP + SP + "std::size_t const strideYBefore) const {\n\n";

      // shared row buffers (values + indices), declared before the early return
      op += SP + SP + SP + "auto& sv = alpaka::declareSharedVar<T[" + PAD + "], __COUNTER__>(acc);\n";
      op += SP + SP + SP + "auto& si = alpaka::declareSharedVar<int64_t[" + PAD + "], __COUNTER__>(acc);\n";
      op += SP + SP + SP + "auto const slice = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];\n";
      op += SP + SP + SP + "auto const tid   = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (slice >= numSlices) return;\n\n";

      op += SP + SP + SP + "std::size_t const ib = slice / nAfter;\n";
      op += SP + SP + SP + "std::size_t const jb = slice % nAfter;\n";
      op += SP + SP + SP + "std::size_t const xbase = ib * strideXBefore + jb;\n";
      op += SP + SP + SP + "std::size_t const ybase = ib * strideYBefore + jb;\n\n";

      // 1) load row into shared, pad the tail with a sentinel that never wins
      op += SP + SP + SP + "for (std::size_t l = tid; l < " + PAD + "u; l += " + BT + "u) {\n";
      op += SP + SP + SP + SP + "if (l < " + NE + "u) { sv[l] = x[xbase + strideXAxis * l]; si[l] = (int64_t)l; }\n";
      op += SP + SP + SP + SP + "else { sv[l] = " + SENT + "; si[l] = (int64_t)" + NE + "; }\n";
      op += SP + SP + SP + "}\n";
      op += SP + SP + SP + "alpaka::syncBlockThreads(acc);\n\n";

      // 2) bitonic sort: kk = bitonic seq size, jj = compare distance, best ends at index 0.
      // each thread owns ONE comparator pair (i, i^jj)
      op += SP + SP + SP + "for (std::size_t kk = 2u; kk <= " + PAD + "u; kk <<= 1) {\n";
      op += SP + SP + SP + SP + "for (std::size_t jj = kk >> 1; jj > 0u; jj >>= 1) {\n";
      op += SP + SP + SP + SP + SP + "for (std::size_t t = tid; t < " + PADH + "u; t += " + BT + "u) {\n";
      op += SP + SP + SP + SP + SP + SP + "std::size_t const i = ((t & ~(jj - 1u)) << 1) | (t & (jj - 1u));\n";
      op += SP + SP + SP + SP + SP + SP + "std::size_t const p = i | jj;\n";
      op += SP + SP + SP + SP + SP + SP + "T av = sv[i]; T bv = sv[p];\n";
      op += SP + SP + SP + SP + SP + SP + "int64_t ai = si[i]; int64_t bi = si[p];\n";
      op += SP + SP + SP + SP + SP + SP + "bool const firstFirst = (av " + OP + " bv) || (av == bv && ai < bi);\n";
      op += SP + SP + SP + SP + SP + SP + "bool const dir = ((i & kk) == 0u);\n";
      op += SP + SP + SP + SP + SP + SP + "bool const sw  = (firstFirst != dir);\n";
      op += SP + SP + SP + SP + SP + SP + "sv[i] = sw ? bv : av; sv[p] = sw ? av : bv;\n";
      op += SP + SP + SP + SP + SP + SP + "si[i] = sw ? bi : ai; si[p] = sw ? ai : bi;\n";
      op += SP + SP + SP + SP + SP + "}\n";
      op += SP + SP + SP + SP + SP + "alpaka::syncBlockThreads(acc);\n";
      op += SP + SP + SP + SP + "}\n";
      op += SP + SP + SP + "}\n\n";

      // 3) write top-K (already best-first)
      op += SP + SP + SP + "for (std::size_t s = tid; s < " + K + "u; s += " + BT + "u) {\n";
      op += SP + SP + SP + SP + "vals[ybase + strideYAxis * s] = sv[s];\n";
      op += SP + SP + SP + SP + "inds[ybase + strideYAxis * s] = si[s];\n";
      op += SP + SP + SP + "}\n";

      op += SP + SP + "}\n";// end operator()
      op += SP + "};\n";// end struct
      return op;
   }

   // top-K of (warp queue + thread queues): load into cand, unrolled lane-stride
   // bitonic sort (shfl_xor across lanes, register swaps within a lane), keep K best as the new
   // warp queue, reset the buffer, refresh bar
   std::string EmitWarpMerge(const std::string &ind) const {
      size_t axis = fAttrAxis < 0 ? fShapeX.size() + fAttrAxis : fAttrAxis;
      std::string NE = std::to_string(fShapeX[axis]);
      const int W = 32;
      size_t TQ = WarpSelectTQ();
      size_t WQS = (fK + W - 1) / W;
      size_t M = WQS + TQ;
      size_t MPAD = 1; while (MPAD < M) MPAD <<= 1;
      size_t total = static_cast<size_t>(W) * MPAD;
      std::string SENT = fAttrLargest ? "std::numeric_limits<T>::lowest()"
                                      : "std::numeric_limits<T>::max()";
      std::string s;
      for (size_t c = 0; c < WQS; ++c) {
         std::string C = std::to_string(c);
         s += ind + "cand[" + C + "] = wqv[" + C + "]; candI[" + C + "] = wqi[" + C + "];\n";
      }
      for (size_t j = 0; j < TQ; ++j) {
         std::string slot = std::to_string(WQS + j), J = std::to_string(j);
         s += ind + "cand[" + slot + "] = (" + J + " < tqn) ? tqv[" + J + "] : " + SENT +
              "; candI[" + slot + "] = (" + J + " < tqn) ? tqi[" + J + "] : (int64_t)" + NE + ";\n";
      }
      for (size_t p = M; p < MPAD; ++p) {
         std::string P = std::to_string(p);
         s += ind + "cand[" + P + "] = " + SENT + "; candI[" + P + "] = (int64_t)" + NE + ";\n";
      }
      for (size_t kk = 2; kk <= total; kk <<= 1) {
         for (size_t jj = kk >> 1; jj > 0; jj >>= 1) {
            std::string KK = std::to_string(kk), WW = std::to_string(W);
            if (jj >= static_cast<size_t>(W)) {
               size_t sd = jj / W;// same lane,different slot
               for (size_t slot = 0; slot < MPAD; ++slot) {
                  size_t ps = slot ^ sd;
                  if (ps <= slot) continue;
                  std::string S = std::to_string(slot), P = std::to_string(ps);
                  s += ind + "{ std::size_t g = lane + " + WW + "u*" + S + "u;\n";
                  s += ind + "  bool ff = " + WS_better("cand[" + S + "]", "candI[" + S + "]",
                                                        "cand[" + P + "]", "candI[" + P + "]") + ";\n";
                  s += ind + "  bool sw = (ff != ((g & " + KK + "u) == 0u));\n";
                  s += ind + "  T av=cand[" + S + "], bv=cand[" + P + "];"
                             " cand[" + S + "]=sw?bv:av; cand[" + P + "]=sw?av:bv;\n";
                  s += ind + "  int64_t ai=candI[" + S + "], bi=candI[" + P + "];"
                             " candI[" + S + "]=sw?bi:ai; candI[" + P + "]=sw?ai:bi; }\n";
               }
            } else {
               std::string JJ = std::to_string(jj);  // different lane, same slot (shfl)
               for (size_t slot = 0; slot < MPAD; ++slot) {
                  std::string S = std::to_string(slot);
                  s += ind + "{ T pv = alpaka::warp::shfl_xor(acc, cand[" + S + "], " + JJ + ");\n";
                  s += ind + "  int64_t pi = alpaka::warp::shfl_xor(acc, candI[" + S + "], " + JJ + ");\n";
                  s += ind + "  bool low = ((lane & " + JJ + "u) == 0u);\n";
                  s += ind + "  T lv = low ? cand[" + S + "] : pv;  int64_t li = low ? candI[" + S + "] : pi;\n";
                  s += ind + "  T hv = low ? pv : cand[" + S + "];  int64_t hi = low ? pi : candI[" + S + "];\n";
                  s += ind + "  std::size_t glow = (lane & ~" + JJ + "u) + " + WW + "u*" + S + "u;\n";
                  s += ind + "  bool ff = " + WS_better("lv", "li", "hv", "hi") + ";\n";
                  s += ind + "  bool sw = (ff != ((glow & " + KK + "u) == 0u));\n";
                  s += ind + "  cand[" + S + "] = sw ? pv : cand[" + S + "]; candI[" + S + "] = sw ? pi : candI[" + S + "]; }\n";
               }
            }
         }
      }
      for (size_t c = 0; c < WQS; ++c) {
         std::string C = std::to_string(c);
         s += ind + "wqv[" + C + "] = cand[" + C + "]; wqi[" + C + "] = candI[" + C + "];\n";
      }
      s += ind + "tqn = 0;\n";
      size_t bslot = (fK - 1) / W, blane = (fK - 1) % W;
      s += ind + "bar = alpaka::warp::shfl(acc, wqv[" + std::to_string(bslot) + "], " +
           std::to_string(static_cast<int>(blane)) + ");\n";
      return s;
   }

   // FAISS WarpSelect (arXiv:1702.08734) implementation: one warp per row topK
   // register thread queue (buffer) + warp queue (running top-K); a warp-uniform ballot
   // triggers the merge. Assumes warpSize as 32 for CUDA
   // note- a non-warp backend would need the bitonic fallback.
   std::string EmitWarpSelectKernel() {
      size_t axis = fAttrAxis < 0 ? fShapeX.size() + fAttrAxis : fAttrAxis;
      std::string NE = std::to_string(fShapeX[axis]);
      std::string K = std::to_string(fK);
      const int W = 32;
      size_t TQ = WarpSelectTQ();
      size_t WQS = (fK + W - 1) / W;
      size_t M = WQS + TQ;
      size_t MPAD = 1; while (MPAD < M) MPAD <<= 1;
      std::string SENT = fAttrLargest ? "std::numeric_limits<T>::lowest()"
                                      : "std::numeric_limits<T>::max()";
      std::string kname = "WarpSelectKernel_" + fNVal;
      std::string TQs = std::to_string(TQ);

      std::string op;
      op  = "\n//------ TopK_KERNEL_ALPAKA (warp-per-row WarpSelect, exact)\n";
      op += SP + "struct " + kname + " {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
      op += SP + SP + SP + "TAcc const& acc,\n";
      op += SP + SP + SP + "T const* __restrict__ x,\n";
      op += SP + SP + SP + "T* __restrict__ vals,\n";
      op += SP + SP + SP + "int64_t* __restrict__ inds,\n";
      op += SP + SP + SP + "std::size_t const numSlices,\n";
      op += SP + SP + SP + "std::size_t const nAfter,\n";
      op += SP + SP + SP + "std::size_t const strideXAxis,\n";
      op += SP + SP + SP + "std::size_t const strideXBefore,\n";
      op += SP + SP + SP + "std::size_t const strideYAxis,\n";
      op += SP + SP + SP + "std::size_t const strideYBefore) const {\n\n";

      op += SP+SP+SP + "constexpr int WARP = " + std::to_string(W) + ";\n";
      op += SP+SP+SP + "auto const tib  = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];\n";
      op += SP+SP+SP + "auto const bid  = alpaka::getIdx<alpaka::Grid,  alpaka::Blocks>(acc)[0];\n";
      op += SP+SP+SP + "auto const bdim = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0];\n";
      op += SP+SP+SP + "std::size_t const lane  = tib % WARP;\n";
      op += SP+SP+SP + "std::size_t const slice = (static_cast<std::size_t>(bid) * bdim + tib) / WARP;\n";
      op += SP+SP+SP + "if (slice >= numSlices) return;\n\n";
      op += SP+SP+SP + "std::size_t const ib = slice / nAfter;\n";
      op += SP+SP+SP + "std::size_t const jb = slice % nAfter;\n";
      op += SP+SP+SP + "std::size_t const xbase = ib * strideXBefore + jb;\n";
      op += SP+SP+SP + "std::size_t const ybase = ib * strideYBefore + jb;\n\n";

      op += SP+SP+SP + "T wqv[" + std::to_string(WQS) + "]; int64_t wqi[" + std::to_string(WQS) + "];\n";
      op += SP+SP+SP + "T tqv[" + TQs + "]; int64_t tqi[" + TQs + "];\n";
      op += SP+SP+SP + "T cand[" + std::to_string(MPAD) + "]; int64_t candI[" + std::to_string(MPAD) + "];\n";
      for (size_t s = 0; s < WQS; ++s) {
         std::string S = std::to_string(s);
         op += SP+SP+SP + "wqv[" + S + "] = " + SENT + "; wqi[" + S + "] = (int64_t)" + NE + ";\n";
      }
      op += SP+SP+SP + "int tqn = 0;\n";
      op += SP+SP+SP + "T bar = " + SENT + ";\n\n";

      op += SP+SP+SP + "std::size_t const N = " + NE + "u;\n";
      op += SP+SP+SP + "std::size_t const padded = ((N + WARP - 1u) / WARP) * WARP;\n";
      op += SP+SP+SP + "for (std::size_t l = lane; l < padded; l += WARP) {\n";
      op += SP+SP+SP+SP + "bool const ok = (l < N);\n";
      op += SP+SP+SP+SP + "T v = ok ? x[xbase + strideXAxis * l] : " + SENT + ";\n";
      op += SP+SP+SP+SP + "int64_t id = ok ? (int64_t)l : (int64_t)N;\n";
      op += SP+SP+SP+SP + "bool keep = ok && " + WS_better("v","id","bar","(int64_t)N") + ";\n";
      op += SP+SP+SP+SP + "bool needMerge = keep && (tqn == " + TQs + ");\n";
      op += SP+SP+SP+SP + "if (alpaka::warp::ballot(acc, needMerge ? 1 : 0)) {\n";
      op += EmitWarpMerge(SP+SP+SP+SP+SP);
      op += SP+SP+SP+SP+SP + "keep = ok && " + WS_better("v","id","bar","(int64_t)N") + ";\n";
      op += SP+SP+SP+SP + "}\n";
      op += SP+SP+SP+SP + "if (keep && tqn < " + TQs + ") { tqv[tqn] = v; tqi[tqn] = id; ++tqn; }\n";
      op += SP+SP+SP + "}\n\n";

      op += SP+SP+SP + "// final flush\n";
      op += EmitWarpMerge(SP+SP+SP);
      op += "\n";

      for (size_t s = 0; s < WQS; ++s) {
         std::string S = std::to_string(s);
         op += SP+SP+SP + "{ std::size_t r = lane + WARP*" + S + "u;\n";
         op += SP+SP+SP + "  if (r < " + K + "u) { vals[ybase + strideYAxis * r] = wqv[" + S +
               "]; inds[ybase + strideYAxis * r] = wqi[" + S + "]; } }\n";
      }

      op += SP + SP + "}\n";
      op += SP + "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string /*opName*/) override {
      if (SelectGpuAlgo() == GpuTopKAlgo::WarpSelect)
         return SP + "WarpSelectKernel_" + fNVal + " warpKernel_" + fNVal + ";\n";
      return SP + "TopKKernel_" + fNVal + " topKernel_" + fNVal + ";\n";
   }

   std::vector<std::string> GetStdLibs() override {
      return { std::string("limits"), std::string("cstdint"),
               std::string("cstring"), std::string("cstdlib") };
   }

   // the geometry is computed here at codegen and passed as args matching the kernel signature.
   std::string Generate_GPU_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      if (fShapeX.empty())
         throw std::runtime_error("SOFIE Operator TopK called to Generate without being initialized first");

      size_t axis = fAttrAxis < 0 ? fShapeX.size() + fAttrAxis : fAttrAxis;
      size_t length = ConvertShapeToLength(fShapeX);
      auto strideX = UTILITY::ComputeStrideFromShape(fShapeX);
      auto strideY = UTILITY::ComputeStrideFromShape(fShapeY); // output is shorter along axis (K, not N_EL)
      size_t n_after = strideX[axis];
      size_t n_before = (axis > 0) ? length / strideX[axis-1] : 1;
      size_t numSlices = n_before * n_after;
      size_t strideX_axis = strideX[axis];
      size_t strideY_axis = strideY[axis];
      size_t strideX_before = (axis > 0) ? strideX[axis-1] : 0; // 0 is safe: i==0 when axis==0
      size_t strideY_before = (axis > 0) ? strideY[axis-1] : 0;

      if (SelectGpuAlgo() == GpuTopKAlgo::WarpSelect) {
         const size_t WARP = 32, wsBlock = 256;
         std::stringstream w;
         w << "\n//-- TopK_GPU_ALPAKA (WarpSelect, one warp per slice)\n";
         w << SP << "alpaka::WorkDivMembers<Dim, Idx> workDiv_" << fNVal << "(\n";
         w << SP << SP << "Vec::all(static_cast<Idx>((" << numSlices << "u * " << WARP << "u + "
           << wsBlock << "u - 1u) / " << wsBlock << "u)),\n";
         w << SP << SP << "Vec::all(Idx{" << wsBlock << "u}),\n";
         w << SP << SP << "Vec::all(Idx{1u}));\n";
         w << SP << "auto task_" << fNVal << " = alpaka::createTaskKernel<Acc>(workDiv_" << fNVal
           << ", warpKernel_" << fNVal
           << ", alpaka::getPtrNative(deviceBuf_" << fNX << ")"
           << ", alpaka::getPtrNative(deviceBuf_" << fNVal << ")"
           << ", alpaka::getPtrNative(deviceBuf_" << fNInd << ")"
           << ", static_cast<std::size_t>(" << numSlices     << "u)"
           << ", static_cast<std::size_t>(" << n_after       << "u)"
           << ", static_cast<std::size_t>(" << strideX_axis  << "u)"
           << ", static_cast<std::size_t>(" << strideX_before<< "u)"
           << ", static_cast<std::size_t>(" << strideY_axis  << "u)"
           << ", static_cast<std::size_t>(" << strideY_before<< "u));\n";
         w << SP << "alpaka::enqueue(queue, task_" << fNVal << ");\n";
         return w.str();
      }

      std::stringstream out;
      out << "\n//-- TopK_GPU_ALPAKA\n";
      const size_t blockThreads = TopKBlockThreads();
      out << SP << "alpaka::WorkDivMembers<Dim, Idx> workDiv_" << fNVal << "(\n";
      out << SP << SP << "Vec::all(static_cast<Idx>(" << numSlices << ")),\n";
      out << SP << SP << "Vec::all(Idx{" << blockThreads << "u}),\n";
      out << SP << SP << "Vec::all(Idx{1u}));\n";
      out << SP << "auto task_" << fNVal << " = alpaka::createTaskKernel<Acc>(workDiv_" << fNVal << ", topKernel_" << fNVal
          << ", alpaka::getPtrNative(deviceBuf_" << fNX << ")"
          << ", alpaka::getPtrNative(deviceBuf_" << fNVal << ")"
          << ", alpaka::getPtrNative(deviceBuf_" << fNInd << ")"
          << ", static_cast<std::size_t>(" << numSlices     << "u)"
          << ", static_cast<std::size_t>(" << n_after       << "u)"
          << ", static_cast<std::size_t>(" << strideX_axis  << "u)"
          << ", static_cast<std::size_t>(" << strideX_before<< "u)"
          << ", static_cast<std::size_t>(" << strideY_axis  << "u)"
          << ", static_cast<std::size_t>(" << strideY_before<< "u));\n";
      out << SP << "alpaka::enqueue(queue, task_" << fNVal << ");\n";
      return out.str();
   }

};

} // nameSPace SOFIE


#endif // SOFIE_ROPERATOR_TOPK
