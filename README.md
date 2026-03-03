# GSoC 2026 Submission — ML Inference on Heterogeneous Architectures using SOFIE

**Candidate:** Harsh Chauhan  
**Mentors:** Lorenzo Moneta, Sanjiban Sengupta  
**Deadline:** 14 March 2026  

---

## Repositories

| Repo | Branch | Purpose |
|------|--------|---------|
| [harz05/root](https://github.com/harz05/root/tree/sofie-alpaka-gsoc26) | `sofie-alpaka-gsoc26` | Operator implementations mirrored into ROOT's `tmva/sofie/` tree |
| [harz05/SOFIE](https://github.com/harz05/SOFIE/tree/gpu/alpaka) | `gpu/alpaka` | Standalone fork — full CMake + alpaka test setup, all tests runnable here |

> The standalone SOFIE repo is where tests are actually built and run.  
> ROOT's CMakeLists does not yet wire up alpaka tests, so the standalone repo
> is necessary to demonstrate correctness. Both repos contain the same operator code.

**Detailed notes for each exercise are in:**
- SOFIE fork: [`GSOC2026_SOFIE_ALPAKA.md`](https://github.com/harz05/SOFIE/blob/gpu/alpaka/GSOC2026_SOFIE_ALPAKA.md)
- ROOT fork: [`README_GSOC.md`](https://github.com/harz05/root/blob/sofie-alpaka-gsoc26/README_GSOC.md)

---

## Summary of Work

### Exercises 1–3
- Built ROOT from source with `tmva-sofie` enabled
- Ran TMVA Higgs Classification, CNN Classification, SOFIE ONNX and PyTorch tutorials
- Explored the existing alpaka GPU inference architecture in the standalone SOFIE repo

### Exercise 4 — GPU Operators via Alpaka (Tanh, Elu, Softmax, Selu)

Each operator required adding three methods to its existing `ROperator_*.hxx` class:
- `Generate_GPU_Kernel_ALPAKA` — defines the alpaka kernel struct
- `Generate_GPU_Kernel_Definitions_ALPAKA` — instantiates the kernel in the Session
- `Generate_GPU_ALPAKA` — emits the kernel launch code into the generated inference header

**Files changed:**

| File | Location in both repos |
|------|----------------------|
| `ROperator_Tanh.hxx` | `tmva/sofie/inc/TMVA/` (ROOT) · `src/SOFIE_core/inc/SOFIE/` (standalone) |
| `ROperator_Elu.hxx` | same |
| `ROperator_Softmax.hxx` | same |
| `ROperator_Selu.hxx` | same |
| `TestCustomModelsFromONNXForAlpakaCuda.cxx` | `tmva/sofie/test/` (ROOT) · `src/SOFIE_core/test/` (standalone) |

**Tests added:** `Tanh`, `Elu`, `Softmax1d`, `LinearWithSelu`

### Exercise 5 (Bonus) — Conv Operator via Alpaka (im2col + GEMM)

Implemented 2D convolution using the standard im2col + GEMM approach:
1. **Im2col GPU kernel** — one thread per output column element, handles padding/strides/dilations
2. **sofieBLAS GEMM** — multiplies weight matrix with im2col output
3. **Bias add kernel** — per-channel bias addition
4. **`GetBlasConfig()`** — added to `ROperator_Conv` so `RModel_ALPAKA` pre-configures sofieBLAS layout in the Session constructor

**Files changed:**

| File | Location |
|------|----------|
| `ROperator_Conv.hxx` | `tmva/sofie/inc/TMVA/` (ROOT) · `src/SOFIE_core/inc/SOFIE/` (standalone) |
| `RModel_ALPAKA.cxx` | standalone SOFIE only — generalized `GetBlasConfig()` handling |
| `TestCustomModelsFromONNXForAlpakaCuda.cxx` | added `ConvWithPadding` test |

**Test added:** `ConvWithPadding` — [1,1,5,5] input, 3×3 kernel, padding=1, verified against CPU reference

---

## Test Results — NVIDIA T4 GPU (Google Colab)

```
[==========] Running 10 tests from 1 test suite.
[       OK ] SofieAlpakaTest.Linear16       (316 ms)
[       OK ] SofieAlpakaTest.Linear32        (16 ms)
[       OK ] SofieAlpakaTest.Linear64         (9 ms)
[       OK ] SofieAlpakaTest.LinearWithLeakyRelu (8 ms)
[       OK ] SofieAlpakaTest.LinearWithSigmoid   (1 ms)
[       OK ] SofieAlpakaTest.Tanh             (1 ms)  ← Exercise 4
[       OK ] SofieAlpakaTest.Elu              (1 ms)  ← Exercise 4
[       OK ] SofieAlpakaTest.Softmax1d        (1 ms)  ← Exercise 4
[       OK ] SofieAlpakaTest.LinearWithSelu   (1 ms)  ← Exercise 4
[       OK ] SofieAlpakaTest.ConvWithPadding  (3 ms)  ← Exercise 5
[  PASSED  ] 10 tests.
```

---

## How to Replicate

**Option A — Google Colab (no local CUDA needed)**

Open either notebook from the SOFIE fork with a T4 GPU runtime and run all cells:
- `SOFIE_Alpaka_Test.ipynb` — full test suite (all 10 tests)
- `SOFIE_Alpaka_Exercise5_Test.ipynb` — Exercise 5 Conv test specifically

**Option B — Local build**

```bash
git clone -b gpu/alpaka https://github.com/harz05/SOFIE.git
cd SOFIE && mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -Dtesting=ON \
  -DENABLE_ALPAKA_TESTS=ON \
  -DALPAKA_BACKEND=cuda
cmake --build . --target TestCustomModelsFromONNXForAlpakaCuda -j$(nproc)
cd src/SOFIE_core/test
./TestCustomModelsFromONNXForAlpakaCuda
```

Requires: CUDA toolkit, CMake ≥ 3.16, Protobuf, ROOT, GTest.  
alpaka and sofieBLAS are fetched automatically via CMake FetchContent.
