# SOFIE Alpaka Standalone - GSoC 2026

This is a fork of the experimental SOFIE alpaka GPU inference repo:
https://github.com/ML4EP/SOFIE/tree/gpu/alpaka

This fork adds GPU implementations for the Tanh, Elu, and Softmax ONNX operators,
along with unit tests verified on an NVIDIA T4 GPU.

The operator implementations also live in the ROOT fork at:
https://github.com/harz05/root/tree/sofie-alpaka-gsoc26

This standalone repo exists because ROOT's CMakeLists does not yet wire up alpaka
tests - the standalone repo has the full CMake setup needed to actually build and
run the GPU tests.

---

## What was added

**Operator implementations** (`src/SOFIE_core/inc/SOFIE/`):
- `ROperator_Tanh.hxx` - added `Generate_GPU_Kernel_ALPAKA`, `Generate_GPU_Kernel_Definitions_ALPAKA`, `Generate_GPU_ALPAKA`
- `ROperator_Elu.hxx` - same three methods, with alpha parameter support
- `ROperator_Softmax.hxx` - same three methods, one thread per row, numerically stable

**Tests** (`src/SOFIE_core/test/`):
- `TestCustomModelsFromONNXForAlpakaCuda.cxx` - added `SofieAlpakaTest.Tanh`, `SofieAlpakaTest.Elu`, `SofieAlpakaTest.Softmax1d`

**Colab notebook**:
- `SOFIE_Alpaka_Test.ipynb` - runs the full build and test suite on a free T4 GPU

---

## Build and Test

Requires: CUDA toolkit, CMake >= 3.16, Protobuf, ROOT, GTest.
alpaka and sofieBLAS are fetched automatically via CMake FetchContent.

```bash
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -Dtesting=ON \
  -DENABLE_ALPAKA_TESTS=ON \
  -DALPAKA_BACKEND=cuda
cmake --build . -j$(nproc)
cd src/SOFIE_core/test
./TestCustomModelsFromONNXForAlpakaCuda
```

To test on a free GPU without a local CUDA setup, open `SOFIE_Alpaka_Test.ipynb`
in Google Colab with a T4 runtime and run all cells.

---

## Test Results (NVIDIA T4, Google Colab)

```
[==========] Running 8 tests from 1 test suite.
[       OK ] SofieAlpakaTest.Linear16 (443 ms)
[       OK ] SofieAlpakaTest.Linear32 (48 ms)
[       OK ] SofieAlpakaTest.Linear64 (18 ms)
[       OK ] SofieAlpakaTest.LinearWithLeakyRelu (417 ms)
[       OK ] SofieAlpakaTest.LinearWithSigmoid (2 ms)
[       OK ] SofieAlpakaTest.Tanh (1 ms)
[       OK ] SofieAlpakaTest.Elu (1 ms)
[       OK ] SofieAlpakaTest.Softmax1d (1 ms)
[  PASSED  ] 8 tests.
```
