# cpp-nnet

A C++17 port of [cnnet](https://github.com/awat/nn-upscale) — a lightweight, zero-dependency neural network framework.

## What changed from the C version

| Aspect | C (`nn.h`) | C++ (`nn.hpp`) |
|---|---|---|
| Language | C99/C11 | C++17 |
| Compiler | `gcc` | `g++` |
| Compound literals | `(Mat){...}` | member-by-member init |
| Designated inits | `.rows = ...` | not used (not standard C++) |
| `NULL` | `NULL` | `nullptr` |
| `tanhf` | built-in | renamed `nn_tanhf` (avoids `<cmath>` clash) |
| Activation macro | `#define NN_ACT` | same macro — kept for per-demo config |
| Cross-entropy | `#define NN_CROSS_ENTROPY` | same macro |
| Arena allocator | `Region` struct | same struct, same API |
| Mat struct | C struct | same struct |
| NN struct | C struct | same struct |
| Build system | GNU Makefile (gcc) | GNU Makefile (g++) |

## What stayed identical

- All mathematical operations (mat_dot, backprop, adam, cost)
- Binary `.mat` file format (magic `nn.h.mat`, same byte layout)
- OpenMP parallelism (`#pragma omp parallel for`)
- Arena allocator reset pattern (`region_reset`)
- Raylib UI (nn_render_raylib, plot_cost)
- All demo applications and their arguments

## Building

```bash
make -j          # build all targets
make xor         # build & run XOR demo
make adder       # build & run Adder demo
make upscale     # build & run Upscale demo
make morph       # build & run Morph demo
make data-shapes # generate shapes dataset
make train-shapes EPOCHS=50
make infer-shapes
```

## Prerequisites

- g++ with C++17 support
- Raylib (with pkg-config)
- OpenMP (usually bundled with GCC)

## Project Structure

```
cpp-nnet/
├── nn.hpp              # Core library (header-only, define NN_IMPLEMENTATION once)
├── nn_ui.hpp           # Raylib visualization (define NN_UI_IMPLEMENTATION once)
├── thirdparty/         # stb_image, stb_image_write, raylib, raymath, sv
├── demos/
│   ├── xor.cpp
│   ├── adder.cpp
│   ├── upscalenn.cpp
│   ├── morph2nn.cpp
│   ├── classifier_train.cpp
│   ├── classifier_infer.cpp
│   ├── mnist_gen.cpp
│   ├── shape_gen.cpp
│   ├── data/           # .mat datasets (gitignored)
│   └── models/         # .arch + .weights.mat files
├── Makefile
└── README.md
```
