## llama2-notorch

<p align="center">
  <img src="assets/llama_cute.jpg" width="300" height="300" alt="Cute Llama">
</p>

Train and inference a baby [Llama 2](https://ai.meta.com/llama/) model — **no pip dependencies**. [notorch](https://github.com/ariannamethod/notorch) is the engine.

Training uses Python (stdlib only: ctypes, os, math, random) calling into `libnotorch` — a pure C neural network library. No PyTorch. No numpy. The [Chuck optimizer](ariannamethod/chuck.py) replaces Adam. Inference uses the existing pure C [run.c](run.c).

This is a fork of [llama2.c](https://github.com/karpathy/llama2.c) by Andrej Karpathy, with the entire PyTorch training pipeline replaced by notorch, following the [nanoGPT-notorch](https://github.com/ariannamethod/nanoGPT-notorch) reference.

## quick start

Build the notorch engine (requires only a C compiler):

```bash
make notorch
```

Train on Dracula (character-level, included in repo):

```bash
python train.py
```

That's it. No `pip install torch`. No 2.7 GB download. Just a C compiler and Python stdlib.

## training

The training script trains a Llama 2 architecture model on `dracula.txt` (character-level). Default config trains a ~15M param model:

```bash
python train.py --dim 288 --layers 6 --heads 6 --ctx 256 --steps 12000
```

You can customize all hyperparameters:

```bash
python train.py --dataset mytext.txt --dim 512 --layers 8 --heads 8 --ctx 512 --steps 20000 --lr 3e-4
```

Resume training from a checkpoint:

```bash
python train.py --resume weights/llama2.bin
```

**Architecture**: RMSNorm, SwiGLU FFN, RoPE positional embeddings, multi-head causal attention — identical to Llama 2.

**Optimizer**: [Chuck](ariannamethod/chuck.py) — a self-aware optimizer that replaces Adam. Loss-aware damping, gradient monitoring, memory. Implemented in C (`nt_tape_chuck_step`).

**LR schedule**: Cosine with warmup (10% of total steps).

## sampling

Generate text from a trained model:

```bash
python sample.py --checkpoint weights/llama2.bin --prompt "CHAPTER" --max_tokens 200
```

## what is notorch?

[notorch](https://github.com/ariannamethod/notorch) — pure C neural network library. No pip. BLAS via Accelerate/OpenBLAS.

The `ariannamethod/` directory contains:
- `notorch.c` / `notorch.h` — the C engine (tensors, autograd tape, ops, Chuck optimizer)
- `notorch_nn.py` — Python ctypes wrapper (Module, Linear, Embedding, RMSNorm, etc.)
- `chuck.py` — Chuck optimizer Python interface

Build: `cd ariannamethod && cc -std=c11 -O2 -fPIC -shared -o libnotorch.so notorch.c -lm`

## C inference

The pure C inference engine [run.c](run.c) is unchanged and still works with pre-existing model checkpoints:

```bash
make run
./run stories15M.bin
```

See the [Makefile](Makefile) for additional build targets (`runfast`, `runomp`, etc.).

## int8 quantization

The (default) script [run.c](run.c) uses a float32 forward pass. The quantized forward pass is implemented in [runq.c](runq.c). See the original llama2.c documentation for details on int8 quantization.

## performance

There are many ways to potentially speed up this code depending on your system. Have a look at the [Makefile](Makefile), which contains a lot of notes. The `make run` command currently uses the `-O3` optimization by default.

To get a much better performance, try to compile with `make runfast`. This turns on the `-Ofast` flag.

**OpenMP**. Big improvements can also be achieved by compiling with OpenMP:

```bash
make runomp
OMP_NUM_THREADS=4 ./run out/model.bin
```

## platforms

On **Windows**, use `build_msvc.bat` in a Visual Studio Command Prompt to build with msvc, or you can use `make win64` to use mingw compiler toolchain.

On **Centos 7**, **Amazon Linux 2018** use `rungnu` Makefile target: `make rungnu` or `make runompgnu`.

On **Mac**, use clang from brew for openmp build: `make runomp CC=/opt/homebrew/opt/llvm/bin/clang`

## tests

```bash
pip install pytest requests
make run
make notorch
pytest
```

This runs tests for the C inference engine (`test_runc`), notorch build verification (`test_notorch_build`), and notorch import (`test_notorch_import`).

There are also some tests in C, in the file [test.c](test.c): `make testcc`

## ack

Original llama2.c by [Andrej Karpathy](https://github.com/karpathy/llama2.c). notorch by [Arianna Method](https://github.com/ariannamethod/notorch).

## License

MIT
