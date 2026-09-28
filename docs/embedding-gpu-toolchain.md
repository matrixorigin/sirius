# Pixi provider for MatrixOne embedding

Refs [MatrixOne #28966](https://github.com/matrixorigin/matrixone/issues/28966).

The `mo` Pixi environment is the sole CUDA/RAPIDS build provider for the
combined MatrixOne/Sirius binary. It locks CUDA 13.3, GCC 14, cuVS/cuDF
26.08.01 and RMM. Sirius's ordinary CUDA 13.2 and `cuda12` profiles are
unchanged; a compatible NVIDIA driver and GPU remain host requirements.

```sh
pixi run --frozen -e mo mo-build-embedding-sdk
```

The profile-specific configure task disables `sccache` only for CUDA. The
ordinary presets retain their launchers; the `mo` SDK avoids the observed NVCC
temporary-PTX failure with CUDA 13.3 and sccache 0.15.

The existing `embedding-sdk/link.json` records the exact C smoke compiler,
link arguments, native artifacts, and source provenance. It does not export a
second GPU toolchain manifest. Build MO's GPU native libraries under the same
activated `mo` Pixi environment. The MatrixOne bridge verifies its native
generation and that the SDK compiler belongs to this prefix, then hashes and
packages the actual ELF dependency closure of both the SDK and `libmo`.
Linker stubs and NVIDIA driver libraries are never packaged as runtime
implementations. The deployed binary uses relative paths to staged shared
libraries and still requires the host driver.

The embedding SDK is build-tree metadata, not a relocatable artifact. Rebuild
after changing the Pixi lock or environment. Compilation and packaging do not
establish GPU execution: the combined MO cuVS/Sirius test with two streams is
the runtime gate.
