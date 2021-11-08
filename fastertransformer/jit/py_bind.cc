#include <pybind11/pybind11.h>

#include "fastertransformer/jit/passes/fuse_layer_norm.h"

#include <torch/csrc/jit/passes/pass_manager.h>

namespace py = pybind11;
using namespace torch::jit;

PYBIND11_MODULE(fastseq_extension, m) {
  // auto jit = m.def_submodule("_jit");

  m.def("_jit_fuse_layer_norm", &fastseq::jit::FuseLayerNorm);
  
  RegisterPass pass([](std::shared_ptr<torch::jit::Graph>& g) {
    fastseq::jit::FuseLayerNorm(g);
  });

  registerPrePass([](std::shared_ptr<torch::jit::Graph>& g) {
    fastseq::jit::FuseLayerNorm(g);
  });
}

