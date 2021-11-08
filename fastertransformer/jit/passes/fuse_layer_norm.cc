#include <fastertransformer/jit/passes/fuse_layer_norm.h>

#include <torch/csrc/jit/passes/subgraph_rewrite.h>

#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/fuse_linear.h>
#include <torch/csrc/jit/passes/quantization/helper.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include<torch/csrc/jit/ir/subgraph_matcher.h>

namespace fastseq {
namespace jit {

using namespace torch::jit;

void FuseLayerNorm(std::shared_ptr<torch::jit::Graph>& graph) {
    // std::cout << "++++++++++++ Enter FuseLayerNorm ++++++++++++\n" << std::endl;
    Inline(*graph);
    // std::cout << ">>>>>>>>>>>> Original graph: \n\n" << graph->toString() << std::endl;
    
    std::string pattern_0 = R"(
      graph(
          %input,
          %residual,
          %bias,
          %normalized_shape,
          %normalized_weight,
          %normalized_bias,
          %normalized_eps):
        %8 = prim::Constant[value=1]()
        %9 = prim::Constant[value=1]()
        %x = aten::add(%input, %bias, %9)
        %y = aten::add(%x, %residual, %9)
        %z = aten::layer_norm(%y, %normalized_shape, %normalized_weight, %normalized_bias, %normalized_eps, %8)
        return (%z))";
    
    std::string fused_pattern_0 = R"(
      graph(
          %input,
          %residual,
          %bias,
          %normalized_shape,
          %normalized_weight,
          %normalized_bias,
          %normalized_eps):
        %res = fastseq::fused_add_layer_norm(%input, %residual, %normalized_weight, %normalized_bias, %bias)
        return (%res))";
  
    SubgraphRewriter subgraph_rewriter_0;
    subgraph_rewriter_0.RegisterRewritePattern(
        pattern_0, fused_pattern_0);
    subgraph_rewriter_0.runOnGraph(graph);

    std::string pattern_1 = R"(
        graph(%input8.1, %weight.17, %bias.17, %17, %16, %input0.1, %18, %507, %weight.1, %bias.1, %23, %37):
          %445 : Tensor = aten::linear(%input8.1, %weight.17, %bias.17)
          %x3.1 : Tensor = aten::dropout(%445, %17, %16)
          %input9.1 : Tensor = aten::add(%input0.1, %x3.1, %18)
          %x.1 : Tensor = aten::layer_norm(%input9.1, %507, %weight.1, %bias.1, %23, %37)
          return (%x.1))";
    
    std::string fused_pattern_1 = R"(
        graph(%input8.1, %weight.17, %bias.17, %17, %16, %input0.1, %18, %507, %weight.1, %bias.1, %23, %37):
          %res : Tensor = fastseq::fused_add_layer_norm(%input8.1, %input0.1,  %weight.17, %bias.17, %weight.1, %bias.1)
          return (%res))";
    
    SubgraphRewriter subgraph_rewriter_1;
    subgraph_rewriter_1.RegisterRewritePattern(
        pattern_1, fused_pattern_1);
    subgraph_rewriter_1.runOnGraph(graph);


    
    // std::cout << "<<<<<<<<<<<< Optimized graph: \n\n" << graph->toString() << std::endl;
    // std::cout << "++++++++++++ Exit FuseLayerNorm ++++++++++++\n" << std::endl;
}
} // namespace jit
} // namespace fastseq
