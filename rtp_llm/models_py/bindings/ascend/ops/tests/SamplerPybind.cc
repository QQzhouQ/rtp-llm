/**
 * Minimal pybind11 wrapper that exposes execSampleGreedy to Python.
 *
 * This module is designed to be loaded AFTER Python `import torch_npu`
 * (which registers NPU dispatch).  Because the module is a separate .so
 * loaded via Python's import system (not a DT_NEEDED pre-load), there
 * is no "Two accelerators" conflict.
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace py = pybind11;
using namespace rtp_llm;

PYBIND11_MODULE(sampler_test_module, m) {
    m.doc() = "Ascend sampler test bindings – wraps execSampleGreedy";

    m.def(
        "exec_sample_greedy",
        [](torch::Tensor logits,
           torch::Tensor input_lengths,
           torch::Tensor sequence_lengths,
           torch::Tensor token_ids,
           int64_t step,
           torch::Tensor top_k,
           torch::Tensor top_p,
           torch::Tensor temperature,
           std::optional<torch::Tensor> repetition_penalty,
           std::optional<torch::Tensor> no_repeat_ngram_size,
           std::optional<torch::Tensor> cum_log_probs,
           std::optional<torch::Tensor> output_log_probs,
           std::optional<torch::Tensor> output_all_probs,
           std::optional<torch::Tensor> presence_penalty,
           std::optional<torch::Tensor> frequency_penalty,
           std::optional<torch::Tensor> do_sample,
           std::optional<std::vector<at::Generator>> generators) -> py::dict {
            GreedyParams params{
                logits,
                input_lengths,
                sequence_lengths,
                token_ids,
                static_cast<size_t>(step),
                top_k,
                top_p,
                temperature,
                repetition_penalty,
                no_repeat_ngram_size,
                cum_log_probs,
                output_log_probs,
                output_all_probs,
                presence_penalty,
                frequency_penalty,
                do_sample,
                generators.value_or(std::vector<at::Generator>{}),
            };

            auto output = execSampleGreedy(params);

            py::dict result;
            result["token_ids"] = token_ids;  // modified in-place
            if (output.success.defined()) {
                result["success"] = output.success;
            }
            if (cum_log_probs.has_value()) {
                result["cum_log_probs"] = cum_log_probs.value();
            }
            if (output_all_probs.has_value()) {
                result["output_all_probs"] = output_all_probs.value();
            }
            return result;
        },
        py::arg("logits"),
        py::arg("input_lengths"),
        py::arg("sequence_lengths"),
        py::arg("token_ids"),
        py::arg("step"),
        py::arg("top_k"),
        py::arg("top_p"),
        py::arg("temperature"),
        py::arg("repetition_penalty")  = py::none(),
        py::arg("no_repeat_ngram_size") = py::none(),
        py::arg("cum_log_probs")       = py::none(),
        py::arg("output_log_probs")    = py::none(),
        py::arg("output_all_probs")    = py::none(),
        py::arg("presence_penalty")    = py::none(),
        py::arg("frequency_penalty")   = py::none(),
        py::arg("do_sample")           = py::none(),
        py::arg("generators")          = py::none(),
        "Execute greedy/random sampling on Ascend NPU.\n\n"
        "Returns dict with keys: token_ids, success, cum_log_probs, output_all_probs.");

    // Chain speculative sampling (MTP verify): wraps
    // execChainSpeculativeSampling with the same tensor contract as
    // SpeculativeSampler::batchSample.
    m.def(
        "chain_speculative_sampling",
        [](torch::Tensor draft_probs,
           torch::Tensor draft_token_ids,
           torch::Tensor uniform_samples,
           torch::Tensor target_probs) -> py::dict {
            const int64_t batch = draft_probs.size(0);
            const int64_t k     = draft_probs.size(1);

            auto output_token_ids = torch::zeros(
                {batch, k + 1}, torch::TensorOptions().dtype(torch::kInt32).device(draft_probs.device()));
            auto output_accepted_token_num =
                torch::zeros({batch}, torch::TensorOptions().dtype(torch::kInt32).device(draft_probs.device()));
            auto output_emitted_token_num =
                torch::zeros({batch}, torch::TensorOptions().dtype(torch::kInt32).device(draft_probs.device()));

            SpeculativeSamplingParams params(draft_probs,
                                             draft_token_ids,
                                             uniform_samples,
                                             target_probs,
                                             output_token_ids,
                                             output_accepted_token_num,
                                             output_emitted_token_num);
            execChainSpeculativeSampling(params);

            py::dict result;
            result["output_token_ids"]         = output_token_ids.cpu();
            result["output_emitted_token_num"] = output_emitted_token_num.cpu();
            result["output_accepted_token_num"] = output_accepted_token_num.cpu();
            return result;
        },
        py::arg("draft_probs"),
        py::arg("draft_token_ids"),
        py::arg("uniform_samples"),
        py::arg("target_probs"),
        "Execute chain speculative sampling on Ascend NPU.\n\n"
        "draft_probs [B,k,V] f32, draft_token_ids [B,k] i32, uniform_samples [B,k+1] f32,\n"
        "target_probs [B,k+1,V] f32. Returns dict: output_token_ids [B,k+1] i32,\n"
        "output_emitted_token_num [B] i32, output_accepted_token_num [B] i32.");
}
