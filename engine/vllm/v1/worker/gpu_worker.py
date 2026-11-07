# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A GPU worker class."""

import copy
import gc
import os
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
    set_custom_all_reduce,
)
from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.model_executor.warmup.kernel_warmup import kernel_warmup
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.tasks import SupportedTask
from vllm.utils import GiB_bytes, MemorySnapshot, memory_profiling
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    DraftTokenIds,
    ModelRunnerOutput,
)
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.utils import is_residual_scattered_for_sp
from vllm.v1.worker.worker_base import WorkerBase

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.core.sched.output import SchedulerOutput


class Worker(WorkerBase):
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules

            init_cached_hf_modules()

        # Buffers saved before sleep
        self._sleep_saved_buffers: dict[str, torch.Tensor] = {}

        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        if envs.VLLM_TORCH_PROFILER_DIR:
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            worker_name = f"{vllm_config.instance_id}-rank-{self.rank}"
            logger.info(
                "Profiling enabled. Traces will be saved to: %s",
                torch_profiler_trace_dir,
            )
            logger.debug(
                "Profiler config: record_shapes=%s,"
                "profile_memory=%s,with_stack=%s,with_flops=%s",
                envs.VLLM_TORCH_PROFILER_RECORD_SHAPES,
                envs.VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY,
                envs.VLLM_TORCH_PROFILER_WITH_STACK,
                envs.VLLM_TORCH_PROFILER_WITH_FLOPS,
            )
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=envs.VLLM_TORCH_PROFILER_RECORD_SHAPES,
                profile_memory=envs.VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY,
                with_stack=envs.VLLM_TORCH_PROFILER_WITH_STACK,
                with_flops=envs.VLLM_TORCH_PROFILER_WITH_FLOPS,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir, worker_name=worker_name, use_gzip=True
                ),
            )
        else:
            self.profiler = None

    def sleep(self, level: int = 1) -> None:
        from vllm.device_allocator.cumem import CuMemAllocator

        free_bytes_before_sleep = torch.cuda.mem_get_info()[0]

        # Save the buffers before level 2 sleep
        if level == 2:
            model = self.model_runner.model
            self._sleep_saved_buffers = {
                name: buffer.cpu().clone() for name, buffer in model.named_buffers()
            }

        allocator = CuMemAllocator.get_instance()
        allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())
        free_bytes_after_sleep, total = torch.cuda.mem_get_info()
        freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        used_bytes = total - free_bytes_after_sleep
        assert freed_bytes >= 0, "Memory usage increased after sleeping."
        logger.info(
            "Sleep mode freed %.2f GiB memory, %.2f GiB memory is still in use.",
            freed_bytes / GiB_bytes,
            used_bytes / GiB_bytes,
        )

    def wake_up(self, tags: list[str] | None = None) -> None:
        from vllm.device_allocator.cumem import CuMemAllocator

        allocator = CuMemAllocator.get_instance()
        allocator.wake_up(tags)

        # Restore the buffers after level 2 sleep
        if len(self._sleep_saved_buffers):
            model = self.model_runner.model
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        if self.vllm_config.model_config.enable_sleep_mode:
            from vllm.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()
            if tag == "weights":
                assert allocator.get_current_usage() == 0, (
                    "Sleep mode can only be used for one instance per process."
                )
            context = allocator.use_memory_pool(tag=tag)
        else:
            context = nullcontext()
        return context

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def init_device(self):
        if self.device_config.device.type == "cuda":
            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            current_platform.set_device(self.device)

            current_platform.check_if_supports_dtype(self.model_config.dtype)

            # Initialize the distributed environment BEFORE taking
            # memory snapshot
            # This ensures NCCL buffers are allocated before we measure
            # available memory
            init_worker_distributed_environment(
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
                current_platform.dist_backend,
            )

            # Set random seed.
            set_random_seed(self.model_config.seed)

            # Now take memory snapshot after NCCL is initialized
            gc.collect()
            torch.cuda.empty_cache()

            # take current memory snapshot
            self.init_snapshot = MemorySnapshot()
            self.requested_memory = (
                self.init_snapshot.total_memory
                * self.cache_config.gpu_memory_utilization
            )
            if self.init_snapshot.free_memory < self.requested_memory:
                GiB = lambda b: round(b / GiB_bytes, 2)
                raise ValueError(
                    f"Free memory on device "
                    f"({GiB(self.init_snapshot.free_memory)}/"
                    f"{GiB(self.init_snapshot.total_memory)} GiB) on startup "
                    f"is less than desired GPU memory utilization "
                    f"({self.cache_config.gpu_memory_utilization}, "
                    f"{GiB(self.requested_memory)} GiB). Decrease GPU memory "
                    f"utilization or reduce GPU memory used by other processes."
                )
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        # Construct the model runner
        self.model_runner: GPUModelRunner = GPUModelRunner(
            self.vllm_config, self.device
        )

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)

    # FIXME(youkaichao & ywang96): Use TorchDispatchMode instead of memory pool
    # to hijack tensor allocation.
    def load_model(self) -> None:
        eep_scale_up = os.environ.get("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH") == "1"
        with self._maybe_get_memory_pool_context(tag="weights"):
            self.model_runner.load_model(eep_scale_up=eep_scale_up)

    def update_config(self, overrides: dict[str, Any]) -> None:
        self.model_runner.update_config(overrides)

    def reload_weights(self) -> None:
        self.model_runner.reload_weights()

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Profiles the peak memory usage of the model to determine how much
        memory can be used for KV cache without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculates the free memory that can be used for KV cache in
        bytes.

        Tip:
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        GiB = lambda b: b / GiB_bytes
        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
            # still need a profile run which compiles the model for
            # max_num_batched_tokens
            self.model_runner.profile_run()

            msg = (
                f"Initial free memory {GiB(self.init_snapshot.free_memory):.2f} "
                f"GiB, reserved {GiB(kv_cache_memory_bytes):.2f} GiB memory for "
                "KV Cache as specified by kv_cache_memory_bytes config and "
                "skipped memory profiling. This does not respect the "
                "gpu_memory_utilization config. Only use kv_cache_memory_bytes "
                "config when you want manual control of KV cache memory "
                "size. If OOM'ed, check the difference of initial free "
                "memory between the current run and the previous run "
                "where kv_cache_memory_bytes is suggested and update it "
                "correspondingly."
            )
            logger.info(msg)
            return kv_cache_memory_bytes

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        with memory_profiling(
            self.init_snapshot,
            weights_memory=int(self.model_runner.model_memory_usage),
        ) as profile_result:
            self.model_runner.profile_run()

        self.non_torch_memory = profile_result.non_torch_increase
        self.peak_activation_memory = profile_result.torch_peak_increase

        free_gpu_memory = profile_result.after_profile.free_memory
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        assert self.init_snapshot.free_memory > free_gpu_memory, (
            "Error in memory profiling. "
            f"Initial free memory {GiB(self.init_snapshot.free_memory)} GiB, "
            f"current free memory {GiB(free_gpu_memory)} GiB. "
            "This happens when other processes sharing the same container "
            "release GPU memory while vLLM is profiling during initialization. "
            "To fix this, ensure consistent GPU memory allocation or "
            "isolate vLLM in its own container."
        )
        self.available_kv_cache_memory_bytes = (
            self.requested_memory - profile_result.non_kv_cache_memory
        )

        unrequested_memory = self.init_snapshot.free_memory - self.requested_memory
        logger.debug(
            "Initial free memory: %.2f GiB; Requested memory: %.2f (util), %.2f GiB",
            GiB(self.init_snapshot.free_memory),
            self.cache_config.gpu_memory_utilization,
            GiB(self.requested_memory),
        )
        logger.debug(
            "Free memory after profiling: %.2f GiB (total), "
            "%.2f GiB (within requested)",
            GiB(free_gpu_memory),
            GiB(free_gpu_memory - unrequested_memory),
        )
        logger.debug(profile_result)
        logger.info(
            "Available KV cache memory: %.2f GiB",
            GiB(self.available_kv_cache_memory_bytes),
        )
        gc.collect()

        return int(self.available_kv_cache_memory_bytes)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""

        if self.vllm_config.model_config.enable_sleep_mode:
            from vllm.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()
            context = allocator.use_memory_pool(tag="kv_cache")
        else:
            context = nullcontext()
        with context:
            self.model_runner.initialize_kv_cache(kv_cache_config)

    def compile_or_warm_up_model(self) -> None:
        # warm up sizes that are not in cudagraph capture sizes,
        # but users still want to compile for better performance,
        # e.g. for the max-num-batched token size in chunked prefill.
        warmup_sizes = self.vllm_config.compilation_config.compile_sizes.copy()
        if not self.model_config.enforce_eager:
            warmup_sizes = [
                x
                for x in warmup_sizes
                if x not in self.vllm_config.compilation_config.cudagraph_capture_sizes
            ]
        # We skip EPLB here since we don't want to record dummy metrics
        for size in sorted(warmup_sizes, reverse=True):
            logger.info("Compile and warming up model for size %d", size)
            self.model_runner._dummy_run(size, skip_eplb=True, remove_lora=False)
        self.model_runner.maybe_remove_all_loras(self.model_runner.lora_config)

        # Warmup and tune the kernels used during model execution before
        # cuda graph capture.
        kernel_warmup(self)

        cuda_graph_memory_bytes = 0
        if not self.model_config.enforce_eager:
            cuda_graph_memory_bytes = self.model_runner.capture_model()

        if self.cache_config.kv_cache_memory_bytes is None and hasattr(
            self, "peak_activation_memory"
        ):
            # Suggests optimal kv cache memory size if we rely on
            # memory_profiling to guess the kv cache memory size which
            # provides peak_activation_memory and a few other memory
            # consumption. `memory_profiling` does not consider
            # CUDAGraph memory size and may not utilize all gpu memory.
            # Users may want fine-grained control to specify kv cache
            # memory size.
            GiB = lambda b: round(b / GiB_bytes, 2)

            # empirically observed that the memory profiling may
            # slightly underestimate the memory consumption.
            # So leave a small buffer (=150MiB) to avoid OOM.
            redundancy_buffer_memory = 150 * (1 << 20)
            non_kv_cache_memory = (
                self.model_runner.model_memory_usage
                + self.peak_activation_memory
                + self.non_torch_memory
                + cuda_graph_memory_bytes
            )
            kv_cache_memory_bytes_to_gpu_limit = (
                self.init_snapshot.free_memory
                - non_kv_cache_memory
                - redundancy_buffer_memory
            )
            kv_cache_memory_bytes_to_requested_limit = (
                int(self.requested_memory)
                - non_kv_cache_memory
                - redundancy_buffer_memory
            )

            msg = (
                f"Free memory on device "
                f"({GiB(self.init_snapshot.free_memory)}/"
                f"{GiB(self.init_snapshot.total_memory)} GiB) on startup. "
                f"Desired GPU memory utilization is "
                f"({self.cache_config.gpu_memory_utilization}, "
                f"{GiB(self.requested_memory)} GiB). "
                f"Actual usage is {GiB(self.model_runner.model_memory_usage)} "
                f"GiB for weight, {GiB(self.peak_activation_memory)} GiB "
                f"for peak activation, {GiB(self.non_torch_memory)} GiB "
                f"for non-torch memory, and {GiB(cuda_graph_memory_bytes)} "
                f"GiB for CUDAGraph memory. Replace gpu_memory_utilization "
                f"config with `--kv-cache-memory="
                f"{kv_cache_memory_bytes_to_requested_limit}` "
                f"({GiB(kv_cache_memory_bytes_to_requested_limit)} GiB) to fit "
                f"into requested memory, or `--kv-cache-memory="
                f"{kv_cache_memory_bytes_to_gpu_limit}` "
                f"({GiB(kv_cache_memory_bytes_to_gpu_limit)} GiB) to fully "
                f"utilize gpu memory. Current kv cache memory in use is "
                f"{GiB(self.available_kv_cache_memory_bytes)} GiB."
            )

            logger.debug(msg)

        # Warm up sampler and preallocate memory buffer for logits and other
        # sampling related tensors of max possible shape to avoid memory
        # fragmentation issue.
        # NOTE: This is called after `capture_model` on purpose to prevent
        # memory buffers from being cleared by `torch.cuda.empty_cache`.
        if get_pp_group().is_last_rank:
            max_num_reqs = min(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens,
            )

            # We skip EPLB here since we don't want to record dummy metrics
            hidden_states, last_hidden_states = self.model_runner._dummy_run(
                num_tokens=max_num_reqs,
                skip_eplb=True,
            )
            if self.model_runner.is_pooling_model:
                self.model_runner._dummy_pooler_run(hidden_states)
            else:
                self.model_runner._dummy_sampler_run(hidden_states=last_hidden_states)

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    def reset_mm_cache(self) -> None:
        self.model_runner.reset_mm_cache()

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        intermediate_tensors = None
        forward_pass = scheduler_output.total_num_scheduled_tokens > 0
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        num_input_tokens = self.model_runner._get_num_input_tokens(num_scheduled_tokens)
        all_gather_tensors = {
            "residual": not is_residual_scattered_for_sp(
                self.vllm_config, num_input_tokens
            )
        }
        if forward_pass and not get_pp_group().is_first_rank:
            intermediate_tensors = IntermediateTensors(
                get_pp_group().recv_tensor_dict(
                    all_gather_group=get_tp_group(),
                    all_gather_tensors=all_gather_tensors,
                )
            )

        output = self.model_runner.execute_model(scheduler_output, intermediate_tensors)
        if isinstance(output, (ModelRunnerOutput, AsyncModelRunnerOutput)):
            return output

        assert isinstance(output, IntermediateTensors)
        parallel_config = self.vllm_config.parallel_config
        assert (
            parallel_config.distributed_executor_backend != ("external_launcher")
            and not get_pp_group().is_last_rank
        )

        get_pp_group().send_tensor_dict(
            output.tensors,
            all_gather_group=get_tp_group(),
            all_gather_tensors=all_gather_tensors,
        )

        kv_connector_output = output.kv_connector_output
        if not kv_connector_output:
            return None

        # In case of PP with kv transfer, we need to pass through the
        # kv_connector_output
        if kv_connector_output.is_empty():
            return EMPTY_MODEL_RUNNER_OUTPUT

        output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.kv_connector_output = kv_connector_output
        return output

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.model_runner.take_draft_token_ids()

    def set_ngram_backoff(self, miss_threshold: int, probe_interval: int) -> None:
        drafter = self.model_runner.drafter
        if not hasattr(drafter, "configure_backoff"):
            raise RuntimeError("the active proposer does not support N-gram backoff")
        drafter.configure_backoff(miss_threshold, probe_interval)

    def set_ngram_enabled(self, enabled: bool) -> None:
        drafter = self.model_runner.drafter
        if not hasattr(drafter, "configure_enabled"):
            raise RuntimeError("the active proposer does not support N-gram gating")
        drafter.configure_enabled(enabled)

    def set_hybrid_retrieval_only(self, enabled: bool) -> None:
        runner = self.model_runner
        if not hasattr(runner, "hybrid_retrieval_only"):
            raise RuntimeError("the active proposer is not the EAGLE hybrid")
        runner.hybrid_retrieval_only = bool(enabled)
        # This control is used only between completed workloads. Resetting the
        # per-request state makes each side of an AB/BA pair independent even
        # if a frontend later reuses request identifiers.
        runner.hybrid_eagle_debt_req_ids.clear()
        runner.hybrid_eagle_initialized_req_ids.clear()
        runner.hybrid_draft_debt_tokens_by_req.clear()
        runner.hybrid_draft_debt_steps = 0
        runner.hybrid_draft_synced_req_ids = frozenset()

    def set_hybrid_ngram_window(self, min_n: int, max_n: int) -> None:
        drafter = getattr(self.model_runner, "ngram_drafter", None)
        if drafter is None or not hasattr(drafter, "configure_match_window"):
            raise RuntimeError("the active proposer is not the N-gram hybrid")
        drafter.configure_match_window(min_n, max_n)

    def set_hybrid_ngram_enabled(self, enabled: bool) -> None:
        """Enable or disable retrieval proposals between drained workloads."""
        drafter = getattr(self.model_runner, "ngram_drafter", None)
        if drafter is None or not hasattr(drafter, "configure_enabled"):
            raise RuntimeError("the active proposer is not the N-gram hybrid")
        if not isinstance(enabled, bool):
            raise ValueError("hybrid N-gram enabled flag must be bool")
        drafter.configure_enabled(enabled)

    def set_hybrid_max_draft_skip_steps(self, value: int) -> None:
        """Switch the 0/1 bounded-debt ablation between completed workloads."""
        if value not in (0, 1):
            raise ValueError(f"hybrid draft skip steps must be 0 or 1, got {value}")
        runner = self.model_runner
        if not hasattr(runner, "hybrid_draft_debt_steps"):
            raise RuntimeError("the active proposer is not a stateful hybrid")
        runner.speculative_config.hybrid_max_draft_skip_steps = value
        runner.hybrid_eagle_debt_req_ids.clear()
        runner.hybrid_eagle_initialized_req_ids.clear()
        runner.hybrid_draft_debt_tokens_by_req.clear()
        runner.hybrid_draft_debt_steps = 0
        runner.hybrid_draft_synced_req_ids = frozenset()

    def set_hybrid_replay_bounds(self, rounds: int, tokens: int | None) -> None:
        """Switch explicit replay bounds between completed generate workloads."""
        runner = self.model_runner
        if not runner.speculative_config.hybrid_state_replay:
            raise RuntimeError("explicit state replay is not enabled")
        if rounds < -1 or (tokens is not None and tokens < 0):
            raise ValueError("invalid replay bounds")
        runner.speculative_config.hybrid_max_draft_skip_steps = rounds
        runner.speculative_config.hybrid_max_draft_debt_tokens = tokens
        runner.hybrid_replay_by_req.clear()
        runner.hybrid_replay_rebuild_until.clear()
        runner.hybrid_draft_debt_tokens_by_req.clear()
        runner.hybrid_draft_debt_steps = 0
        runner.hybrid_draft_synced_req_ids = frozenset()

    def set_hybrid_replay_kv_only_hits(self, enabled: bool) -> None:
        """Switch candidate omission only between drained replay workloads."""
        runner = self.model_runner
        if not runner.speculative_config.hybrid_state_replay:
            raise RuntimeError("explicit state replay is not enabled")
        if runner.hybrid_replay_by_req:
            raise RuntimeError("switch KV-only replay only after requests drain")
        if not isinstance(enabled, bool):
            raise ValueError("KV-only replay switch must be bool")
        runner.speculative_config.hybrid_replay_kv_only_hits = enabled

    def set_hybrid_replay_serial_chunks(self, enabled: bool) -> None:
        """Switch source-round recovery only after explicit replay requests drain."""
        runner = self.model_runner
        if not runner.speculative_config.hybrid_state_replay:
            raise RuntimeError("explicit state replay is not enabled")
        if runner.hybrid_replay_by_req:
            raise RuntimeError("switch serial replay only after requests drain")
        if not isinstance(enabled, bool):
            raise ValueError("serial replay switch must be bool")
        runner.speculative_config.hybrid_replay_serial_chunks = enabled

    def get_hybrid_replay_state(self) -> dict:
        """Read-only state inspection for the experimental correctness harness."""
        runner = self.model_runner
        states = getattr(runner, "hybrid_replay_by_req", {})
        return {req_id: {
            "consumed_end": state.consumed_end,
            "rebuild_until": state.rebuild_until,
            "rounds": state.rounds,
            "debt": state.debt,
            "pending_rows": sum(c[0].numel() for c in state.chunks),
        } for req_id, state in states.items()}

    def set_hybrid_runtime_budgets(self, retrieval_k: int, eagle_k: int) -> None:
        """Switch hybrid proposal lengths within the allocated maximum.

        This control is intended for paired ablations between completed
        requests. The engine is constructed at the largest tested K so target
        verification buffers keep a fixed capacity; only the actual N-gram and
        EAGLE proposal lengths are changed.
        """
        runner = self.model_runner
        ngram_drafter = getattr(runner, "ngram_drafter", None)
        eagle_drafter = getattr(runner, "drafter", None)
        if ngram_drafter is None or eagle_drafter is None:
            raise RuntimeError("the active proposer is not the EAGLE hybrid")
        max_retrieval_k = int(ngram_drafter.valid_ngram_draft.shape[1])
        max_eagle_k = int(runner.speculative_config.num_speculative_tokens)
        if not 1 <= retrieval_k <= max_retrieval_k:
            raise ValueError(
                "retrieval budget must satisfy "
                f"1 <= retrieval_k <= {max_retrieval_k}"
            )
        if not 1 <= eagle_k <= max_eagle_k:
            raise ValueError(
                f"eagle budget must satisfy 1 <= eagle_k <= {max_eagle_k}"
            )
        ngram_drafter.k = int(retrieval_k)
        eagle_drafter.num_speculative_tokens = int(eagle_k)
        runner.speculative_config.hybrid_eagle_num_speculative_tokens = int(eagle_k)
        runner.hybrid_eagle_debt_req_ids.clear()
        runner.hybrid_eagle_initialized_req_ids.clear()
        runner.hybrid_draft_debt_tokens_by_req.clear()
        runner.hybrid_draft_debt_steps = 0
        runner.hybrid_draft_synced_req_ids = frozenset()

    def profile(self, is_start: bool = True):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()
            # only print profiler results on rank 0
            if self.local_rank == 0:
                print(
                    self.profiler.key_averages().table(sort_by="self_cuda_time_total")
                )

    def execute_dummy_batch(self) -> None:
        self.model_runner._dummy_run(1, uniform_decode=True)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_runner.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def check_health(self) -> None:
        # worker will always be healthy as long as it's running.
        return

    def _eplb_before_scale_down(self, old_ep_size: int, new_ep_size: int) -> None:
        from vllm.distributed.parallel_state import get_ep_group

        if get_ep_group().rank == 0:
            logger.info(
                "[Elastic EP] Starting expert resharding before scaling down..."
            )
        rank_mapping = {
            old_ep_rank: old_ep_rank if old_ep_rank < new_ep_size else -1
            for old_ep_rank in range(old_ep_size)
        }
        assert self.model_runner.eplb_state is not None
        self.model_runner.eplb_state.rearrange(
            self.model_runner.model,
            execute_shuffle=True,
            global_expert_load=None,
            rank_mapping=rank_mapping,
        )
        torch.cuda.synchronize()
        if get_ep_group().rank == 0:
            logger.info("[Elastic EP] Expert resharding completed!")

    def _eplb_after_scale_up(
        self,
        old_ep_size: int,
        new_ep_size: int,
        global_expert_load: torch.Tensor | None,
    ) -> None:
        from vllm.distributed.parallel_state import get_ep_group

        if get_ep_group().rank == 0:
            logger.info("[Elastic EP] Starting expert resharding after scaling up...")
        rank_mapping = {old_ep_rank: old_ep_rank for old_ep_rank in range(old_ep_size)}
        assert self.model_runner.eplb_state is not None
        self.model_runner.eplb_state.rearrange(
            self.model_runner.model,
            execute_shuffle=True,
            global_expert_load=global_expert_load,
            rank_mapping=rank_mapping,
        )
        if get_ep_group().rank == 0:
            logger.info("[Elastic EP] Expert resharding completed!")

    def _reconfigure_parallel_config(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        """
        Update parallel config with provided reconfig_request
        """
        parallel_config = self.vllm_config.parallel_config
        parallel_config.data_parallel_size = reconfig_request.new_data_parallel_size
        if (
            reconfig_request.new_data_parallel_rank
            != ReconfigureRankType.KEEP_CURRENT_RANK
        ):
            parallel_config.data_parallel_rank = reconfig_request.new_data_parallel_rank
        if (
            reconfig_request.new_data_parallel_rank_local
            != ReconfigureRankType.KEEP_CURRENT_RANK
        ):
            parallel_config.data_parallel_rank_local = (
                reconfig_request.new_data_parallel_rank_local
            )
        parallel_config.data_parallel_master_ip = (
            reconfig_request.new_data_parallel_master_ip
        )
        parallel_config.data_parallel_master_port = (
            reconfig_request.new_data_parallel_master_port
        )

    def _reconfigure_moe(
        self, old_ep_size: int, new_ep_size: int
    ) -> torch.Tensor | None:
        """
        Reconfigure MoE modules with provided reconfig_request

        Return the global expert load if new_ep_size > old_ep_size,
        otherwise None
        """
        from vllm.distributed.parallel_state import (
            get_dp_group,
            get_ep_group,
            prepare_communication_buffer_for_model,
        )
        from vllm.model_executor.layers.fused_moe.layer import FusedMoEParallelConfig

        parallel_config = self.vllm_config.parallel_config
        moe_modules = [
            module
            for module in self.model_runner.model.modules()
            if (
                module.__class__.__name__ == "FusedMoE"
                or module.__class__.__name__ == "SharedFusedMoE"
            )
        ]
        num_local_experts = moe_modules[0].moe_config.num_local_experts
        assert all(
            module.moe_config.num_local_experts == num_local_experts
            for module in moe_modules
        ), "All MoE modules must have the same number of experts"
        for module in moe_modules:
            module.moe_config.num_experts = num_local_experts * new_ep_size
            module.global_num_experts = module.moe_config.num_experts
            module.moe_parallel_config = FusedMoEParallelConfig.make(
                tp_size_=get_tp_group().world_size,
                dp_size_=get_dp_group().world_size,
                vllm_parallel_config=parallel_config,
            )
            module.moe_config.moe_parallel_config = module.moe_parallel_config
        if new_ep_size < old_ep_size:
            num_local_physical_experts = num_local_experts
            assert self.model_runner.eplb_state is not None
            new_physical_experts = (
                self.model_runner.eplb_state.physical_to_logical_map.shape[1]
            )
            parallel_config.eplb_config.num_redundant_experts = (
                new_physical_experts
                - self.model_runner.eplb_state.logical_replica_count.shape[1]
            )
            global_expert_load = None
        else:
            num_local_physical_experts = torch.tensor(
                [num_local_experts], dtype=torch.int32, device="cpu"
            )
            torch.distributed.broadcast(
                num_local_physical_experts, group=get_ep_group().cpu_group, group_src=0
            )
            num_local_physical_experts = num_local_physical_experts.item()
            new_physical_experts = num_local_physical_experts * new_ep_size
            assert self.model_runner.eplb_state is not None
            global_expert_load = self.model_runner.eplb_state.rearrange(
                self.model_runner.model, execute_shuffle=False
            )
            parallel_config.eplb_config.num_redundant_experts = (
                new_physical_experts - global_expert_load.shape[1]
            )
        prepare_communication_buffer_for_model(self.model_runner.model)
        self.model_runner.model.update_physical_experts_metadata(
            num_physical_experts=new_physical_experts,
            num_local_physical_experts=num_local_physical_experts,
        )
        return global_expert_load

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        from vllm.config import set_current_vllm_config
        from vllm.distributed.parallel_state import (
            cleanup_dist_env_and_memory,
            get_ep_group,
        )

        old_ep_size = get_ep_group().world_size
        old_ep_rank = get_ep_group().rank
        new_ep_size = (
            reconfig_request.new_data_parallel_size
            * get_tp_group().world_size
            * get_pp_group().world_size
        )
        if new_ep_size < old_ep_size:
            self._eplb_before_scale_down(old_ep_size, new_ep_size)

        cleanup_dist_env_and_memory()

        if (
            reconfig_request.new_data_parallel_rank
            == ReconfigureRankType.SHUTDOWN_CURRENT_RANK
        ):
            assert old_ep_rank >= new_ep_size
            # shutdown
            return

        self._reconfigure_parallel_config(reconfig_request)

        with set_current_vllm_config(self.vllm_config):
            init_worker_distributed_environment(
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
            )

        global_expert_load = self._reconfigure_moe(old_ep_size, new_ep_size)

        if new_ep_size > old_ep_size:
            assert global_expert_load is not None
            self._eplb_after_scale_up(old_ep_size, new_ep_size, global_expert_load)

    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        from vllm.model_executor.model_loader import ShardedStateLoader

        ShardedStateLoader.save_model(
            self.model_runner.model,
            path,
            pattern=pattern,
            max_size=max_size,
        )

    def save_tensorized_model(
        self,
        tensorizer_config: "TensorizerConfig",
    ) -> None:
        self.model_runner.save_tensorized_model(
            tensorizer_config=tensorizer_config,
        )

    def shutdown(self) -> None:
        if runner := getattr(self, "model_runner", None):
            runner.ensure_kv_transfer_shutdown()


def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: str | None = None,
    local_rank: int = -1,
    backend: str = "nccl",
) -> None:
    """Initialize the distributed environment."""
    parallel_config = vllm_config.parallel_config
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_distributed_environment(
        parallel_config.world_size, rank, distributed_init_method, local_rank, backend
    )

    ensure_model_parallel_initialized(
        parallel_config.tensor_parallel_size,
        parallel_config.pipeline_parallel_size,
        parallel_config.decode_context_parallel_size,
    )

    ensure_kv_transfer_initialized(vllm_config)
