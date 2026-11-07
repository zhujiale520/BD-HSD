# 简历项目描述

以下内容可以直接放入中文或英文简历。数字与 [`results/BD-HSD-results.md`](../results/BD-HSD-results.md) 和 [`results/summary.json`](../results/summary.json) 对应。

## 中文版

**BD-HSD：预算解耦混合投机解码引擎**｜Python、PyTorch、CUDA、vLLM、Qwen3-8B、EAGLE-3、N-gram、RTX 4090/A100

- 设计并实现 vLLM 推理运行时，将 N-gram 检索与 EAGLE-3 神经草稿组成混合候选路径，由 target model 批量验证并统一处理 token 接受、拒绝尾部和 EOS。
- 解耦 retrieval 与 neural speculation 预算，在 RTX 4090 上对 Qwen3-8B/Llama-3.1-8B 的 128 请求、batch 1/4、3 次独立重复进行配对实验；相对选定共享预算，端到端延迟降低 **13.05%–28.09%**，95% bootstrap CI 均为正。
- 在 CanItEdit 四配置消融中，`K10/E10 → K10/E3` 的预算解耦额外降低 **21.52%**，加入一步有界状态债务再降低 **4.43%**；6/6 配对正向，所有配置通过 40/40 逐 token 正确性门控。
- 实现 EAGLE hidden-state debt、KV Cache 生命周期和 mixed-batch selective draft sub-batch；组件 trace 显示有界债务减少 **32.63%** 的 EAGLE 执行、减少 **32.02%** 的 EAGLE 时间，并保留 target verification 正确性。

## English version

**BD-HSD: Budget-Decoupled Hybrid Speculative Decoding** | Python, PyTorch, CUDA, vLLM, Qwen3-8B, EAGLE-3, N-gram, RTX 4090/A100

- Designed and implemented a vLLM inference runtime that routes N-gram retrieval hits and EAGLE-3 neural fallback candidates into target-model batched verification with unified accept/reject/EOS handling.
- Decoupled retrieval and neural speculation budgets and ran paired experiments on Qwen3-8B and Llama-3.1-8B with 128 requests, batch sizes 1/4, and three independent repeats; latency decreased **13.05%–28.09%** versus selected shared budgets, with positive 95% bootstrap intervals.
- Isolated the mechanism in a four-way CanItEdit ablation: `K10/E10 → K10/E3` reduced latency by **21.52%**, while one-step bounded state debt added **4.43%**; all six pairs were positive and every configuration passed the 40/40 token-level correctness gate.
- Implemented EAGLE hidden-state debt accounting, KV-cache lifecycle management, and selective draft sub-batching for mixed batches; component tracing reduced EAGLE executions by **32.63%** and EAGLE time by **32.02%**.

## 一句话项目介绍

构建一个把便宜的 N-gram 检索和 EAGLE-3 神经草稿按独立预算协同调度的 vLLM 推理运行时，并通过 token-level correctness gate、KV Cache 状态管理和同引擎配对实验验证收益。

## 面试时的结果口径

推荐优先讲主实验的四个格子和 CanItEdit 的四路消融。A100 数据是独立 32 请求复验，不与 RTX 4090 主实验合并；`Ordinary Target` 只作为工程参考，不用于严格 same-output 速度结论。
