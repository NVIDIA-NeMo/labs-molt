// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The docs library, in reading order: [file name in pages/ without ".md", title, one-line summary].
// How to add a page, and which pages are generated: README.md in this folder.
window.MOLT_DOCS = [
  { section: "Start here", pages: [
    ["installation", "Installation", "The container, a local install, and the optional AutoModel-Slim backend."],
    ["quick-start", "Quick start", "A first RL run, a first SFT run, and the switches to reach for next."],
    ["recipes", "Recipes", "The launch scripts that ship with the repository."]
  ] },
  { section: "Build agents", pages: [
    ["agents", "Agent contract", "Env or ChatAgent, the Result fields, retries, and context compaction."]
  ] },
  { section: "Train", pages: [
    ["algorithms", "Algorithms", "Estimators, KL, the PPO critic, distillation, and adding your own."],
    ["async-training", "Asynchronous training", "The async switches, partial rollout, and the mismatch metrics."],
    ["is-correction", "IS correction", "Correcting stale rollout log-probs, score centering, and the REINFORCE Zero recipe."],
    ["evaluate-debug", "Evaluate and debug", "Scoring a checkpoint, replaying a rollout batch, and metrics to watch."]
  ] },
  { section: "Scale", pages: [
    ["scaling", "Scaling", "TP, EP, and CP for the trainer and for the rollout engines."],
    ["moe-routing", "MoE routing", "Router precision, routing replay, and router freeze."],
    ["mtp-rollout", "MTP rollout", "Faster generation with speculative decoding."],
    ["lora", "LoRA", "Flags, target matching, learning rates, and the RL limits."]
  ] },
  { section: "Reference", pages: [
    ["cli-rl", "RL flags", "Every flag of molt.cli.train_rl_ray, generated from its parser."],
    ["cli-sft", "SFT flags", "Every flag of molt.cli.train_sft, generated from its parser."],
    ["api-agents", "molt.agents API", "The exported classes with their fields, methods, and docstrings."]
  ] },
  { section: "Project", pages: [
    ["development", "Development", "Checking a change, regenerating the reference, and contributing."]
  ] }
];
