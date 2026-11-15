// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// News items and blog posts, newest first. The homepage lists the first eight; blog/ shows all.
// How to add one, and what each field means: README.md in this folder.
window.MOLT_POSTS = [
  {
    title: "Molt powers REINFORCE Zero",
    date: "2026-11",
    tag: "Research",
    summary: "Trainer–inference invariance (TII) meets unclipped token-level importance sampling for stable asynchronous RL. Kernel alignment ships as the train-inference-consistency agent skill.",
    url: "https://hijkzzz.github.io/REINFORCE-Zero/",
    tone: "green",
    cover: ["20,000", "policy updates, stable and asynchronous"]
  },
  {
    title: "FlashREINFORCE lands in Molt",
    date: "2026-09",
    tag: "Release",
    summary: "Critic-free single-rollout RL with stable training beyond 6,000 steps, with a quick-start script in the repo.",
    url: "https://www.alphaxiv.org/abs/2609.flashreinforce-asynchronous-rl-agentic-models",
    tone: "violet",
    cover: ["6,000+", "steps of critic-free, single-rollout RL"]
  }
];
