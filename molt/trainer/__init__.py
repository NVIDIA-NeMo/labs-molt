# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training orchestration, algorithms, rollout generation, and worker utilities."""

# Keep this package lightweight so vLLM/Ray workers do not import training backends implicitly.
