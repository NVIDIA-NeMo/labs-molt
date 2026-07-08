# Third-Party Notices

This document lists the third-party open-source software contained in or used
by this repository, together with the applicable licenses.

## 1. Third-party code contained in this repository

### OpenRLHF — Apache License 2.0

Molt is a derivative work of [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF),
Copyright (c) OpenRLHF contributors, licensed under the Apache License,
Version 2.0 (see [LICENSE](LICENSE) for the full text). Files adapted from
OpenRLHF carry an attribution notice in their header.

### verl — Apache License 2.0

`molt/utils/seqlen_balancing.py` is copied from
[verl](https://github.com/volcengine/verl),
Copyright 2024 Bytedance Ltd. and/or its affiliates, licensed under the
Apache License, Version 2.0 (see [LICENSE](LICENSE) for the full text).

### SkyPilot — Apache License 2.0

`molt/utils/logging_utils.py` is adapted from
[SkyPilot](https://github.com/skypilot-org/skypilot) (`sky/sky_logging.py`),
Copyright (c) SkyPilot authors, licensed under the Apache License,
Version 2.0 (see [LICENSE](LICENSE) for the full text).

### DeepEP — MIT License

`dockerfile/deepep.patch` is a build patch against
[DeepEP](https://github.com/deepseek-ai/DeepEP), Copyright (c) 2025 DeepSeek,
licensed under the MIT License:

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### PyTorch — BSD-3-Clause License

`molt/utils/distributed_sampler.py` is adapted from
[PyTorch](https://github.com/pytorch/pytorch)
(`torch/utils/data/distributed.py`), licensed under the BSD-3-Clause license:

```
From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories
   America and IDIAP Research Institute nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

## 2. Third-party dependencies (not distributed in this repository)

The following packages are declared in `requirements.txt` / `setup.py` and are
installed separately by the user; their source code is not contained in this
repository.

| Package | License | Source |
|---|---|---|
| accelerate | Apache-2.0 | https://github.com/huggingface/accelerate |
| aiohttp | Apache-2.0 | https://github.com/aio-libs/aiohttp |
| datasets | Apache-2.0 | https://github.com/huggingface/datasets |
| dion | MIT | https://github.com/microsoft/dion |
| einops | MIT | https://github.com/arogozhnikov/einops |
| flash-attn (extra) | BSD-3-Clause | https://github.com/Dao-AILab/flash-attention |
| grpcio | Apache-2.0 | https://github.com/grpc/grpc |
| huggingface_hub | Apache-2.0 | https://github.com/huggingface/huggingface_hub |
| jsonlines | BSD-3-Clause | https://github.com/wbolster/jsonlines |
| nemo-automodel | Apache-2.0 | https://github.com/NVIDIA-NeMo/Automodel |
| optree | Apache-2.0 | https://github.com/metaopt/optree |
| packaging | Apache-2.0 OR BSD-2-Clause | https://github.com/pypa/packaging |
| peft | Apache-2.0 | https://github.com/huggingface/peft |
| pylatexenc | MIT | https://github.com/phfaist/pylatexenc |
| pynvml | BSD-3-Clause | https://github.com/gpuopenanalytics/pynvml |
| ray | Apache-2.0 | https://github.com/ray-project/ray |
| sympy | BSD-3-Clause | https://github.com/sympy/sympy |
| tensorboard | Apache-2.0 | https://github.com/tensorflow/tensorboard |
| torch | BSD-3-Clause | https://github.com/pytorch/pytorch |
| torchdata | BSD-3-Clause | https://github.com/pytorch/data |
| torchmetrics | Apache-2.0 | https://github.com/Lightning-AI/torchmetrics |
| tqdm | MPL-2.0 AND MIT | https://github.com/tqdm/tqdm |
| transformers | Apache-2.0 | https://github.com/huggingface/transformers |
| vllm | Apache-2.0 | https://github.com/vllm-project/vllm |
| vllm-router | Apache-2.0 | https://pypi.org/project/vllm-router/ |
| wandb | MIT | https://github.com/wandb/wandb |
| wheel | MIT | https://github.com/pypa/wheel |
