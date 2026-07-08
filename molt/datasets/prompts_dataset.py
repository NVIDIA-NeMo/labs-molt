# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from OpenRLHF (https://github.com/OpenRLHF/OpenRLHF),
# Copyright (c) OpenRLHF contributors, licensed under the Apache License, Version 2.0.

from torch.utils.data import Dataset

from molt.utils.vlm_utils import should_expand_image_placeholder, split_image_placeholder


def preprocess_data(
    data,
    input_key="input",
    label_key=None,
    apply_chat_template=None,
    prerender=True,
    expand_image_placeholder: bool = False,
    tools=None,
):
    # Verifier ground-truth answer for RL reward computation (empty if no label_key).
    label = "" if label_key is None else data[label_key]
    prompt = data[input_key]

    if not apply_chat_template:
        # Completion-style data: a plain pre-rendered string, fed verbatim (step runner only —
        # chat agents require --data.apply_chat_template). A messages list here means chat data
        # without the flag: refuse loudly instead of training on untemplated text.
        if isinstance(prompt, list):
            raise ValueError("Dataset rows are chat messages: pass --data.apply_chat_template.")
        return prompt, label

    chat = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    if not prerender:
        # Chat-agent path: hand the messages through untouched. The chat server renders them
        # exactly once with the model's own template (ChatAgentRunner attaches this row's
        # images/tools at the OpenAI wire), so step and chat runners consume the SAME dataset.
        return chat, label

    if expand_image_placeholder:
        # Qwen2.5-VL / Qwen3.6-VL chat templates iterate over structured
        # content lists `[{"type": "image"}, {"type": "text", "text": ...}]`
        # and emit a model-specific placeholder (e.g. `<|image_pad|>`) that
        # vLLM's multimodal processor recognizes for prompt replacement.
        # Datasets store user content as a single string `"<image>\nproblem"`,
        # so split on each `<image>` and convert to the structured form.
        chat = [split_image_placeholder(msg) for msg in chat]
    # Otherwise: pass content through verbatim. Chat templates that emit
    # `message.content | string` won't iterate structured content; the
    # literal `<image>` in the text is what the processor matches on.
    # `tools` (OpenAI function-call schemas) are rendered by Qwen / Llama
    # chat templates into a system-side preamble that teaches the model
    # the `<tool_call>{...}</tool_call>` emission format natively.
    kwargs = {"tools": tools} if tools else {}
    prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True, **kwargs)
    return prompt, label


class PromptDataset(Dataset):
    """
    Dataset for policy RL prompts.

    Args:
        dataset: prompt dataset
        tokenizer: tokenizer for the actor model
        strategy: training strategy; supplies the ``args.data`` column keys
        prerender: render the chat template here (step runner) or hand the raw
            messages through for the chat server to render (chat runner);
            derived from ``Runner.PRERENDER_PROMPT``
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        prerender=True,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.prerender = prerender

        # LAZY rendering: keep the (memory-mapped Arrow) dataset and apply_chat_template each row
        # ON DEMAND in __getitem__, instead of eagerly rendering EVERY row into Python lists here.
        # Eager preprocessing held the entire rendered corpus in RAM (self.prompts/labels/images),
        # which OOMs and stalls __init__ on large datasets (e.g. multi-million-row, ~100KB/prompt
        # multimodal corpora). Lazy keeps init O(1) memory + near-instant; the dataloader workers
        # render only the rows actually sampled. __getitem__'s return value is unchanged.
        self.dataset = dataset
        self.input_key = getattr(self.strategy.args.data, "input_key", None)
        self.label_key = getattr(self.strategy.args.data, "label_key", None)
        self.tools_key = getattr(self.strategy.args.data, "tools_key", None)
        self.image_key = getattr(self.strategy.args.data, "image_key", "images")
        apply_chat_template = getattr(self.strategy.args.data, "apply_chat_template", False)
        self.apply_chat_template = self.tokenizer.apply_chat_template if apply_chat_template else None
        self.expand_image_placeholder = should_expand_image_placeholder(self.tokenizer)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Render this row's prompt on demand (lazy). The Arrow-backed dataset is memory-mapped,
        # so indexing reads a single row without materializing the corpus.
        data = self.dataset[idx]
        tools = data.get(self.tools_key) if self.tools_key else None
        prompt, label = preprocess_data(
            data,
            self.input_key,
            self.label_key,
            self.apply_chat_template,
            prerender=self.prerender,
            expand_image_placeholder=self.expand_image_placeholder,
            tools=tools,
        )
        return data.get("datasource", "default"), prompt, label, data.get(self.image_key, None), tools

    def collate_fn(self, item_list):
        datasources = []
        prompts = []
        labels = []
        images = []
        tools = []
        for datasource, prompt, label, img, tool in item_list:
            datasources.append(datasource)
            prompts.append(prompt)
            labels.append(label)
            images.append(img)
            tools.append(tool)

        return datasources, prompts, labels, images, tools
