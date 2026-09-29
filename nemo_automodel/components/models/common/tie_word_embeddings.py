# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model-owned policy for input and output embedding ties."""

from nemo_automodel.shared.tied_weights import TieSupport, get_controlling_tie_word_embeddings


def reject_unsupported_tie_word_embeddings(model_cls: type, config: object) -> None:
    """Reject a ``tie_word_embeddings`` setting the model class cannot honor.

    Reads the class's declared :class:`TieSupport` policy from
    ``model_cls.tie_word_embeddings_support`` (defaulting to
    :attr:`TieSupport.BOTH`) and the controlling ``tie_word_embeddings`` flag
    from :func:`get_controlling_tie_word_embeddings`, then raises when the
    requested tying falls outside the supported set:

    - ``UNTIED_ONLY`` classes are separate-head architectures (HF default: distinct
      input/output embeddings) that never build a shared ``lm_head``; honoring
      ``tie_word_embeddings=True`` would leave a randomly-initialized head.
    - ``TIED_ONLY`` classes share ``lm_head`` with the input embedding and ship
      checkpoints without a distinct ``lm_head.weight``; honoring
      ``tie_word_embeddings=False`` would require materializing a head NeMo does
      not build and the checkpoint has no weights for.

    Call at the top of ``__init__`` on the *original* top-level config, before
    unwrapping to ``text_config`` / ``thinker_config`` or calling
    ``super().__init__()``, so the correct controlling flag is inspected and
    construction fails fast.

    Args:
        model_cls: The constructing model class (``type(self)``). Its
            ``tie_word_embeddings_support`` attribute declares the policy, and its
            name is included in validation errors.
        config: The model's original (top-level) config.

    Raises:
        NotImplementedError: If tying is requested on a
            :attr:`TieSupport.UNTIED_ONLY` class, or untying is requested on a
            :attr:`TieSupport.TIED_ONLY` class.
    """
    support = getattr(model_cls, "tie_word_embeddings_support", TieSupport.BOTH)
    if support is TieSupport.BOTH:
        return

    model_class_name = model_cls.__name__
    requested_tied = get_controlling_tie_word_embeddings(config)

    if support is TieSupport.UNTIED_ONLY and requested_tied:
        raise NotImplementedError(
            f"{model_class_name} has separate input and output embeddings and does not "
            f"support tie_word_embeddings=True. The Hugging Face default for this "
            f"architecture is untied; set tie_word_embeddings=False."
        )
    if support is TieSupport.TIED_ONLY and not requested_tied:
        raise NotImplementedError(
            f"{model_class_name} ties its input and output embeddings and does not "
            f"support tie_word_embeddings=False. The Hugging Face default for this "
            f"architecture is tied; set tie_word_embeddings=True."
        )


def reject_tie_word_embeddings_flip(checkpoint_config: object, requested_config: object, model_class_name: str) -> None:
    """Reject loading a checkpoint with ``tie_word_embeddings`` flipped from its own value.

    The class-level :class:`TieSupport` declaration cannot catch flipping the flag away
    from a *specific* checkpoint's value (a ``BOTH`` class accepts either) -- that is a
    ``(checkpoint, requested)`` property. NeMo AutoModel respects the checkpoint's tie
    semantics, so a mismatch in either direction is rejected:

    - untied checkpoint requested tied -> would silently discard the trained ``lm_head``;
    - tied checkpoint requested untied -> would leave a randomly-initialized ``lm_head``
      (the adapters' embed->head copy is gated on the flag).

    Only applies to ``from_pretrained`` (there is a loaded checkpoint to compare against);
    ``from_config`` / scratch has no checkpoint, so the user's config is authoritative.

    Args:
        checkpoint_config: The config parsed from the checkpoint, before user overrides.
        requested_config: The config after user overrides are applied.
        model_class_name: The resolved model class name to include in validation errors.

    Raises:
        NotImplementedError: If the controlling ``tie_word_embeddings`` flag differs
            between the checkpoint and the requested config.
    """
    checkpoint_tied = get_controlling_tie_word_embeddings(checkpoint_config)
    requested_tied = get_controlling_tie_word_embeddings(requested_config)
    if checkpoint_tied != requested_tied:
        raise NotImplementedError(
            f"{model_class_name}: requested tie_word_embeddings={requested_tied} but the checkpoint "
            f"declares tie_word_embeddings={checkpoint_tied}. NeMo AutoModel respects the checkpoint's "
            f"tie semantics; flipping the flag is not supported (it would leave a randomly-initialized "
            f"or discarded lm_head). Load the checkpoint with its own tie_word_embeddings value."
        )
