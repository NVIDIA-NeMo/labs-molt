# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.


import pytest

from nemo_automodel.components.models.glm5_next.processing import (
    _MEDIA_REMINDER,
    _enable_image_placeholders,
)

# Over the default 5s budget on purpose: this module launches a fresh interpreter, which re-imports torch from scratch.
# Shrink the work or the process count before raising this further.
pytestmark = pytest.mark.timeout(60)


def test_training_template_enables_images_but_keeps_other_media_disabled():
    template = "prefix " + _MEDIA_REMINDER + " suffix"

    patched = _enable_image_placeholders(template)

    assert "<|begin_of_image|><|image|><|end_of_image|>" in patched
    assert "media_type == 'image'" in patched
    assert "unable to process this" in patched


def test_existing_multimodal_template_is_not_rewritten():
    template = "{{ '<|image|>' }}"
    assert _enable_image_placeholders(template) == template


def test_unrecognized_text_only_template_fails_loudly():
    with pytest.raises(ValueError, match="no recognized media rendering branch"):
        _enable_image_placeholders("{{ messages }}")


