import torch

from molt.trainer.fsdp.refit import should_refit_state_dict_entry, state_dict_parameter_trainability


class _TiedHeadModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(4, 3)
        self.lm_head = torch.nn.Linear(3, 4, bias=False)
        self.lm_head.weight = self.embed_tokens.weight


def test_state_dict_trainability_preserves_tied_parameter_aliases():
    model = _TiedHeadModel()

    trainability = state_dict_parameter_trainability(model)

    assert "embed_tokens.weight" in trainability
    assert "lm_head.weight" in trainability
    assert should_refit_state_dict_entry(
        "lm_head.weight",
        model.state_dict()["lm_head.weight"],
        trainability,
        is_vlm=False,
    )


def test_refit_filter_skips_non_parameter_buffers():
    trainability = {}

    assert not should_refit_state_dict_entry(
        "layers.0.mlp.gate.e_score_correction_bias", torch.zeros(2), trainability, is_vlm=False
    )
    assert not should_refit_state_dict_entry("some_metadata", torch.zeros(2), trainability, is_vlm=False)
