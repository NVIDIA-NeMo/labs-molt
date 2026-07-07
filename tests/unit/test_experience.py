from types import SimpleNamespace

import torch

from molt.trainer.algorithm.experience import Experience, balance_experiences, get_model_parallel_size


def _args(cp=1, tp=1, ep=1, actor_gpus=1):
    return SimpleNamespace(
        actor=SimpleNamespace(num_nodes=1, num_gpus_per_node=actor_gpus),
        fsdp=SimpleNamespace(cp_size=cp, tp_size=tp, ep_size=ep),
    )


def test_model_parallel_size_excludes_ep():
    assert get_model_parallel_size(_args(cp=2, tp=3, ep=4)) == 6


def test_balance_experiences_uses_fsdp_data_parallel_size():
    exp = Experience(
        sequences=torch.arange(8).view(4, 2),
        attention_mask=torch.ones(4, 2, dtype=torch.long),
        total_length=torch.tensor([8, 7, 6, 5]),
    )

    balanced = balance_experiences([exp], _args(ep=2, actor_gpus=4))

    assert len(balanced) == 4
    assert [len(item.sequences) for item in balanced] == [1, 1, 1, 1]


def test_balance_experiences_equalizes_per_rank_counts():
    # 10 samples across 4 DP ranks: the 2-sample remainder is dropped so every
    # rank receives the SAME count. Unequal counts would desync num_steps across
    # ranks and deadlock the world all_reduce in setup_dynamic_batch.
    exp = Experience(
        sequences=torch.arange(20).view(10, 2),
        attention_mask=torch.ones(10, 2, dtype=torch.long),
        total_length=torch.arange(10, 0, -1),
    )

    balanced = balance_experiences([exp], _args(actor_gpus=4))

    assert len(balanced) == 4
    counts = [len(item.sequences) for item in balanced]
    assert counts == [2, 2, 2, 2]  # equal counts, trailing remainder dropped
    assert sum(counts) == 8
