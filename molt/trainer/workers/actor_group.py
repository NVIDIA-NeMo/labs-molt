# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from OpenRLHF (https://github.com/OpenRLHF/OpenRLHF),
# Copyright (c) OpenRLHF contributors, licensed under the Apache License, Version 2.0.

import logging
import os
import socket
from typing import Dict, Type

import ray
import torch
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from tqdm import tqdm

from molt.models import Actor
from molt.trainer.fsdp import FsdpStrategy
from molt.trainer.placement import get_bundle_indices, ray_noset_visible_devices


class BaseDistributedActor:
    def __init__(self, world_size, rank, master_addr, master_port):
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._world_size = world_size
        self._rank = rank
        self._master_addr = master_addr if master_addr else self._get_current_node_ip()
        self._master_port = master_port if master_port else self._get_free_port()
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # NOTE: Ray will automatically set CUDA_VISIBLE_DEVICES for each actor,
        # unless RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES is set, so
        # set local rank to 0 when the flag is not applicable.
        os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0]) if ray_noset_visible_devices() else "0"

    @staticmethod
    def _get_current_node_ip():
        address = ray._private.services.get_node_ip_address()
        # strip ipv6 address
        return address.strip("[]")

    @staticmethod
    def _get_free_port():
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def get_master_addr_port(self):
        return self._master_addr, self._master_port


class BaseModelActor(BaseDistributedActor):
    def _setup_distributed(self, strategy: FsdpStrategy):
        # configure strategy
        self.strategy = strategy
        strategy.setup_distributed()

    def init_model_from_pretrained(self, *args, **kwargs):
        raise NotImplementedError()

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def execute_batch(self, method_name: str, all_data, start_idx, end_idx):
        """Call ``self.<method_name>`` once per item in a slice of the batched data.

        Args:
            method_name (str): name of the per-sample method to call.
            all_data (dict): dict of equal-length lists; each is sliced to
                ``[start_idx:end_idx]`` and zipped into per-item kwargs.
            start_idx (int): start of this worker's slice (inclusive).
            end_idx (int): end of this worker's slice (exclusive).

        Returns:
            List[Any]: one result per item in the slice.
        """

        # Get the first parameter to determine list length
        kwargs = {key: value[start_idx:end_idx] for key, value in all_data.items()}
        first_param = next(iter(kwargs.values()))
        list_length = len(first_param)

        # Verify all parameters have same length
        for param_name, param_value in kwargs.items():
            if len(param_value) != list_length:
                raise ValueError(f"Parameter {param_name} has length {len(param_value)}, expected {list_length}")

        # Get the function to execute
        func = getattr(self, method_name)
        if not callable(func):
            raise ValueError(f"Function {method_name} is not callable")

        results = []
        for i in tqdm(range(list_length), desc=f"{method_name}", disable=not self.strategy.is_rank_0()):
            # Create kwargs for single item
            sample_kwargs = {param_name: param_value[i] for param_name, param_value in kwargs.items()}

            result = func(**sample_kwargs)
            results.append(result)

        return results


@ray.remote(num_gpus=1)
class ReferenceModelActor(BaseModelActor):
    def init_model_from_pretrained(self, strategy: FsdpStrategy, pretrain):
        self._setup_distributed(strategy)
        model = Actor(
            pretrain,
            attn_implementation=strategy.args.fsdp.attn_implementation,
            param_dtype=strategy.args.fsdp.param_dtype,
            device_mesh=strategy.device_mesh,
            moe_mesh=strategy.moe_mesh,
            distributed_config=strategy.distributed_config,
            moe_config=strategy.moe_config,
            activation_checkpointing=False,
            packing_samples=strategy.args.fsdp.packing_samples,
            temperature=strategy.args.rollout.temperature,
            # Keep reference numerics on the same AutoModel path as the
            # trainable actor. Custom AutoModel Llama is not fp32/bf16 forward
            # equivalent under autocast, so bf16 ref makes step-0 KL non-zero.
            use_fp32_master_weights=True,
        )
        strategy.print(model)

        self.model = self.strategy.prepare(model)
        self.model.eval()

    def forward(self, experience) -> torch.Tensor:
        """Reference log-probs for one rollout Experience. reload() first fetches the sample's heavy
        tensors (token ids / images) from the producing runner's shared-memory store — they reach
        this rank straight from the runner, never through the controller. Called per sample by
        execute_batch; the controller attaches the result as base_action_log_probs."""
        experience = experience.reload()
        device = torch.cuda.current_device()

        # VLM: merge pre-processed multimodal inputs.
        mm_inputs = {}
        if experience.mm_train_inputs and getattr(self.model, "is_vlm", False):
            from molt.utils.vlm_utils import merge_mm_train_inputs

            mm_inputs = merge_mm_train_inputs(experience.mm_train_inputs, device)

        with torch.no_grad():
            output = self.model(
                experience.sequences.to(device),
                experience.action_mask.to(device),
                experience.attention_mask.to(device),
                **mm_inputs,
            )
        return output["action_log_probs"].to("cpu")


class RayActorGroup:
    """
    A group of ray actors
    Functions start with 'async' should return list of object refs

    Args:
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        ray_actor_type (Type[BaseModelActor]): model actor type served by this group.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
    """

    def __init__(
        self,
        num_nodes,
        num_gpus_per_node,
        ray_actor_type: Type[BaseModelActor],
        pg: PlacementGroup = None,
        num_gpus_per_actor=1,
        duplicate_actors: int = 1,
        resources: Dict[str, float] = None,
        num_resources_per_node: int = None,
    ) -> None:
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.ray_actor_type = ray_actor_type
        # CP/TP ranks share data; EP ranks do not.
        self.duplicate_actors = duplicate_actors

        # custom resources, see https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        self._resources = resources
        self._num_resources_per_node = num_resources_per_node

        self._initiate_actors(pg, num_gpus_per_actor)

    def _initiate_actors(self, pg, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node

        # Use placement group to lock resources for models of same type
        if self._num_gpus_per_node > 1 and pg is None:
            bundles = [{"GPU": 1, "CPU": 1} for _ in range(self._num_nodes * self._num_gpus_per_node)]
            if self._resources:
                resources_name = list(self._resources.keys())[0]
                for i in range(len(bundles)):
                    bundles[i][resources_name] = self._num_resources_per_node

            # PACK (not SPREAD) packs the actor's GPUs onto as few nodes as possible.
            pg = placement_group(bundles, strategy="PACK")

        # Ray PACK does NOT guarantee bundle index == node order (ray #51117): at
        # >2 nodes Ray can interleave bundle indices across nodes, so rank i -> bundle i
        # scatters the EP=8/CP=8 group across a node boundary and deep_ep's intra-node
        # NVLink dispatch hits a cross-node peer (illegal memory access, deep_ep.cpp:278;
        # 2 nodes happened to pack contiguously, 4 nodes did not). Map rank i -> the i-th
        # bundle in NODE-SORTED order so each EP/CP group stays within one node.
        bundle_order = list(range(world_size))
        if pg is not None:
            ray.get(pg.ready())
            bundle_order = get_bundle_indices(pg, 0, world_size)

        def _spawn(rank, master_addr, master_port):
            options = {
                "num_cpus": num_gpus_per_actor,
                "num_gpus": num_gpus_per_actor,
                "resources": self._resources,
            }
            if pg:
                options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=bundle_order[rank]
                )
            return self.ray_actor_type.options(**options).remote(world_size, rank, master_addr, master_port)

        master_actor = _spawn(0, None, None)
        self._actor_handlers = [master_actor]
        if world_size > 1:
            master_addr, master_port = ray.get(master_actor.get_master_addr_port.remote())
            for rank in range(1, world_size):
                self._actor_handlers.append(_spawn(rank, master_addr, master_port))

    def async_init_model_from_pretrained(
        self,
        *args,
        **kwargs,
    ):
        """Init model from pretrained checkpoint.

        Returns:
            List: list of remote object refs.
        """
        return [actor.init_model_from_pretrained.remote(*args, **kwargs) for actor in self._actor_handlers]

    def async_export_hf_model(self):
        """Export HF safetensors snapshot from each actor (rank 0 writes).

        Returns:
            List: list of remote object refs.
        """
        return [actor.export_hf_model.remote() for actor in self._actor_handlers]

    def async_run_method(self, method_name, *args, **kwargs):
        refs = []
        for actor in self._actor_handlers:
            method = getattr(actor, method_name)
            refs.append(method.remote(*args, **kwargs))
        return refs

    def async_run_method_batch(self, method_name, **kwargs):
        """Run method on all actors with batched input data asynchronously using round-robin scheduling.
        Each actor processes one chunk of data at a time. Actors in the same ring / tensor parallel group process the same chunk.

        Args:
            method_name (str): Name of the method to run
            **kwargs: Keyword arguments for the method. Each value should be a list/tensor of the same length.

        Returns:
            List[ray.ObjectRef]: List of remote object references to the results
        """
        # Check if all kwargs parameters are iterable
        for key, value in kwargs.items():
            if not hasattr(value, "__len__"):
                raise ValueError(f"Parameter {key} must be iterable")

        # Get the length of the first parameter as reference
        first_param = next(iter(kwargs.values()))
        total_length = len(first_param)

        # Verify all parameters have the same length
        for key, value in kwargs.items():
            if len(value) != total_length:
                raise ValueError(
                    f"All parameters must have the same length. {key} has length {len(value)}, expected {total_length}"
                )

        # Calculate chunk size based on number of effective actors (considering ring groups)
        num_actors = len(self._actor_handlers)
        effective_actors = num_actors // self.duplicate_actors
        if total_length == 0 or total_length < effective_actors:
            raise ValueError(
                f"Insufficient batch size for async_run_method_batch: total_length={total_length}, "
                f"effective_actors={effective_actors}"
            )
        base_chunk_size, remainder = divmod(total_length, effective_actors)
        # Each forward is an FSDP all-gather (a collective over the DP group), so every DP rank must run
        # the same number of forwards; an uneven split would deadlock the rank with an extra item. The
        # batch is built to divide evenly (samples_generator hands back complete groups => a multiple of
        # n_samples), so fail loud here if that invariant is ever broken rather than silently hang.
        if remainder != 0:
            raise ValueError(
                f"async_run_method_batch: batch of {total_length} is not divisible across "
                f"{effective_actors} DP ranks (remainder {remainder}); an uneven per-rank forward count "
                "deadlocks the FSDP all-gather. Ensure the rollout batch is a multiple of the DP-rank "
                "count (complete groups: rollout.batch_size * n_samples_per_prompt)."
            )

        # Pre-slice data before ray.put so each worker only receives its chunk.
        # This avoids transferring the full batch to every node (critical at scale).
        refs = []
        for chunk_idx in range(effective_actors):
            start_idx = chunk_idx * base_chunk_size
            end_idx = start_idx + base_chunk_size

            chunk_data = {key: value[start_idx:end_idx] for key, value in kwargs.items()}
            chunk_ref = ray.put(chunk_data)

            for j in range(self.duplicate_actors):
                actor_idx = chunk_idx * self.duplicate_actors + j
                actor = self._actor_handlers[actor_idx]
                refs.append(actor.execute_batch.remote(method_name, chunk_ref, 0, base_chunk_size))

        return refs
