# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Full definition of a GPT NeoX Language Model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from typing_extensions import Self

# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Type, Union

import torch
from typing_extensions import Self

import language.gpt.model as model

# import model
from utils import find_multiple


@dataclass
class Config:
    name: str = ""
    hf_config: dict = field(default_factory=dict)
    block_size: int = 4096
    vocab_size: int = 50254
    padding_multiple: int = 512
    padded_vocab_size: Optional[int] = None
    n_layer: int = 16
    n_head: int = 32
    n_embd: int = 4096
    rotary_percentage: float = 0.25
    parallel_residual: bool = True
    bias: bool = True
    lm_head_bias: bool = False
    # to use multi-head attention (MHA), set this to `n_head` (default)
    # to use multi-query attention (MQA), set this to 1
    # to use grouped-query attention (GQA), set this to a value in between
    # Example with `n_head=4`
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │         │        │                 │
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
    # ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
    # │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
    # └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
    # ◀──────────────────▶  ◀──────────────────▶  ◀──────────────────▶
    #         MHA                    GQA                   MQA
    #   n_query_groups=4       n_query_groups=2      n_query_groups=1
    #
    # credit https://arxiv.org/pdf/2305.13245.pdf
    n_query_groups: Optional[int] = None
    shared_attention_norm: bool = False
    _norm_class: Literal["LayerNorm", "RMSNorm"] = "LayerNorm"
    norm_eps: float = 1e-5
    _mlp_class: Literal["GptNeoxMLP", "LLaMAMLP"] = "GptNeoxMLP"
    gelu_approximate: str = "none"
    intermediate_size: Optional[int] = None
    rope_condense_ratio: int = 1
    rope_base: int = 10000
    n_expert: int = 0
    n_expert_per_token: int = 0

    def __post_init__(self):
        if not self.name:
            self.name = self.hf_config.get("name", self.name)

        assert self.n_embd % self.n_head == 0
        self.head_size = self.n_embd // self.n_head

        # vocab size should be a power of 2 to be optimal on hardware. compute the closest value
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, self.padding_multiple)
        else:
            # vocab size shouldn't be larger than padded vocab size
            self.vocab_size = min(self.vocab_size, self.padded_vocab_size)

        # compute the number of query groups
        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head

        # compute the intermediate size for MLP if not set
        if self.intermediate_size is None:
            if self._mlp_class == "LLaMAMLP":
                raise ValueError("The config needs to set the `intermediate_size`")
            self.intermediate_size = 4 * self.n_embd

        self.rope_n_elem = int(self.rotary_percentage * self.head_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        if name not in name_to_config:
            # search through all `config['hf_config']['name']`
            try:
                conf_dict = next(config for config in configs if name == config["hf_config"]["name"])
            except StopIteration:
                raise ValueError(f"{name!r} is not a supported config name")
        else:
            conf_dict = name_to_config[name]

        conf_dict = conf_dict.copy()
        if "condense_ratio" in kwargs:  # legacy name
            kwargs["rope_condense_ratio"] = kwargs.pop("condense_ratio")
        conf_dict.update(kwargs)
        return cls(**conf_dict)

    @classmethod
    def from_json(cls, path: Union[str, Path], **kwargs: Any) -> Self:
        with open(path, encoding="utf-8") as fp:
            json_kwargs = json.load(fp)
        if "condense_ratio" in json_kwargs:  # legacy name
            json_kwargs["rope_condense_ratio"] = json_kwargs.pop("condense_ratio")
        if "condense_ratio" in kwargs:  # legacy name
            kwargs["rope_condense_ratio"] = kwargs.pop("condense_ratio")
        if "org" in json_kwargs:  # legacy name
            json_kwargs["hf_config"] = {"name": json_kwargs["name"], "org": json_kwargs.pop("org")}
        if "org" in kwargs:  # legacy name
            kwargs["hf_config"] = {"name": kwargs.get("name", json_kwargs["name"]), "org": kwargs.pop("org")}
        json_kwargs.update(kwargs)
        return cls(**json_kwargs)

    @classmethod
    def from_checkpoint(cls, path: Path, **kwargs: Any) -> Self:
        """Automatically load `lit_config.json` and if it doesn't exist - a matching config from `lit_gpt/config.py`."""
        if (config_path := path / "lit_config.json").is_file():
            return cls.from_json(config_path, **kwargs)
        if (model_name := path.name) in name_to_config:
            return cls.from_name(model_name, **kwargs)
        raise FileNotFoundError(f"For {str(path)!r} neither 'lit_config.json' nor matching config exists.")

    @property
    def mlp_class(self) -> Type:
        # `self._mlp_class` cannot be the type to keep the config json serializable
        return getattr(model, self._mlp_class)

    @property
    def norm_class(self) -> Type:
        # `self._norm_class` cannot be the type to keep the config json serializable
        if self._norm_class == "RMSNorm":
            from rmsnorm import RMSNorm

            return RMSNorm
        return getattr(torch.nn, self._norm_class)


########################
# Stability AI StableLM
########################
configs = [
    # https://huggingface.co/stabilityai/stablelm-base-alpha-3b/blob/main/config.json
    dict(name="stablelm-base-alpha-3b", hf_config=dict(org="stabilityai", name="stablelm-base-alpha-3b")),
    # https://huggingface.co/stabilityai/stablelm-base-alpha-7b/blob/main/config.json
    dict(
        name="stablelm-base-alpha-7b",
        hf_config=dict(org="stabilityai", name="stablelm-base-alpha-7b"),
        n_head=48,
        n_embd=6144,
        padding_multiple=256,
    ),
    # https://huggingface.co/stabilityai/stablelm-tuned-alpha-3b/blob/main/config.json
    dict(name="stablelm-tuned-alpha-3b", hf_config=dict(org="stabilityai", name="stablelm-tuned-alpha-3b"), n_head=32),
    # https://huggingface.co/stabilityai/stablelm-tuned-alpha-7b/blob/main/config.json
    dict(
        name="stablelm-tuned-alpha-7b",
        hf_config=dict(org="stabilityai", name="stablelm-tuned-alpha-7b"),
        n_head=48,
        n_embd=6144,
        padding_multiple=256,
    ),
    # https://huggingface.co/stabilityai/stablelm-zephyr-3b/blob/main/config.json
    dict(
        name="stablelm-zephyr-3b",
        hf_config=dict(org="stabilityai", name="stablelm-zephyr-3b"),
        padded_vocab_size=50304,
        n_layer=32,
        n_head=32,
        n_embd=2560,
        parallel_residual=False,
        bias=False,
        _mlp_class="LLaMAMLP",
        intermediate_size=6912,
    ),
]

####################
# EleutherAI Pythia
####################
pythia = [
    # https://huggingface.co/EleutherAI/pythia-14m/blob/main/config.json
    dict(
        name="pythia-14m",
        hf_config=dict(org="EleutherAI", name="pythia-14m"),
        block_size=512,
        n_layer=6,
        n_embd=128,
        n_head=4,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-31m/blob/main/config.json
    dict(
        name="pythia-31m",
        hf_config=dict(org="EleutherAI", name="pythia-31m"),
        block_size=1024,
        n_layer=6,
        n_embd=256,
        n_head=8,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-70m/blob/main/config.json
    dict(
        name="pythia-70m",
        hf_config=dict(org="EleutherAI", name="pythia-70m"),
        block_size=2048,
        n_layer=6,
        n_embd=512,
        n_head=8,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-160m/blob/main/config.json
    dict(
        name="pythia-160m",
        hf_config=dict(org="EleutherAI", name="pythia-160m"),
        block_size=2048,
        n_layer=12,
        n_embd=768,
        n_head=12,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-410m/blob/main/config.json
    dict(
        name="pythia-410m",
        hf_config=dict(org="EleutherAI", name="pythia-410m"),
        block_size=2048,
        n_layer=24,
        n_embd=1024,
        n_head=16,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-1b/blob/main/config.json
    dict(
        name="pythia-1b",
        hf_config=dict(org="EleutherAI", name="pythia-1b"),
        block_size=2048,
        n_embd=2048,
        n_head=8,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-1.4b/blob/main/config.json
    dict(
        name="pythia-1.4b",
        hf_config=dict(org="EleutherAI", name="pythia-1.4b"),
        block_size=2048,
        n_layer=24,
        n_embd=2048,
        n_head=16,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-2.8b/blob/main/config.json
    dict(
        name="pythia-2.8b",
        hf_config=dict(org="EleutherAI", name="pythia-2.8b"),
        block_size=2048,
        n_layer=32,
        n_embd=2560,
        padding_multiple=128,
    ),
    # https://huggingface.co/EleutherAI/pythia-6.9b/blob/main/config.json
    dict(
        name="pythia-6.9b",
        hf_config=dict(org="EleutherAI", name="pythia-6.9b"),
        block_size=2048,
        n_layer=32,
        padding_multiple=256,
    ),
    # https://huggingface.co/EleutherAI/pythia-12b/blob/main/config.json
    dict(
        name="pythia-12b",
        hf_config=dict(org="EleutherAI", name="pythia-12b"),
        block_size=2048,
        n_layer=36,
        n_embd=5120,
        n_head=40,
    ),
]
configs.extend(pythia)
for c in pythia:
    # "pythia-14m" and "pythia-31m" don't have deduped version
    if c["name"] in ("pythia-14m", "pythia-31m"):
        continue
    copy = deepcopy(c)
    copy["name"] = f"{c['name']}-deduped"
    copy["hf_config"]["name"] = f"{c['hf_config']['name']}-deduped"
    configs.append(copy)


###################
# databricks Dolly
###################
dolly = [
    # https://huggingface.co/databricks/dolly-v2-3b/blob/main/config.json
    dict(
        name="dolly-v2-3b",
        hf_config=dict(org="databricks", name="dolly-v2-3b"),
        block_size=2048,
        n_layer=32,
        n_embd=2560,
        padded_vocab_size=50280,
    ),
    # https://huggingface.co/databricks/dolly-v2-7b/blob/main/config.json
    dict(
        name="dolly-v2-7b",
        hf_config=dict(org="databricks", name="dolly-v2-7b"),
        block_size=2048,
        n_layer=32,
        padded_vocab_size=50280,
    ),
    # https://huggingface.co/databricks/dolly-v2-12b/blob/main/config.json
    dict(
        name="dolly-v2-12b",
        hf_config=dict(org="databricks", name="dolly-v2-12b"),
        block_size=2048,
        n_layer=36,
        n_embd=5120,
        n_head=40,
        padded_vocab_size=50280,
    ),
]
configs.extend(dolly)


####################################
# togethercomputer RedPajama INCITE
####################################
redpajama_incite = [
    # https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-3B-v1/blob/main/config.json
    dict(
        name="RedPajama-INCITE-{}-3B-v1",
        hf_config=dict(org="togethercomputer", name="RedPajama-INCITE-{}-3B-v1"),
        block_size=2048,
        n_layer=32,
        n_embd=2560,
        padding_multiple=256,
        rotary_percentage=1.0,
        parallel_residual=False,
    ),
    # https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Base/blob/main/config.json
    dict(
        name="RedPajama-INCITE-7B-{}",
        hf_config=dict(org="togethercomputer", name="RedPajama-INCITE-7B-{}"),
        block_size=2048,
        n_layer=32,
        padding_multiple=256,
        rotary_percentage=1.0,
        parallel_residual=False,
    ),
    # this redirects to the checkpoint above. kept for those who had the old weights already downloaded
    dict(
        name="RedPajama-INCITE-{}-7B-v0.1",
        hf_config=dict(org="togethercomputer", name="RedPajama-INCITE-{}-7B-v0.1"),
        block_size=2048,
        n_layer=32,
        padding_multiple=256,
        rotary_percentage=1.0,
        parallel_residual=False,
    ),
]
for c in redpajama_incite:
    for kind in ("Base", "Chat", "Instruct"):
        copy = deepcopy(c)
        copy["name"] = c["name"].format(kind)
        copy["hf_config"]["name"] = c["hf_config"]["name"].format(kind)
        configs.append(copy)


#################
# TII UAE Falcon
#################
falcon = [
    # https://huggingface.co/tiiuae/falcon-7b/blob/main/config.json
    dict(
        name="falcon-7b{}",
        hf_config=dict(org="tiiuae", name="falcon-7b{}"),
        block_size=2048,
        vocab_size=65024,
        padded_vocab_size=65024,
        n_layer=32,
        n_head=71,
        n_embd=4544,
        rotary_percentage=1.0,
        n_query_groups=1,
        bias=False,
        # this is not in the config, but in the original model implementation, only for this config
        shared_attention_norm=True,
    ),
    # https://huggingface.co/tiiuae/falcon-40b/blob/main/config.json
    dict(
        name="falcon-40b{}",
        hf_config=dict(org="tiiuae", name="falcon-40b{}"),
        block_size=2048,
        vocab_size=65024,
        padded_vocab_size=65024,
        n_layer=60,
        n_head=128,
        n_embd=8192,
        rotary_percentage=1.0,
        n_query_groups=8,
        bias=False,
    ),
]
for c in falcon:
    for kind in ("", "-instruct"):
        copy = deepcopy(c)
        copy["name"] = c["name"].format(kind)
        copy["hf_config"]["name"] = c["hf_config"]["name"].format(kind)
        configs.append(copy)

# https://huggingface.co/tiiuae/falcon-180b/blob/main/config.json
falcon180b = dict(
    name="falcon-180B{}",
    hf_config=dict(org="tiiuae", name="falcon-180B{}"),
    block_size=2048,
    vocab_size=65024,
    padded_vocab_size=65024,
    n_layer=80,
    n_head=232,
    n_embd=14848,
    rotary_percentage=1.0,
    n_query_groups=8,
    bias=False,
)

for kind in ("", "-chat"):
    copy = deepcopy(falcon180b)
    copy["name"] = falcon180b["name"].format(kind)
    copy["hf_config"]["name"] = falcon180b["hf_config"]["name"].format(kind)
    configs.append(copy)


#############################
# OpenLM Research Open LLaMA
#############################
open_LLaMA = [
    # https://huggingface.co/openlm-research/open_llama_3b/blob/main/config.json
    dict(
        name="open_llama_3b",
        hf_config=dict(org="openlm-research", name="open_llama_3b"),
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=26,
        n_embd=3200,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-6,
        _mlp_class="LLaMAMLP",
        intermediate_size=8640,
    ),
    # https://huggingface.co/openlm-research/open_llama_7b/blob/main/config.json
    dict(
        name="open_llama_7b",
        hf_config=dict(org="openlm-research", name="open_llama_7b"),
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-6,
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
    ),
    # https://huggingface.co/openlm-research/open_llama_13b/blob/main/config.json
    dict(
        name="open_llama_13b",
        hf_config=dict(org="openlm-research", name="open_llama_13b"),
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-6,
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
    ),
]
configs.extend(open_LLaMA)


###############
# LMSYS Vicuna
###############
vicuna = [
    # https://huggingface.co/lmsys/vicuna-7b-v1.3/blob/main/config.json
    dict(
        name="vicuna-7b-v1.3",
        hf_config=dict(org="lmsys", name="vicuna-7b-v1.3"),
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-6,
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
    ),
    # https://huggingface.co/lmsys/vicuna-13b-v1.3/blob/main/config.json
    dict(
        name="vicuna-13b-v1.3",
        hf_config=dict(org="lmsys", name="vicuna-13b-v1.3"),
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-6,
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
    ),
    # https://huggingface.co/lmsys/vicuna-33b-v1.3/blob/main/config.json
    dict(
        name="vicuna-33b-v1.3",
        hf_config=dict(org="lmsys", name="vicuna-33b-v1.3"),
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=60,
        n_head=52,
        n_embd=6656,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-6,
        _mlp_class="LLaMAMLP",
        intermediate_size=17920,
    ),
    # https://huggingface.co/lmsys/vicuna-7b-v1.5/blob/main/config.json
    dict(
        name="vicuna-7b-v1.5",
        hf_config=dict(org="lmsys", name="vicuna-7b-v1.5"),
        vocab_size=32000,
        padding_multiple=64,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
    ),
    # https://huggingface.co/lmsys/vicuna-7b-v1.5-16k/blob/main/config.json
    dict(
        name="vicuna-7b-v1.5-16k",
        hf_config=dict(org="lmsys", name="vicuna-7b-v1.5-16k"),
        block_size=16384,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
        rope_condense_ratio=4,
    ),
    # https://huggingface.co/lmsys/vicuna-13b-v1.5/blob/main/config.json
    dict(
        name="vicuna-13b-v1.5",
        hf_config=dict(org="lmsys", name="vicuna-13b-v1.5"),
        vocab_size=32000,
        padding_multiple=64,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
    ),
    # https://huggingface.co/lmsys/vicuna-13b-v1.5-16k/blob/main/config.json
    dict(
        name="vicuna-13b-v1.5-16k",
        hf_config=dict(org="lmsys", name="vicuna-13b-v1.5-16k"),
        block_size=16384,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
        rope_condense_ratio=4,
    ),
]
configs.extend(vicuna)


#################
# LMSYS LongChat
#################
long_chat = [
    # https://huggingface.co/lmsys/longchat-7b-16k/blob/main/config.json
    dict(
        name="longchat-7b-16k",
        hf_config=dict(org="lmsys", name="longchat-7b-16k"),
        block_size=16384,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-6,
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
        rope_condense_ratio=8,
    ),
    # https://huggingface.co/lmsys/longchat-13b-16k/blob/main/config.json
    dict(
        name="longchat-13b-16k",
        hf_config=dict(org="lmsys", name="longchat-13b-16k"),
        block_size=16384,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-6,
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
        rope_condense_ratio=8,
    ),
]
configs.extend(long_chat)


######################
# NousResearch Hermes
######################
nous_research = [
    # https://huggingface.co/NousResearch/Nous-Hermes-llama-2-7b/blob/main/config.json
    dict(
        name="Nous-Hermes-llama-2-7b",
        hf_config=dict(org="NousResearch", name="Nous-Hermes-llama-2-7b"),
        padded_vocab_size=32000,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
    ),
    # https://huggingface.co/NousResearch/Nous-Hermes-13B/blob/main/config.json
    dict(
        name="Nous-Hermes-13b",
        hf_config=dict(org="NousResearch", name="Nous-Hermes-13b"),
        block_size=2048,
        vocab_size=32000,
        padded_vocab_size=32001,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-6,
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
    ),
    # https://huggingface.co/NousResearch/Nous-Hermes-Llama2-13b
    dict(
        name="Nous-Hermes-Llama2-13b",
        hf_config=dict(org="NousResearch", name="Nous-Hermes-Llama2-13b"),
        vocab_size=32000,
        padded_vocab_size=32032,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
    ),
]
configs.extend(nous_research)


###############
# Meta LLaMA 2
###############
llama_2 = [
    # https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/main/config.json
    dict(
        name="Llama-2-7b{}-hf",
        hf_config=dict(org="meta-llama", name="Llama-2-7b{}-hf"),
        vocab_size=32000,
        padding_multiple=64,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
    ),
    # https://huggingface.co/meta-llama/Llama-2-13b-hf/blob/main/config.json
    dict(
        name="Llama-2-13b{}-hf",
        hf_config=dict(org="meta-llama", name="Llama-2-13b{}-hf"),
        vocab_size=32000,
        padding_multiple=64,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
    ),
    # https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/config.json
    dict(
        name="Llama-2-70b{}-hf",
        hf_config=dict(org="meta-llama", name="Llama-2-70b{}-hf"),
        vocab_size=32000,
        padding_multiple=64,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=28672,
    ),
]
for c in llama_2:
    for kind in ("", "-chat"):
        copy = deepcopy(c)
        copy["name"] = c["name"].format(kind)
        copy["hf_config"]["name"] = c["hf_config"]["name"].format(kind)
        configs.append(copy)


##########################
# Stability AI FreeWilly2
##########################
freewilly_2 = [
    # https://huggingface.co/stabilityai/FreeWilly2/blob/main/config.json
    dict(
        name="FreeWilly2",
        hf_config=dict(org="stabilityai", name="FreeWilly2"),
        vocab_size=32000,
        padding_multiple=64,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=28672,
    )
]
configs.extend(freewilly_2)


##################
# Meta Code Llama
##################
code_llama = [
    # https://huggingface.co/codellama/CodeLlama-7b-hf/blob/main/config.json
    dict(
        name="CodeLlama-7b-hf",
        hf_config=dict(org="codellama", name="CodeLlama-7b-hf"),
        block_size=16384,
        vocab_size=32016,
        padding_multiple=16,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-13b-hf/blob/main/config.json
    dict(
        name="CodeLlama-13b-hf",
        hf_config=dict(org="codellama", name="CodeLlama-13b-hf"),
        block_size=16384,
        vocab_size=32016,
        padding_multiple=16,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-34b-hf/blob/main/config.json
    dict(
        name="CodeLlama-34b-hf",
        hf_config=dict(org="codellama", name="CodeLlama-34b-hf"),
        block_size=16384,
        vocab_size=32000,
        padded_vocab_size=32000,
        n_layer=48,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=22016,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-70b-hf/blob/main/config.json
    dict(
        name="CodeLlama-70b-hf",
        hf_config=dict(org="codellama", name="CodeLlama-70b-hf"),
        block_size=16384,
        vocab_size=32016,
        padding_multiple=16,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=28672,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-7b-Python-hf/blob/main/config.json
    dict(
        name="CodeLlama-7b-Python-hf",
        hf_config=dict(org="codellama", name="CodeLlama-7b-Python-hf"),
        block_size=16384,
        vocab_size=32000,
        padded_vocab_size=32000,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-13b-Python-hf/blob/main/config.json
    dict(
        name="CodeLlama-13b-Python-hf",
        hf_config=dict(org="codellama", name="CodeLlama-13b-Python-hf"),
        block_size=16384,
        vocab_size=32000,
        padded_vocab_size=32000,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-34b-Python-hf/blob/main/config.json
    dict(
        name="CodeLlama-34b-Python-hf",
        hf_config=dict(org="codellama", name="CodeLlama-34b-Python-hf"),
        block_size=16384,
        vocab_size=32000,
        padded_vocab_size=32000,
        n_layer=48,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=22016,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-70b-Python-hf/blob/main/config.json
    dict(
        name="CodeLlama-70b-Python-hf",
        hf_config=dict(org="codellama", name="CodeLlama-70b-Python-hf"),
        block_size=16384,
        vocab_size=32016,
        padding_multiple=16,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=28672,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf/blob/main/config.json
    dict(
        name="CodeLlama-7b-Instruct-hf",
        hf_config=dict(org="codellama", name="CodeLlama-7b-Instruct-hf"),
        block_size=16384,
        vocab_size=32016,
        padding_multiple=16,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf/blob/main/config.json
    dict(
        name="CodeLlama-13b-Instruct-hf",
        hf_config=dict(org="codellama", name="CodeLlama-13b-Instruct-hf"),
        block_size=2048,
        vocab_size=32016,
        padding_multiple=16,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf/blob/main/config.json
    dict(
        name="CodeLlama-34b-Instruct-hf",
        hf_config=dict(org="codellama", name="CodeLlama-34b-Instruct-hf"),
        block_size=16384,
        vocab_size=32000,
        padded_vocab_size=32000,
        n_layer=48,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=22016,
        rope_base=1000000,
    ),
    # https://huggingface.co/codellama/CodeLlama-70b-Instruct-hf/blob/main/config.json
    dict(
        name="CodeLlama-70b-Instruct-hf",
        hf_config=dict(org="codellama", name="CodeLlama-70b-Instruct-hf"),
        block_size=16384,
        vocab_size=32016,
        padding_multiple=16,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=28672,
        rope_base=1000000,
    ),
]
configs.extend(code_llama)


########################
# garage-bAInd Platypus
########################
platypus = [
    # https://huggingface.co/garage-bAInd/Platypus-30B/blob/main/config.json
    dict(
        name="Platypus-30B",
        hf_config=dict(org="garage-bAInd", name="Platypus-30B"),
        block_size=2048,
        padded_vocab_size=32000,
        n_layer=60,
        n_head=52,
        n_embd=6656,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-06,
        _mlp_class="LLaMAMLP",
        intermediate_size=17920,
    ),
    # https://huggingface.co/garage-bAInd/Platypus2-7B/blob/main/config.json
    dict(
        name="Platypus2-7B",
        hf_config=dict(org="garage-bAInd", name="Platypus2-7B"),
        padded_vocab_size=32000,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
    ),
    # https://huggingface.co/garage-bAInd/Platypus2-13B/blob/main/config.json
    dict(
        name="Platypus2-13B",
        hf_config=dict(org="garage-bAInd", name="Platypus2-13B"),
        padded_vocab_size=32000,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
    ),
    # https://huggingface.co/garage-bAInd/Platypus2-70B/blob/main/config.json
    dict(
        name="Platypus2-70B",
        hf_config=dict(org="garage-bAInd", name="Platypus2-70B"),
        padded_vocab_size=32000,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=28672,
    ),
    # https://huggingface.co/garage-bAInd/Camel-Platypus2-13B/blob/main/config.json
    dict(
        name="Camel-Platypus2-13B",
        hf_config=dict(org="garage-bAInd", name="Camel-Platypus2-13B"),
        padded_vocab_size=32000,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
    ),
    # https://huggingface.co/garage-bAInd/Camel-Platypus2-70B/blob/main/config.json
    dict(
        name="Camel-Platypus2-70B",
        hf_config=dict(org="garage-bAInd", name="Camel-Platypus2-70B"),
        padded_vocab_size=32000,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=28672,
    ),
    # https://huggingface.co/garage-bAInd/Stable-Platypus2-13B/blob/main/config.json
    dict(
        name="Stable-Platypus2-13B",
        hf_config=dict(org="garage-bAInd", name="Stable-Platypus2-13B"),
        padded_vocab_size=32000,
        n_layer=40,
        n_head=40,
        n_embd=5120,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=13824,
    ),
    # https://huggingface.co/garage-bAInd/Platypus2-70B-instruct/blob/main/config.json
    dict(
        name="Platypus2-70B-instruct",
        hf_config=dict(org="garage-bAInd", name="Platypus2-70B-instruct"),
        padded_vocab_size=32000,
        n_layer=80,
        n_head=64,
        n_embd=8192,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=28672,
    ),
]
configs.extend(platypus)


##########################
# Stability AI StableCode
##########################
stablecode = [
    # https://huggingface.co/stabilityai/stablecode-completion-alpha-3b/blob/main/config.json
    dict(
        name="stablecode-completion-alpha-3b",
        hf_config=dict(org="stabilityai", name="stablecode-completion-alpha-3b"),
        block_size=16384,
        vocab_size=49152,
        n_layer=32,
        n_embd=2560,
    ),
    # https://huggingface.co/stabilityai/stablecode-completion-alpha-3b-4k/blob/main/config.json
    dict(
        name="stablecode-completion-alpha-3b-4k",
        hf_config=dict(org="stabilityai", name="stablecode-completion-alpha-3b-4k"),
        vocab_size=49152,
        n_layer=32,
        n_embd=2560,
    ),
    # https://huggingface.co/stabilityai/stablecode-instruct-alpha-3b/blob/main/config.json
    dict(
        name="stablecode-instruct-alpha-3b",
        hf_config=dict(org="stabilityai", name="stablecode-instruct-alpha-3b"),
        vocab_size=49152,
        n_layer=32,
        n_embd=2560,
    ),
]
configs.extend(stablecode)


##################################
# togethercomputer LLaMA-2-7B-32K
##################################
together_llama2_32k = [
    # https://huggingface.co/togethercomputer/LLaMA-2-7B-32K/blob/main/config.json
    dict(
        name="LLaMA-2-7B-32K",
        hf_config=dict(org="togethercomputer", name="LLaMA-2-7B-32K"),
        vocab_size=32000,
        padding_multiple=64,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
        rope_condense_ratio=8,
    )
]
configs.extend(together_llama2_32k)


################
# Microsoft Phi
################
phi = [
    # https://huggingface.co/microsoft/phi-1_5/blob/main/config.json
    dict(
        name="phi-1_5",
        hf_config=dict(org="microsoft", name="phi-1_5"),
        vocab_size=50257,
        padded_vocab_size=51200,
        block_size=2048,
        n_embd=2048,
        n_layer=24,
        rotary_percentage=0.5,  # 32 / (n_embd / n_head) = 32 / 64
        shared_attention_norm=True,
        lm_head_bias=True,
        gelu_approximate="tanh",
    ),
    # https://huggingface.co/microsoft/phi-2/blob/main/config.json
    dict(
        name="phi-2",
        hf_config=dict(org="microsoft", name="phi-2"),
        vocab_size=50257,
        padded_vocab_size=51200,
        block_size=2048,
        n_embd=2560,
        n_layer=32,
        rotary_percentage=0.4,  # 32 / (n_embd / n_head) = 32 / 80
        shared_attention_norm=True,
        lm_head_bias=True,
        gelu_approximate="tanh",
    ),
]
configs.extend(phi)


#############
# Mistral AI
#############
mistral = [
    # https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json
    dict(
        name="Mistral-7B-{}v0.1",
        hf_config=dict(org="mistralai", name="Mistral-7B-{}v0.1"),
        padded_vocab_size=32000,
        block_size=4096,  # should be 32768 but sliding window attention is not implemented
        n_layer=32,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=14336,
    ),
    # https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json
    dict(
        name="Mixtral-8x7B-{}v0.1",
        hf_config=dict(org="mistralai", name="Mixtral-8x7B-{}v0.1"),
        padded_vocab_size=32000,
        block_size=32768,
        n_layer=32,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMoE",
        intermediate_size=14336,
        rope_base=1000000,
        n_expert=8,
        n_expert_per_token=2,
    ),
]
for c in mistral:
    for kind in ("", "Instruct-"):
        copy = deepcopy(c)
        copy["name"] = c["name"].format(kind)
        copy["hf_config"]["name"] = c["hf_config"]["name"].format(kind)
        configs.append(copy)
configs.append(
    # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/blob/main/config.json
    dict(
        name="Mistral-7B-Instruct-v0.2",
        hf_config=dict(org="mistralai", name="Mistral-7B-Instruct-v0.2"),
        padded_vocab_size=32000,
        block_size=32768,
        n_layer=32,
        n_query_groups=8,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-05,
        _mlp_class="LLaMAMLP",
        intermediate_size=14336,
    )
)


############
# TinyLlama
############
tiny_llama = [
    dict(
        name="tiny-llama-1.1b{}",
        hf_config=dict(org="TinyLlama", name="TinyLlama-1.1B{}"),
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=22,
        n_head=32,
        n_embd=2048,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",  # original TinyLlama uses FusedRMSNorm
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=5632,
        n_query_groups=4,
    )
]
for c in tiny_llama:
    for kind, hf_postfix in (("", "-intermediate-step-1431k-3T"), ("-chat", "-Chat-v1.0")):
        copy = deepcopy(c)
        copy["name"] = c["name"].format(kind)
        copy["hf_config"]["name"] = c["hf_config"]["name"].format(hf_postfix)
        configs.append(copy)


##########################
# Trelis Function Calling
##########################
llama_2_function_calling = [
    # https://huggingface.co/Trelis/Llama-2-7b-chat-hf-function-calling-v2/blob/main/config.json
    dict(
        name="Llama-2-7b-chat-hf-function-calling-v2",
        hf_config=dict(org="Trelis", name="Llama-2-7b-chat-hf-function-calling-v2"),
        padding_multiple=64,
        n_layer=32,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        _mlp_class="LLaMAMLP",
        intermediate_size=11008,
        norm_eps=1e-6,
        block_size=4096,
        vocab_size=32000,
        n_head=32,
        n_embd=4096,
        rope_base=10000,
    )
]

configs.extend(llama_2_function_calling)

name_to_config = {config["name"]: config for config in configs}

class GPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.max_seq_length = self.config.block_size
        self.mask_cache: Optional[torch.Tensor] = None

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        """
        When doing inference, the sequences used might be shorter than the model's context length.
        This allows setting a smaller number to avoid allocating unused memory
        """
        if value > self.config.block_size:
            raise ValueError(f"Cannot attend to {value}, block size is only {self.config.block_size}")
        self._max_seq_length = value
        if not hasattr(self, "cos"):
            # first call
            cos, sin = self.rope_cache()
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        # override
        elif value != self.cos.size(0):
            self.cos, self.sin = self.rope_cache(device=self.cos.device)
        # the mask and kv cache size will get updated on `set_kv_cache`. we cannot update it here because we don't know
        # if the kv cache is expected

    def reset_parameters(self) -> None:
        # Trigger resetting the rope-cache
        self.cos, self.sin = self.rope_cache()

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

        if input_pos is not None:  # use the kv cache
            cos = self.cos.index_select(0, input_pos)
            sin = self.sin.index_select(0, input_pos)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = self.mask_cache.index_select(2, input_pos)
        else:
            cos = self.cos[:T]
            sin = self.sin[:T]
            mask = None

        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        for block in self.transformer.h:
            x = block(x, cos, sin, mask, input_pos)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)  # (b, t, vocab_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def rope_cache(self, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return build_rope_cache(
            seq_len=self.max_seq_length,
            n_elem=self.config.rope_n_elem,
            device=device,
            condense_ratio=self.config.rope_condense_ratio,
            base=self.config.rope_base,
        )

    def set_kv_cache(
        self,
        batch_size: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if rope_cache_length is None:
            rope_cache_length = self.cos.size(-1)
        max_seq_length = self.max_seq_length

        # initialize the kv cache for all blocks
        for block in self.transformer.h:
            block.attn.kv_cache = block.attn.build_kv_cache(
                batch_size, max_seq_length, rope_cache_length, device, dtype
            )

        if self.mask_cache is None or self.mask_cache.size(3) != max_seq_length:
            # passing `attn_mask` to SDPA disables the flash implementation. since we only need the mask
            # for the kv-cache support (only during inference), we only create it in that situation
            self.mask_cache = build_mask_cache(max_seq_length, device)

    def clear_kv_cache(self) -> None:
        self.mask_cache = None
        for block in self.transformer.h:
            block.attn.kv_cache = None


class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = None if config.shared_attention_norm else config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = config.mlp_class(config)

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_1 = self.norm_1(x)
        h = self.attn(n_1, cos, sin, mask, input_pos)
        if self.config.parallel_residual:
            n_2 = n_1 if self.config.shared_attention_norm else self.norm_2(x)
            x = self.mlp(n_2) + h + x
        else:
            if self.config.shared_attention_norm:
                raise NotImplementedError(
                    "No checkpoint amongst the ones we support uses this configuration"
                    " (non-parallel residual and shared attention norm)."
                )
            x = h + x
            x = self.mlp(self.norm_2(x)) + x
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # disabled by default
        self.kv_cache: Optional[KVCache] = None

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.attn(x)

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size)
        qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)

        # maybe repeat k and v if for the non multi-head attention cases
        # training: flash attention requires it
        # inference: multi-query would require a full kv cache so avoid it to limit its memory usage
        if self.config.n_query_groups != self.config.n_head and (input_pos is None or self.config.n_query_groups != 1):
            k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
            v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)

        q = q.reshape(B, -1, T, self.config.head_size)  # (B, nh_q, T, hs)
        k = k.reshape(B, -1, T, self.config.head_size)  # (B, nh_k, T, hs)
        v = v.reshape(B, -1, T, self.config.head_size)  # (B, nh_v, T, hs)

        q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos, sin)
        k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
        k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)

        if input_pos is not None:
            if not isinstance(self.kv_cache, KVCache):
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            k, v = self.kv_cache(input_pos, k, v)

        y = self.scaled_dot_product_attention(q, k, v, mask)

        y = y.reshape(B, T, self.config.n_embd)  # re-assemble all head outputs side by side

        # output projection
        return self.proj(y)

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.config.head_size)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
        )
        return y.transpose(1, 2)

    def build_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "KVCache":
        heads = 1 if self.config.n_query_groups == 1 else self.config.n_head
        v_shape = (batch_size, heads, max_seq_length, self.config.head_size)
        if rope_cache_length is None:
            if self.config.rotary_percentage != 1.0:
                raise TypeError("Please pass the `rope_cache_length=gpt.cos.size(-1)` value")
            k_shape = v_shape
        else:
            k_shape = (
                batch_size,
                heads,
                max_seq_length,
                rope_cache_length + self.config.head_size - self.config.rope_n_elem,
            )
        return KVCache(k_shape, v_shape, device=device, dtype=dtype)


class GptNeoxMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.gelu(x, approximate=self.config.gelu_approximate)
        return self.proj(x)


class LLaMAMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.fc_2 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        return self.proj(x)


class LLaMAMoE(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.gate = nn.Linear(config.n_embd, config.n_expert, bias=False)
        self.experts = nn.ModuleList(LLaMAMLP(config) for _ in range(config.n_expert))

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Derived from: https://github.com/mistralai/mistral-src/blob/b46d6/moe_one_file_ref.py#L203-L219
        See also figure 1 in https://arxiv.org/abs/2211.15841
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        x = x.view(-1, C)  # (B*T, C)
        router = self.gate(x)  # (B*T, n_expert)
        probs, indices = torch.topk(router, self.config.n_expert_per_token)  # (B*T, n_expert_per_token)
        probs = probs.softmax(dim=1, dtype=torch.float).to(dtype=x.dtype)
        masks = indices.unsqueeze(-1) == torch.arange(self.config.n_expert, device=x.device)
        masks = masks.permute(2, 0, 1)  # (n_expert, B*T, n_expert_per_token)
        y = torch.zeros_like(x)  # (B*T, C)
        for mask, expert in zip(masks, self.experts):
            token_idx, expert_idx = torch.where(mask)
            y[token_idx] += probs[token_idx, expert_idx, None] * expert(x[token_idx])
        return y.view(B, T, C)


def build_rope_cache(
    seq_len: int, n_elem: int, device: Optional[torch.device] = None, base: int = 10000, condense_ratio: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)

    return torch.cos(idx_theta), torch.sin(idx_theta)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    roped = (x * cos) + (rotated * sin)
    return roped.to(dtype=x.dtype)


class KVCache(nn.Module):
    def __init__(
        self,
        k_shape: Tuple[int, int, int, int],
        v_shape: Tuple[int, int, int, int],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.register_buffer("k", torch.zeros(k_shape, device=device, dtype=dtype), persistent=False)
        self.register_buffer("v", torch.zeros(v_shape, device=device, dtype=dtype), persistent=False)

    def forward(self, input_pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # move the buffer to the activation dtype for when AMP is used
        self.k = self.k.to(k.dtype)
        self.v = self.v.to(v.dtype)
        # update the cache
        k = self.k.index_copy_(2, input_pos, k)
        v = self.v.index_copy_(2, input_pos, v)
        return k, v

    def reset_parameters(self) -> None:
        torch.nn.init.zeros_(self.k)
        torch.nn.init.zeros_(self.v)


def build_mask_cache(max_seq_length: int, device: Optional[torch.device] = None) -> torch.Tensor:
    ones = torch.ones((max_seq_length, max_seq_length), device=device, dtype=torch.bool)
    return torch.tril(ones).unsqueeze(0).unsqueeze(0)