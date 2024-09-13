from dataloading.super.super_text_iterable_dataset import SUPERTextDataset
from litgpt.tokenizer import Tokenizer
import torch
import json
from pathlib import Path
from typing import Optional, Union, Iterable, Iterator
from torch.utils.data import DataLoader

class Tokenizer:
    def __init__(self, checkpoint_dir: Union[Path, str]) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise NotADirectoryError(f"The checkpoint directory does not exist: {str(checkpoint_dir)}")

        self.model_name = checkpoint_dir.stem
        self.use_bos = self.check_if_bos_token_used(checkpoint_dir)
        self.bos_id = None
        self.eos_id = None

        # some checkpoints have both files, `.json` takes precedence
        if (vocabulary_path := checkpoint_dir / "tokenizer.json").is_file():
            from tokenizers import Tokenizer as HFTokenizer

            self.processor = HFTokenizer.from_file(str(vocabulary_path))
            self.backend = "huggingface"

            if (special_tokens_path := checkpoint_dir / "tokenizer_config.json").is_file():
                with open(special_tokens_path, encoding="utf-8") as fp:
                    config = json.load(fp)
                bos_token = config.get("bos_token")
                eos_token = config.get("eos_token")
                if bos_token is not None and isinstance(bos_token, dict):
                    bos_token = bos_token.get("content")
                if eos_token is not None and isinstance(eos_token, dict):
                    eos_token = eos_token.get("content")
                self.bos_id = self.token_to_id(bos_token) if bos_token is not None else None
                self.eos_id = self.token_to_id(eos_token) if eos_token is not None else None
            if (special_tokens_path := checkpoint_dir / "generation_config.json").is_file():
                with open(special_tokens_path, encoding="utf-8") as fp:
                    config = json.load(fp)
                if self.bos_id is None:
                    self.bos_id = config.get("bos_token_id")
                if self.eos_id is None:
                    self.eos_id = config.get("eos_token_id")

        elif (vocabulary_path := checkpoint_dir / "tokenizer.model").is_file():
            from sentencepiece import SentencePieceProcessor

            self.processor = SentencePieceProcessor(model_file=str(vocabulary_path))
            self.backend = "sentencepiece"
            self.bos_id = self.processor.bos_id()
            self.eos_id = self.processor.eos_id()
        else:
            raise NotImplementedError

        # NOTE: A temporary fix until it's resolved on Tokenizers side.
        # LlaMA tokenizer strips leading spaces if to decode a single token at a time.
        # https://github.com/huggingface/transformers/issues/31643
        self.apply_decoding_fix = None
        if (config_path := checkpoint_dir / "tokenizer_config.json").is_file():
            with open(config_path, encoding="utf-8") as fp:
                self.apply_decoding_fix = "LlamaTokenizer" in json.load(fp)["tokenizer_class"]

    @property
    def vocab_size(self) -> int:
        if self.backend == "huggingface":
            return self.processor.get_vocab_size(with_added_tokens=False)
        if self.backend == "sentencepiece":
            return self.processor.vocab_size()
        raise RuntimeError

    def token_to_id(self, token: str) -> int:
        if self.backend == "huggingface":
            id_ = self.processor.token_to_id(token)
        elif self.backend == "sentencepiece":
            id_ = self.processor.piece_to_id(token)
        else:
            raise RuntimeError
        if id_ is None:
            raise ValueError(f"token {token!r} not found in the collection.")
        return id_

    def check_if_bos_token_used(self, checkpoint_dir: Path) -> bool:
        if not (tokenizer_config_path := checkpoint_dir / "tokenizer_config.json").is_file():
            return False
        with open(tokenizer_config_path, encoding="utf-8") as fp:
            config = json.load(fp)
        # for LlaMA-3 tokenizer there is no `add_bos_token` at all and `tokenizer_class` is only
        # `PreTrainedTokenizerFast`
        if checkpoint_dir.stem.startswith("Meta-Llama-3"):
            return True
        if "add_bos_token" in config:
            return config["add_bos_token"]
        # if `add_bos_token` isn't in the config file, but LLaMA tokenizer is used - return True.
        # ex: https://huggingface.co/stabilityai/StableBeluga2/blob/main/tokenizer_config.json#L2
        return config.get("tokenizer_class") == "LlamaTokenizer"

    def encode(
        self,
        string: str,
        device: Optional[torch.device] = None,
        bos: Optional[bool] = None,
        eos: bool = False,
        max_length: int = -1,
    ) -> torch.Tensor:
        if self.backend == "huggingface":
            tokens = self.processor.encode(string).ids
        elif self.backend == "sentencepiece":
            tokens = self.processor.encode(string)
        else:
            raise RuntimeError(f"`{self.backend}` is not supported.")
        if tokens is None:
            raise ValueError("`self.processor` returned tokens of None value.")

        if bos or (bos is None and self.use_bos):
            if self.bos_id is None:
                raise NotImplementedError("This tokenizer does not have a defined bos token.")
            if not tokens or tokens[0] != self.bos_id:
                tokens = [self.bos_id] + tokens
        # if the processor misbehaves and adds `bos` token no matter what
        elif tokens and tokens[0] == self.bos_id:
            tokens = tokens[1:]

        if eos and (not tokens or tokens[-1] != self.eos_id):
            tokens = tokens + [self.eos_id]

        if max_length > 0:
            tokens = tokens[:max_length]
        return torch.tensor(tokens, dtype=torch.int, device=device)

    def decode(self, tensor: torch.Tensor) -> str:
        tokens = [tensor.item()] if tensor.ndim == 0 else tensor.tolist()
        if len(tokens) == 1 and self.apply_decoding_fix:
            dummy_token_id = 33  # \x1e
            dummy_token = self.processor.decode([dummy_token_id])
            return self.processor.decode([dummy_token_id] + tokens)[len(dummy_token) :]
        return self.processor.decode(tokens)

    

# Example Usage
if __name__ == "__main__":

    data_dir = 's3://owt-5mb-text-chunks/train'  # Adjust the path to your .txt files
    tokenizer = Tokenizer('mlworkloads\checkpoints\EleutherAI\pythia-14m')
    block_size = 512
    batch_size = 4

    dataset = SUPERTextDataset(
        job_id=1,
        s3_data_dir=data_dir,
        tokenizer=tokenizer,
        transform=None,
        block_size=block_size,
        batch_size=batch_size,
        grpc_server_address="localhost:50051",
        world_size=1,
        cache_address=None,
        simulate_delay=None)
        
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    for  batch, data_loading_time, transformation_time, cache_hit, cached_after_fetc in dataloader:
        input_ids, targets = batch
        print(f"Input IDs: {input_ids.shape}, Targets: {targets.shape}")



        # break
