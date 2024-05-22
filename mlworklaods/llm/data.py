import os
import re
import glob
import random
import unicodedata
from collections import Counter
from typing import List, Callable, Tuple, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer, GPT2Tokenizer

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize

from mlworklaods.s3utils import S3Url
from mlworklaods.dataloaders.torch_lru.torch_lru_text_dataset import TorchLRUTextDataset
from mlworklaods.args import LLMTrainArgs, DataArgs, SUPERArgs, LRUTorchArgs

# Check and download NLTK data if not already available
def check_and_download_nltk_resource(resource_name: str):
    try:
        nltk.data.find(resource_name)
    except LookupError:
        nltk.download(resource_name)

# Regex patterns
UNICODE_PUNCT = {
    "，": ",", "。": ".", "、": ",", "„": '"', "”": '"', "“": '"', "«": '"', "»": '"',
    "１": '"', "」": '"', "「": '"', "《": '"', "》": '"', "´": "'", "∶": ":", "：": ":",
    "？": "?", "！": "!", "（": "(", "）": ")", "；": ";", "–": "-", "—": " - ", "．": ". ",
    "～": "~", "’": "'", "…": "...", "━": "-", "〈": "<", "〉": ">", "【": "[", "】": "]",
    "％": "%", "►": "-"
}
UNICODE_PUNCT_RE = re.compile(f"[{''.join(UNICODE_PUNCT.keys())}]")
NON_PRINTING_CHARS_RE = re.compile(f"[{''.join(map(chr, list(range(0,32)) + list(range(127,160))))}]")
DIGIT_RE = re.compile(r"\d")
PUNCT_OR_NON_PRINTING_CHARS_RE = re.compile((UNICODE_PUNCT_RE.pattern + NON_PRINTING_CHARS_RE.pattern).replace("][", ""))

class TextTransformations:
    """Class for text normalization and augmentation transformations."""

    def normalize(self, line: str, accent: bool = True, case: bool = True, numbers: bool = True, punct: int = 1) -> str:
        """Normalize a line of text."""
        line = line.strip()
        if not line:
            return line
        if case:
            line = line.lower()
        if accent:
            line = self.strip_accents(line)
        if numbers:
            line = DIGIT_RE.sub("0", line)
        if punct == 1:
            line = self.replace_unicode_punct(line)
        line = self.remove_non_printing_char(line)
        line = self.remove_pii(line)
        line = self.normalize_spacing_for_tok(line)
        line = self.remove_stop_words(line)
        line = self.remove_rare_words(line)
        line = self.random_insertion(line)
        line = self.word_swapping(line)
        line = self.synonym_replacement(line)
        return line

    def remove_stop_words(self, text: str) -> str:
        """Remove stop words from text."""
        tokens = text.split()
        stopwords_set = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stopwords_set]
        return ' '.join(filtered_tokens)

    def remove_rare_words(self, text: str) -> str:
        """Remove rare words from text."""
        tokens = text.split()
        word_counts = Counter(tokens)
        threshold = 2
        filtered_tokens = [word for word in tokens if word_counts[word] > threshold]
        return ' '.join(filtered_tokens)

    def replace_with_synonym(self, word: str) -> str:
        """Replace a word with its synonym."""
        synonyms = [lemma.name() for syn in wordnet.synsets(word) for lemma in syn.lemmas()]
        return random.choice(synonyms) if synonyms else word

    def synonym_replacement(self, text: str, num_replacements: int = 1) -> str:
        """Replace words in text with their synonyms."""
        words = text.split()
        for _ in range(num_replacements):
            idx = random.randint(0, len(words) - 1)
            words[idx] = self.replace_with_synonym(words[idx])
        return ' '.join(words)

    def remove_pii(self, text: str) -> str:
        """Remove personally identifiable information (PII) from text."""
        pii_patterns = [
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
            (r'\b(?:\+\d{1,2}\s*)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]'),
            (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),
            (r'\b(?:\d[ -]*?){13,16}\b', '[CREDIT_CARD]'),
            (r'\b\d{1,5}\s+\w+\s+\w+\b', '[ADDRESS]')
        ]
        for pattern, replacement in pii_patterns:
            text = re.sub(pattern, replacement, text)
        return text

    def random_insertion(self, text: str, num_insertions: int = 1) -> str:
        """Insert random words into the text."""
        words = text.split()
        for _ in range(num_insertions):
            words.insert(random.randint(0, len(words)), random.choice(words))
        return ' '.join(words)

    def word_swapping(self, text: str, num_swaps: int = 1) -> str:
        """Swap random words in the text."""
        words = text.split()
        for _ in range(num_swaps):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        return ' '.join(words)

    def remove_non_printing_char(self, text: str) -> str:
        """Remove non-printing characters from text."""
        return NON_PRINTING_CHARS_RE.sub("", text)

    def remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text."""
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)

    def remove_unicode_punct(self, text: str) -> str:
        """Remove Unicode punctuation from text."""
        return UNICODE_PUNCT_RE.sub("", text)

    def replace_unicode_punct(self, text: str) -> str:
        """Replace Unicode punctuation in text."""
        return "".join(UNICODE_PUNCT.get(c, c) for c in text)

    def strip_accents(self, line: str) -> str:
        """Strip accents from characters in the text."""
        nfd = unicodedata.normalize("NFD", line)
        return "".join(c for c in nfd if unicodedata.category(c) != "Mn")

    def normalize_spacing_for_tok(self, text: str, language: str = "en") -> str:
        """Normalize spacing for tokenization."""
        res = (
            text.replace("\r", "")
            .replace("(", " (")
            .replace(")", ") ")
            .replace(" +", " ")
        )
        res = re.sub(r"\) ([\.\!\:\?\;\,])", r"\)\1", res)
        res = res.replace("( ", "(").replace(" )", ")")
        res = re.sub(r"(\d) \%", r"\1\%", res)
        res = res.replace(" :", ":").replace(" ;", ";")
        res = res.replace("`", "'").replace("''", ' " ')
        res = (
            res.replace("„", '"')
            .replace("“", '"')
            .replace("”", '"')
            .replace("–", "-")
            .replace("—", " - ")
            .replace(" +", " ")
            .replace("´", "'")
            .replace("([a-z])‘([a-z])", r"\1'\2/")
            .replace("([a-z])’([a-z])", r"\1'\2/")
            .replace("‘", '"')
            .replace("‚", '"')
            .replace("’", '"')
            .replace("''", '"')
            .replace("´´", '"')
            .replace("…", "...")
            .replace(" « ", ' "')
            .replace("« ", '"')
            .replace("«", '"')
            .replace(" » ", '" ')
            .replace(" »", '"')
            .replace("»", '"')
            .replace(" %", "%")
            .replace("nº ", "nº ")
            .replace(" :", ":")
            .replace(" ºC", " ºC")
            .replace(" cm", " cm")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ;", ";")
            .replace(", ", ", ")
            .replace(" +", " ")
            .replace("．", ". ")
        )
        if language == "en":
            res = re.sub(r"\"([,\.]+)", r"\1\"", res)
        else:
            res = res.replace(',"', '",')
            res = re.sub(r"(\.+)\"(\s*[^<])", r"\"\1\2", res)
        if language in {"de", "es", "cz", "cs", "fr"}:
            res = re.sub(r"(\d) (\d)", r"\1,\2", res)
        else:
            res = re.sub(r"(\d) (\d)", r"\1.\2", res)
        return res

class BaseDataModule:
    """Base class for data modules."""
    
    def __init__(self, transform: Callable, tokenizer: PreTrainedTokenizer):
        self.transform = transform
        self.tokenizer = tokenizer

    def make_dataloaders(self, train_args: LLMTrainArgs, data_args: DataArgs, dataloader_args, model_max_seq_length: int) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        if isinstance(dataloader_args, SUPERArgs):
            pass
            # return self.make_super_dataloaders(train_args, data_args, dataloader_args, world_size)
        elif isinstance(dataloader_args, LRUTorchArgs):
            return self.make_lru_torch_dataloaders(train_args, data_args, dataloader_args, model_max_seq_length)
        else:
            raise Exception(f"Unknown dataloader_kind {train_args.dataloader_kind}")

    def make_lru_torch_dataloaders(self, train_args: LLMTrainArgs, data_args: DataArgs, lru_torch_args: LRUTorchArgs, model_max_seq_length: int) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        train_dataloader = None
        val_dataloader = None
        
        if train_args.run_training:
            train_dataset = TorchLRUTextDataset(
                data_dir=data_args.train_data_dir,
                tokenizer=self.tokenizer,
                transform=self.transform,
                block_size=model_max_seq_length,
                batch_size=train_args.micro_batch_size
            )
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=None, num_workers=lru_torch_args.num_pytorch_workers)
        
        if train_args.run_evaluation:
            val_dataset = TorchLRUTextDataset(
                data_dir=data_args.val_data_dir,
                tokenizer=self.tokenizer,
                transform=self.transform,
                block_size=model_max_seq_length,
                batch_size=train_args.micro_batch_size
            )
            val_dataloader = DataLoader(dataset=val_dataset, batch_size=None, num_workers=lru_torch_args.num_pytorch_workers)
        
        return train_dataloader, val_dataloader

class OpenWebTextDataModule(BaseDataModule):

    
    def __init__(self):
        """Data module for OpenWebText dataset."""
        check_and_download_nltk_resource('stopwords')
        check_and_download_nltk_resource('wordnet')
        text_transformer = TextTransformations()
        transform_func = text_transformer.normalize
        super().__init__(transform=transform_func, tokenizer=GPT2Tokenizer.from_pretrained('gpt2'))
