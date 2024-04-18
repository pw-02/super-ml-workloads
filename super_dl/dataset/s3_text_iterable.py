import functools
import super_dl.s3utils as s3utils
from  super_dl.s3utils import S3Url
from typing import List, Tuple, Dict
from torch.utils.data import SequentialSampler, IterableDataset, RandomSampler, DataLoader
import torchvision
import torch
import torch.nn.functional as F
import tiktoken
import time
import re
import unicodedata
import nltk
import random
from nltk.corpus import wordnet
nltk.download('wordnet')
nltk.download('stopwords')
UNICODE_PUNCT = {
    "，": ",",
    "。": ".",
    "、": ",",
    "„": '"',
    "”": '"',
    "“": '"',
    "«": '"',
    "»": '"',
    "１": '"',
    "」": '"',
    "「": '"',
    "《": '"',
    "》": '"',
    "´": "'",
    "∶": ":",
    "：": ":",
    "？": "?",
    "！": "!",
    "（": "(",
    "）": ")",
    "；": ";",
    "–": "-",
    "—": " - ",
    "．": ". ",
    "～": "~",
    "’": "'",
    "…": "...",
    "━": "-",
    "〈": "<",
    "〉": ">",
    "【": "[",
    "】": "]",
    "％": "%",
    "►": "-",
}
UNICODE_PUNCT_RE = re.compile(f"[{''.join(UNICODE_PUNCT.keys())}]")
# Build a regex matching all control characters.
NON_PRINTING_CHARS_RE = re.compile(
    f"[{''.join(map(chr, list(range(0,32)) + list(range(127,160))))}]"
)
DIGIT_RE = re.compile(r"\d")
PUNCT_OR_NON_PRINTING_CHARS_RE = re.compile(
    (UNICODE_PUNCT_RE.pattern + NON_PRINTING_CHARS_RE.pattern).replace("][", "")
)



class S3TextIterableDataset(IterableDataset):
    def __init__(self,data_dir:str, tokenizer, block_size:int, shuffle = False):
        super().__init__()
        self.epoch = 0
        self.block_size = block_size
        self.shuffle_urls = shuffle
        # if dataset_kind == 'image':
        self.samples:List[str] = s3utils.load_unpaired_s3_object_keys(data_dir, False, True)
        self.bucket_name = S3Url(data_dir).bucket
        self.tokenizer = tokenizer

    def normalize(self, line: str, accent=True, case=True, numbers=True, punct=1) -> str:
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
    
    def remove_stop_words(self, text):
        tokens = text.split()
        from nltk.corpus import stopwords
        stopwords_set = set(stopwords.words('english'))
        filtered_tokens  = [word for word in tokens if word not in stopwords_set]
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text

    def remove_rare_words(self, text):
        from collections import Counter
        tokens = text.split()
        # Count the frequency of each token
        word_counts = Counter(tokens)
        # Set a threshold for word frequency
        threshold = 2
        # Remove tokens with frequency less than or equal to the threshold
        filtered_tokens = [word for word in tokens if word_counts[word] > threshold]
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text
        # Function to perform synonym replacement augmentation
    
    def replace_with_synonym(self,word):
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        return random.choice(synonyms) if synonyms else word
    
    def synonym_replacement(self, text, num_replacements=1):
        words = text.split()
        for _ in range(num_replacements):
            idx = random.randint(0, len(words) - 1)
            words[idx] = self.replace_with_synonym(words[idx])
        return ' '.join(words)

    
    def remove_pii(self, text):
        # Regular expressions for various PII types
        email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_regex = r'\b(?:\+\d{1,2}\s*)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b'
        ssn_regex = r'\b\d{3}-\d{2}-\d{4}\b'
        credit_card_regex = r'\b(?:\d[ -]*?){13,16}\b'
        address_regex = r'\b\d{1,5}\s+\w+\s+\w+\b'

        # Replace PII with placeholders
        text = re.sub(email_regex, '[EMAIL]', text)
        text = re.sub(phone_regex, '[PHONE]', text)
        text = re.sub(ssn_regex, '[SSN]', text)
        text = re.sub(credit_card_regex, '[CREDIT_CARD]', text)
        text = re.sub(address_regex, '[ADDRESS]', text)
        return text
    
    # Function to perform random insertion augmentation
    def random_insertion(self, text, num_insertions=1):
        words = text.split()
        for _ in range(num_insertions):
            words.insert(random.randint(0, len(words)), random.choice(words))
        return ' '.join(words)
    
    # Function to perform word swapping augmentation
    def word_swapping(self, text, num_swaps=1):
        words = text.split()
        for _ in range(num_swaps):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        return ' '.join(words)


    def remove_non_printing_char(self, text: str) -> str:
        return NON_PRINTING_CHARS_RE.sub("", text)

    def __len__(self):
        return len(self.samples)
    
    def remove_html_tags(self,text):
        import re
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)
    
    def remove_unicode_punct(self, text: str) -> str:
        """More aggressive version of replace_unicode_punct but also faster."""
        return UNICODE_PUNCT_RE.sub("", text)
    
    def replace_unicode_punct(self, text: str) -> str:
        return "".join((UNICODE_PUNCT.get(c, c) for c in text))
    
    def strip_accents(self, line: str) -> str:
        """Strips accents from a piece of text."""
        nfd = unicodedata.normalize("NFD", line)
        output = [c for c in nfd if unicodedata.category(c) != "Mn"]
        if len(output) == line:
            return line
        return "".join(output)
    
    def normalize_spacing_for_tok(self, text: str, language: str = "en") -> str:
        res = (
            text.replace("\r", "")
            # remove extra spaces
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
            # French quotes
            .replace(" « ", ' "')
            .replace("« ", '"')
            .replace("«", '"')
            .replace(" » ", '" ')
            .replace(" »", '"')
            .replace("»", '"')
            # handle pseudo-spaces
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
        # English "quotation," followed by comma, style
        if language == "en":
            res = re.sub(r"\"([,\.]+)", r"\1\"", res)
        # Czech is confused
        elif language == "cs" or language == "cz":
            pass
        # German/Spanish/French "quotation", followed by comma, style
        else:
            res = res.replace(',"', '",')
            res = re.sub(
                r"(\.+)\"(\s*[^<])", r"\"\1\2", res
            )  # don't fix period at end of sentence

        if (
            language == "de"
            or language == "es"
            or language == "cz"
            or language == "cs"
            or language == "fr"
        ):
            res = re.sub(r"(\d) (\d)", r"\1,\2", res)
        else:
            res = re.sub(r"(\d) (\d)", r"\1.\2", res)
        return res
    
    def tokenize(self, text):
        ids = self.tokenizer.encode_ordinary(text) # encode_ordinary ignores any special tokens
        #ids.append(self.tokenizer.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        # print(f"tokens: {len(ids)}")

        # Tokenize text into chunks of block_size
        chunks = []
        start_idx = 0
        while start_idx < len(ids):
            end_idx = min(start_idx + self.block_size, len(ids))
            x = torch.tensor(ids[start_idx:end_idx], dtype=torch.long)
            y = torch.tensor(ids[start_idx+1:end_idx+1], dtype=torch.long)
            if len(x) < self.block_size:
                # print(len(ids) + (self.block_size - len(x)))
                x = F.pad(x, (0, self.block_size - len(x)))
            if len(y) < self.block_size:
                y = F.pad(y, (0, self.block_size - len(y)))
         
            chunks.append((x, y))
            start_idx = end_idx

        return chunks
    

    def __iter__(self):

        if self.shuffle_urls:
            sampler = RandomSampler(self)
        else:
            sampler = SequentialSampler(self)
        
        for idx in sampler:
            file_path = self.samples[idx]
            sample_input = s3utils.get_s3_object(self.bucket_name, file_path)
            normalized_input = self.normalize(sample_input)

            tokenized_chunks = self.tokenize(normalized_input)
            for x, y in tokenized_chunks:
                yield x, y

    
    def set_epoch(self, epoch):
        self.epoch = epoch


if __name__ == "__main__":
    def get_batch_size_mb(batch_tensor):
        import sys
        # Get the size of the tensor in bytes
        size_bytes = sys.getsizeof(batch_tensor.storage()) + sys.getsizeof(batch_tensor)
        # Convert bytes to megabytes
        size_mb = size_bytes / (1024 ** 2)
        # Convert bytes to kb
        size_in_kb = size_bytes / 1024
        return size_mb,size_in_kb

    # # Example usage
    train_data_dir = 's3://openwebtxt/owt/train/'
    block_size = 2048

    dataset = S3TextIterableDataset(data_dir=train_data_dir,
                                    tokenizer=tiktoken.get_encoding("gpt2"),
                                    block_size=2048,
                                    shuffle=True)

    data_loader = DataLoader(dataset, batch_size=12)
    # Get the size of the tensor using pympler
    end = time.perf_counter()
    for input, target in data_loader:
        print(f"Time to preprocess: {time.perf_counter() - end:.02f} seconds")
        batch_size_mb,size_in_kb  = get_batch_size_mb(input)
        print(f"Batch size: {batch_size_mb:.2f} MB, {size_in_kb:.2f} KB")
        print(input.shape, target.shape)
        end = time.perf_counter()