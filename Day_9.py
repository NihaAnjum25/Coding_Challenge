# Import required libraries
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase
from tokenizers.processors import TemplateProcessing

# Sample dataset

text_data = [
    "Tokenization is an important step in Natural Language Processing.",
    "Different models use different tokenization techniques.",
    "Understanding BPE and WordPiece helps in NLP tasks."
]

sample_text = "Tokenization helps models understand text."

print("Sample Text:")
print(sample_text)
print("\n")

# 1. Whitespace Tokenization

print("Whitespace Tokenization")

whitespace_tokens = sample_text.split()

print(whitespace_tokens)
print("\n")

# 2. BPE Tokenization

print("BPE Tokenization")

bpe_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
bpe_tokenizer.pre_tokenizer = Whitespace()
bpe_tokenizer.normalizer = Lowercase()

trainer = BpeTrainer(vocab_size=50, special_tokens=["[UNK]"])

bpe_tokenizer.train_from_iterator(text_data, trainer)

bpe_output = bpe_tokenizer.encode(sample_text)

print(bpe_output.tokens)
print("\n")

# 3. WordPiece Tokenization

print("WordPiece Tokenization")

wp_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
wp_tokenizer.pre_tokenizer = Whitespace()
wp_tokenizer.normalizer = Lowercase()

trainer = WordPieceTrainer(vocab_size=50, special_tokens=["[UNK]"])

wp_tokenizer.train_from_iterator(text_data, trainer)

wp_output = wp_tokenizer.encode(sample_text)

print(wp_output.tokens)