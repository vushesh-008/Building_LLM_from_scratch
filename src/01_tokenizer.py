import re

class simpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text:str) -> list[int]:
        preprocessed = re.split(r'([,.:;?_"()\']|--|\s)' , text )
        preprocessed = [token.strip() for token in preprocessed if token.strip()]
        preprocessed = [token if token in self.str_to_int else "[UNK]" for token in preprocessed]
        ids = [self.str_to_int[token] for token in preprocessed]
        return ids

    def decode(self, ids: list[int]) -> str:
        text = "".join([self.int_to_str[id] for id in ids])

        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


with open("../data/verdict.txt", "r" , encoding="utf-8") as f:
    raw_text = f.read()
preprocessed = re.split(r'([,.:;?_"()\']|--|\s)' , raw_text )
preprocessed = [item.strip() for item in preprocessed if item.strip()]

# Step-2: Build vocabulary
all_words = sorted(list(set(preprocessed)))
all_words.extend(["[UNK]","<|endoftext|>"])
vocab = {token:integer for integer,token in enumerate(all_words)}
vocab_size = len(vocab)
print("Vocabulary size: ", vocab_size)

tokenizer = simpleTokenizer(vocab)
text = """It's the last he painted, you know,"
          Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))
    
for i , item in enumerate(list(vocab.items())[-5:]):
    print(i, item)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join([text1, text2])
print(text)
print(tokenizer.encode(text))

tokenizer = simpleTokenizer(vocab)
print(tokenizer.encode(text))