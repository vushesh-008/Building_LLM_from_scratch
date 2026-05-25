import re
from torch.utils.data import Dataset, DataLoader
import tiktoken

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


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):

        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+1+max_length]

            self.input_ids.append(input_chunk)
            self.target_ids.append(target_chunk)

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):

    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader


if __name__ == "__main__":
    
    with open("../data/verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("Raw text sample: ", raw_text[:200])
    print("Number of characters in raw text: ", len(raw_text))

    dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
    data_iter = iter(dataloader)
    input_ids, target_ids = next(data_iter)
    print(f"Input IDs: {input_ids} -> Target IDs: {target_ids}")

    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
    data_iter = iter(dataloader)
    inputs , targets = next(data_iter)
    print("Batch of input IDs:\n", inputs)
    print("Batch of target IDs:\n", targets)

