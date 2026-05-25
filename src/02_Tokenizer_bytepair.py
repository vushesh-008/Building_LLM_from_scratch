from importlib.metadata import version
# pyrefly: ignore [missing-import]
import tiktoken
print(version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace. <|endoftext|> "
ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(ids)

strings = tokenizer.decode(ids)
print(strings)

text2 = "Akwirw ier"
ids2 = tokenizer.encode(text2)
print(ids2)

strings2 = tokenizer.decode(ids2)
print(strings2)



with open("../data/verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

encoded_text = tokenizer.encode(raw_text)
print(len(encoded_text))

enc_sample = encoded_text[:50]
print(enc_sample)

print(tokenizer.decode(enc_sample))

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]

print(x, "->", y)

for i in range(1,context_size+ 1):
    context = enc_sample[:i]
    target = enc_sample[i]
    print(f"prompt: {context} -> target: {target} , prompt: {tokenizer.decode(context)} -> target: {tokenizer.decode([target])}")


