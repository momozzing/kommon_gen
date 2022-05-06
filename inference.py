"""
python interactive.py
"""
from argparse import ArgumentParser
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer
import pandas as pd
from tqdm import tqdm


model_name = "skt/kogpt2-base-v2"
ckpt = 'skt-kogpt2-base-v2-5.pt'
ckpt_name = f"model_save/{ckpt}"

model = AutoModelWithLMHead.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

SPECIAL_TOKENS = {
    "bos_token": "[bos]",
    "eos_token": "[eos]",
    "pad_token": "[pad]",
    "sep_token": "[sep]"
    }

tokenizer.add_special_tokens(SPECIAL_TOKENS)

model.resize_token_embeddings(len(tokenizer)) 


model.load_state_dict(torch.load(ckpt_name, map_location="cpu"))
model.cuda()


test_data = pd.read_csv("data/test_data.csv", delimiter="\t")
# test_data = test_data[:5]
test_text, test_labels = (
    test_data["concept_set"].values,
    test_data["label"].values,
)

dataset = [
    {"data": t , "label": l }
    for t, l in zip(test_text, test_labels)
]


gen_list = []

for data in tqdm(dataset):
    text, label = data["data"], data["label"]
    tokens = tokenizer(
        text + '[sep]',
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=30
    )

    input_ids = tokens.input_ids.cuda()
    attention_mask = tokens.attention_mask.cuda()

    sample_output = model.generate(
        input_ids, 
        max_length=30, 
        min_length=10,
        num_beams=10, 
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    gen = sample_output[0]
    gen_text = str(tokenizer.decode(gen[len(input_ids[0]):-1], skip_special_tokens=True)).split('.')[0]
    gen_list.append(gen_text)


gen_df = pd.DataFrame(gen_list, columns = ['gen'])
gen_df = pd.concat([gen_df, test_data["concept_set"]], axis = 1)
gen_df.to_csv(f'result/gen_text_{ckpt}.csv', sep='\t')