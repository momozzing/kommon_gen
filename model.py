'''
deepspeed --num_gpus=1 model.py
'''

from argparse import ArgumentParser
import os
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelWithLMHead
from torch.optim import AdamW
import deepspeed

import wandb

#############################################    -> 실험결과 FIX
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
##################################

os.environ["TOKENIZERS_PARALLELISM"] = "true"

model_name = "skt/kogpt2-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
SPECIAL_TOKENS = {
    "bos_token": "[bos]",
    "eos_token": "[eos]",
    "pad_token": "[pad]",
    "sep_token": "[sep]"
    }
tokenizer.add_special_tokens(SPECIAL_TOKENS)

model = AutoModelWithLMHead.from_pretrained(
    model_name
).cuda()

model.resize_token_embeddings(len(tokenizer)) 

parser = ArgumentParser()
parser.add_argument("--deepspeed_config", type=str, default="ds_config.json")
parser.add_argument("--local_rank", type=int)
parser.add_argument("--epoch", default=10, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--sep_token", default=tokenizer.sep_token, type=str)
parser.add_argument("--bos_token", default=tokenizer.bos_token, type=str)
parser.add_argument("--eos_token", default=tokenizer.eos_token, type=str)
args = parser.parse_args()

wandb.init(project="kommongen", name=f"kommongen-{model_name}")
train_data = pd.read_csv("data/train_data.csv", delimiter="\t")
train_text, train_labels = (
    train_data["concept_set"].values,
    train_data["label"].values,
)

dataset = [
    {"data": t + str(args.sep_token) + l + str(args.eos_token), "label": l }
    for t, l in zip(train_text, train_labels)
]
train_loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
)

eval_data = pd.read_csv("data/val_data.csv", delimiter="\t")

eval_text, eval_labels = (
    eval_data["concept_set"].values,
    eval_data["label"].values,
)

dataset = [
    {"data": t + str(args.sep_token) + l + str(args.eos_token), "label": l }
    for t, l in zip(eval_text, eval_labels)
]
eval_loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
)

engine, _, _, _ = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=model.parameters(),
)

for epoch in range(args.epoch):
    model.train()
    for train in tqdm(train_loader):
        engine.zero_grad()
        text, label = train["data"], train["label"]
        text_tokens = tokenizer(
            text,
            return_tensors="pt",
            max_length=70,
            truncation=True,
            padding=True,
        )

        input_ids = text_tokens.input_ids.cuda()
        attention_mask = text_tokens.attention_mask.cuda()

        output = engine.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )

        loss = output.loss
        # print({"loss": loss})
        wandb.log({"loss": loss})
        engine.backward(loss)
        engine.step()


    with torch.no_grad():
        model.eval()
        for eval in tqdm(eval_loader):
            eval_text, eval_label = eval["data"], eval["label"]
            eval_text_tokens = tokenizer(
                eval_text,
                return_tensors="pt",
                max_length=70,
                truncation=True,
                padding=True,
            )

            input_ids = eval_text_tokens.input_ids.cuda()
            attention_mask = eval_text_tokens.attention_mask.cuda()

            eval_out = engine.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )

            eval_loss = eval_out.loss   

            # print({"eval_loss": eval_loss})
        wandb.log({"eval_loss": eval_loss})
        wandb.log({"epoch": epoch+1})

        torch.save(model.state_dict(), f"model_save/{model_name.replace('/', '-')}-{epoch+1}.pt")
