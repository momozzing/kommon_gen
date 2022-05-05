'''
deepspeed --num_gpus=1 GPT-2_fine_tune.py
'''

from argparse import ArgumentParser
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelWithLMHead
from torch.optim import AdamW
import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "true"

model_name = "skt/kogpt2-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# SPECIAL_TOKENS = {
#     "bos_token": "<bos>",
#     "eos_token": "<eos>",
#     "pad_token": "<pad>",
#     "sep_token": "<seq>"
#     }
# tokenizer.add_special_tokens(SPECIAL_TOKENS)

model = AutoModelWithLMHead.from_pretrained(
    model_name
).cuda()

# model.resize_token_embeddings(len(tokenizer)) 

parser = ArgumentParser()
parser.add_argument("--epoch", default=50, type=int)
parser.add_argument("--batch_size", default=128, type=int)
# parser.add_argument("--sep_token", default=tokenizer.sep_token, type=str)
parser.add_argument("--bos_token", default=tokenizer.bos_token, type=str)
parser.add_argument("--eos_token", default=tokenizer.eos_token, type=str)
args = parser.parse_args()

wandb.init(project="kommongen", name=f"kommongen-{model_name}")
train_data = pd.read_csv("data/Training/label_data/train_data.csv", delimiter="\t")
train_data = train_data[:3000]
train_text, train_labels = (
    train_data["concept_set"].values,
    train_data["label"].values,
)

dataset = [
    {"data": t + str(args.bos_token) + l + str(args.eos_token), "label": l }
    for t, l in zip(train_text, train_labels)
]
train_loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
)

eval_data = pd.read_csv("data/Training/label_data/train_data.csv", delimiter="\t")
eval_data = eval_data[3000:3500]

eval_text, eval_labels = (
    eval_data["concept_set"].values,
    eval_data["label"].values,
)

dataset = [
    {"data": t + str(args.bos_token) + l + str(args.eos_token), "label": l }
    for t, l in zip(eval_text, eval_labels)
]
eval_loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
)

optimizer = AdamW(params=model.parameters(),
    lr=3e-5, weight_decay=3e-7
)

for epoch in range(args.epoch):
    model.train()
    for train in tqdm(train_loader):
        optimizer.zero_grad()
        text, label = train["data"], train["label"]
        text_tokens = tokenizer(
            text,
            return_tensors="pt",
            max_length=50,
            truncation=True,
            padding=True,
        )
        label_tokens = tokenizer(
            label,
            return_tensors="pt",
            max_length=50,
            truncation=True,
            padding=True,
        )

        input_ids = text_tokens.input_ids.cuda()
        attention_mask = text_tokens.attention_mask.cuda()
        label_input_ids = label_tokens.input_ids.cuda()

        output = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )

        loss = output.loss
        wandb.log({"loss": loss})
        loss.backward()
        optimizer.step()

        pred = output.logits.argmax(-1)     

        correct = 0
        for pre, lab in zip(pred, label):
            if pre == lab:
                correct += 1

        wandb.log({"acc": correct / len(train_loader)})

    model.eval()
    for eval in tqdm(eval_loader):
        eval_text, eval_label = eval["data"], eval["label"]
        eval_text_tokens = tokenizer(
            eval_text,
            return_tensors="pt",
            max_length=50,
            truncation=True,
            padding=True,
        )
        eval_label_tokens = tokenizer(
            eval_label,
            return_tensors="pt",
            max_length=50,
            truncation=True,
            padding=True,
        )

        input_ids = eval_text_tokens.input_ids.cuda()
        attention_mask = eval_text_tokens.attention_mask.cuda()
        label_input_ids = eval_label_tokens.input_ids.cuda()

        eval_out = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )

        eval_loss = eval_out.loss   
        wandb.log({"eval_loss": eval_loss})

        eval_pred = eval_out.logits.argmax(-1)        

        eval_correct = 0
        for pre, lab in zip(eval_pred, eval_label):
            if pre == lab:
                eval_correct += 1

        wandb.log({"eval_acc": eval_correct / len(eval_loader)})
        wandb.log({"epoch": epoch})

    torch.save(model.state_dict(), f"model_save/{model_name.replace('/', '-')}.pt")
