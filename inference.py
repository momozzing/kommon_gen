"""
python interactive.py
"""
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer

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

with torch.no_grad():
    while True:
        t = input("\nUser: ")
        tokens = tokenizer(
            t+'[sep]',
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=30
        )

        input_ids = tokens.input_ids.cuda()
        # print(input_ids)
        # print(tokenizer.convert_ids_to_tokens(input_ids[0])) 
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
        print("System: " + tokenizer.decode(gen[len(input_ids[0]):-1], skip_special_tokens=True))



#     en_label.append(str(label))
#     gen_label.append(str(tokenizer.decode(sample_output[0], skip_special_tokens=True)))

# label_df = pd.DataFrame(en_label, columns = ['label'])
# gen_df = pd.DataFrame(gen_label, columns = ['gen'])
# all_df = pd.concat([label_df, gen_df], axis=1)
# all_df.to_csv(f'result/gen_text_{ckpt}.csv', sep='\t')