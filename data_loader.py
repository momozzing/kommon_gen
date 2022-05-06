import json
import pandas as pd

with open('data/Test/label_data/korean_commongen_test_labeled.json') as train_file:
    train_data = json.load(train_file)

# print(train_data)


concept_set = []
label = []

for i in range(len(train_data["concept_set"])):
    concept_set.append(train_data['concept_set'][i]['concept_set'].replace("#", " "))
    label.append(train_data['concept_set'][i]['reference_1'])


concept_set_df = pd.DataFrame(concept_set, columns = ['concept_set'])
label_df = pd.DataFrame(label, columns = ['label'])
all_df = pd.concat([concept_set_df, label_df], axis=1)

all_df.to_csv('data/Test/label_data/test_data.csv', sep='\t')

# tmp = []
# for i, text in enumerate(all_df['label']):                    concept maxlen 38  label maxlen 65
#     # print(len(str(text)))
#     tmp.append(len(str(text)))

# max_value = None

# for num in tmp:
#     if (max_value is None or num > max_value):
#         max_value = num

# print('Maximum value:', max_value)        