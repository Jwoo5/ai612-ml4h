from transformers import AutoTokenizer
import torch
import numpy as np
import pandas as pd

import pickle
from tqdm import tqdm

print("[Ready] setup the environment")
with open('/home/ghhur/data/input/tmp/concat_b/tokenized_mimic_12_all_150.pkl', 'rb') as f:
    data = pickle.load(f)
tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

valued_token_type_ids = []
print("[Processing] transform token_type_ids with value encoding")
with tqdm(total = 18536 * 150) as pbar:
    for input_ids, token_types in zip(data['all_name_12hr_input_ids'], data['all_name_12hr_token_type_ids']):
        for idcs, token_type in zip(input_ids, token_types):
            def decode_transform(idx, n_digits, is_decimal):
                try:
                    victim = tokenizer.decode(idcs[idx])
                except IndexError:
                    if is_decimal:
                        return 0
                    else:
                        return 2

                if victim.isdigit():
                    if is_decimal:
                        digit = n_digits
                        decode_transform(idx + 1, n_digits + 1, is_decimal)
                    else:
                        digit = decode_transform(idx + 1, n_digits + 1, is_decimal)
                elif victim == '.':
                    if is_decimal:
                        decode_transform(idx + 1, n_digits = 0, is_decimal = False)
                        return 0
                    else:
                        decode_transform(idx + 1, n_digits = 7, is_decimal = True)
                        return 2
                else:
                    if is_decimal:
                        decode_transform(idx + 1, n_digits = 0, is_decimal = False)
                        return 0
                    else:
                        decode_transform(idx + 1, n_digits = 0, is_decimal = False)
                        return 2

                try:
                    token_type[idx] = torch.LongTensor([digit])
                except:
                    breakpoint()

                return (digit + 1)

            decode_transform(idx = 0, n_digits = 0, is_decimal = False)

            valued_token_type_ids.append(token_type)

            pbar.update(1)
        break

breakpoint()

print("[End] transform np.array into DataFrame")
data['valued_token_type_ids'] = np.stack(valued_token_type_ids, axis = 0)

print("[End] dump")
with open('value_tokenized_mimic_12_all_150.pkl', 'wb') as f:
    pickle.dump(f)