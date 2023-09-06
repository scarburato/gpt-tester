# Copyright (c) 2023 Eric Mitchell
# Copyright (c) 2023 Dario Pagani
# Copyright (c) 2023 Francesco Londretti

import numpy as np
import transformers
import torch
import re
import os
import socket

BUFFER_SIZE = 1
DEVICE = "cpu"
MASK_FILLING_MODEL_NAME = "t5-large" #"t5-3b" # "t5-large"
CACHE_FOLDER = None

pattern = re.compile(r"<extra_id_\d+>")

# @TODO capire che è sto model_max_len
mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(MASK_FILLING_MODEL_NAME, **{}, **dict(torch_dtype=torch.bfloat16))
mask_tokenizer = transformers.AutoTokenizer.from_pretrained(MASK_FILLING_MODEL_NAME, model_max_length=mask_model.config.n_positions)
base_model = transformers.AutoModelForCausalLM.from_pretrained("gpt2-medium", **{})
base_tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2-medium", **{})

def tokenize_and_mask(text, span_length, pct, ceil_pct=False):
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + BUFFER_SIZE * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - BUFFER_SIZE)
        search_end = min(len(tokens), end + BUFFER_SIZE)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1

    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text

def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]

def replace_masks(texts):
    n_expected = count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    outputs = mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=1.0, num_return_sequences=1, eos_token_id=stop_id)
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)

def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills

def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts

def get_ll(text):
    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        labels = tokenized.input_ids
        return -base_model(**tokenized, labels=labels).loss.item()

def get_lls(texts):
    return [get_ll(text) for text in texts]


def evaluate_texts(texts: list):
    texts_masked = [tokenize_and_mask(x, 2, 0.35) for x in texts]
    print(len(texts_masked))

    texts_masked = replace_masks(texts_masked)
    print("finito replace_masks")

    # raw_fills = replace_masks(texts_masked)
    # print("finito texts_masked")

    extracted_fills = extract_fills(texts_masked)
    print("finito extract_fills")

    perturbed_texts = apply_extracted_fills(texts_masked, extracted_fills)
    print("finito apply_extracted_fills")

    orginal_lls = get_lls(texts)
    perturbed_lls = get_lls(perturbed_texts)
    print("finito getllss")

    return [{"isGPT": abs(oll - pll) > 0.85, "lp": abs(oll - pll)} for oll, pll in zip(orginal_lls, perturbed_lls)]


if __name__ == '__main__':
    #text = "Custom logits processors that complement the default logits processors built from arguments and generation config. If a logit processor is passed that is already created with the
    # arguments or a generation config an error is thrown. This feature is intended for advanced users."

    texts = []
    labels = []
    folder = "samples"

    for filename in os.listdir(folder):
        if filename not in ["2", "7"]:
            continue

        fn = os.path.join(folder, filename)
        print(fn)
        with open(fn, 'r') as f:
            texts.append(f.read())
            labels.append("MACCHINA" if int(filename) < 5 else "UOMO")

    texts_masked = [tokenize_and_mask(x, 2, 0.35) for x in texts]
    print(len(texts_masked))

    texts_masked = replace_masks(texts_masked)
    print("finito replace_masks")

    #raw_fills = replace_masks(texts_masked)
    #print("finito texts_masked")

    extracted_fills = extract_fills(texts_masked)
    print("finito extract_fills")

    perturbed_texts = apply_extracted_fills(texts_masked, extracted_fills)
    print("finito apply_extracted_fills")

    orginal_lls = get_lls(texts)
    perturbed_lls = get_lls(perturbed_texts)
    print("finito getllss")

    for label, oll, pll in zip(labels, orginal_lls, perturbed_lls):
        print(label + '\t' + str(oll) + '\t' + str(pll) + '\n' + str(abs(oll - pll)))
