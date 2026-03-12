import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import transformers
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Config
from transformers import TrainingArguments, Trainer
from transformers import OPTForCausalLM, GPT2Tokenizer
from modeling_qwen2_protected import Qwen2ForFICausalLM
# from transformers.models.gpt2.modeling_gpt2 import GPT2FlashAttention2
import numpy as np
import evaluate
import random
import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

torch.multiprocessing.set_start_method('spawn')
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import copy
import time
import re
import string
import threading
import sys
from collections import Counter
import struct
from functools import partial


def seed_torch(seed=196):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch()
device = "cuda"
torch.set_printoptions(threshold=np.inf)


def tokenize_function(examples):
    texts = [context + "\n\n" + question
             for context, question in zip(examples["context"], examples["question"])]
    result = tokenizer(texts, padding='max_length', truncation=True, max_length=1024)
    result["valid_length"] = [len(x) for x in tokenizer.batch_encode_plus(texts)["input_ids"]]

    return result


def hook1_fn(module, input, output):
    global flag
    global key_tokens
    global index
    if flag == key_tokens[index]:
        #print(flag)
        #print(index)
        #print(key_tokens[index])
        position1 = random.randint(0, 15)
        # print(output.shape)
        x = random.randint(0, output.shape[1] - 1)
        # print(x)
        y = random.randint(0, output.shape[2] - 1)
        # print(y)
        bs = output[0, x, y].item()
        # print(bs)
        bs = np.float16(bs)
        bs = struct.pack('<H', bs.view(np.uint16))
        # print(bs)
        #print(bs[0])
        #print(bs[1])
        mask = 1 << position1
        #print(position1)
        # mask1 = 0x6fff
        ins = struct.unpack('<H', bs)[0]
        # print(ins)
        ins = ins ^ mask
        # ins = ins & mask1
        bs = struct.pack('<H', ins)
        #print(bs[0])
        #print(bs[1])
        mod = np.frombuffer(bs, dtype=np.float16)
        tem = np.copy(mod)
        # print(tem)
        output[0, x, y] = torch.from_numpy(tem)[0]

    if flag == key_tokens[index + 1]:
        #print(flag)
        #print(index+1)
        #print(key_tokens[index+1])
        position2 = random.randint(0, 15)
        x = random.randint(0, output.shape[1] - 1)
        # print(x)
        y = random.randint(0, output.shape[2] - 1)
        # print(y)
        bs = output[1, x, y].item()
        # print(bs)
        bs = np.float16(bs)
        bs = struct.pack('<H', bs.view(np.uint16))
        # print(bs)
        # print(bs[0])
        # print(bs[1])
        mask = 1 << position2
        #print(position2)
        # mask1 = 0x6fff
        ins = struct.unpack('<H', bs)[0]
        # print(ins)
        ins = ins ^ mask
        # ins = ins & mask1
        bs = struct.pack('<H', ins)
        # print(bs[0])
        # print(bs[1])
        mod = np.frombuffer(bs, dtype=np.float16)
        tem = np.copy(mod)
        # print(tem)
        output[1, x, y] = torch.from_numpy(tem)[0]

    return output

def hook2_fn(module, input, output):
    global flag
    global key_tokens
    global index

    if flag == key_tokens[index]:
        position1 = random.randint(0, 15)
        #print(flag)
        #print(index)
        #print(key_tokens[index])
        # print(output.shape)
        x = random.randint(0, output.shape[0] / 2 - 1)
        # print(x)
        y = random.randint(0, output.shape[1] - 1)
        # print(y)
        bs = output[x, y].item()
        # print(bs)
        bs = np.float16(bs)
        bs = struct.pack('<H', bs.view(np.uint16))
        # print(bs)
        #print(bs[0])
        #print(bs[1])
        mask = 1 << position1
        #print(position1)
        # mask1 = 0x6fff
        ins = struct.unpack('<H', bs)[0]
        # print(ins)
        ins = ins ^ mask
        # ins = ins & mask1
        bs = struct.pack('<H', ins)
        #print(bs[0])
        #print(bs[1])
        mod = np.frombuffer(bs, dtype=np.float16)
        tem = np.copy(mod)
        # print(tem)
        output[x, y] = torch.from_numpy(tem)[0]

    if flag == key_tokens[index + 1]:
        #print(flag)
        #print(index+1)
        #print(key_tokens[index+1])
        position2 = random.randint(0, 15)
        x = random.randint(output.shape[0] / 2, output.shape[0] - 1)
        # print(x)
        y = random.randint(0, output.shape[1] - 1)
        # print(y)
        bs = output[x, y].item()
        # print(bs)
        bs = np.float16(bs)
        bs = struct.pack('<H', bs.view(np.uint16))
        # print(bs)
        # print(bs[0])
        # print(bs[1])
        mask = 1 << position2
        #print(position2)
        # mask1 = 0x6fff
        ins = struct.unpack('<H', bs)[0]
        # print(ins)
        ins = ins ^ mask
        # ins = ins & mask1
        bs = struct.pack('<H', ins)
        # print(bs[0])
        # print(bs[1])
        mod = np.frombuffer(bs, dtype=np.float16)
        tem = np.copy(mod)
        # print(tem)
        output[x, y] = torch.from_numpy(tem)[0]

    return output

def clamp_to_zero(tensor, lower, upper):
    mask = (tensor < lower) | (tensor > upper)

    return torch.where(mask, torch.zeros_like(tensor), tensor)

def hook_fn(module, input, output, l, layer):
    global flag
    #global index
    global min1
    global max1
    if flag > 0:
        output = clamp_to_zero(output, lower=min1[l][layer].item(), upper=max1[l][layer].item())
        #output.clamp_(min=min1[l][layer].item(), max=max1[l][layer].item())
        output = torch.nan_to_num(output, nan=0.0)
    # print(index)
        #index += 1
        #if index == 24:
        #    index = 0

            # print(output[0, :100, 500])
            # print(module.weight.data[500, value])

    return output


def hook_fn2(module, input, output, l, layer):
    global flag
    #global index
    global min1
    global max1
    if flag == 0:
        min1[l][layer] = torch.min(output)*2
        max1[l][layer] = torch.max(output)*2
    #if min1[l][layer] > min_val:
    #    min1[l][layer] = min_val
    #if max1[l][layer] < max_val:
    #    max1[l][layer] = max_val

            # print(output[0, :100, 500])
        # print(module.weight.data[500, value])
        # print(index)
    #print(layer)
    #print(l)
    #index += 1
    #if index == 24:
    #    index = 0
    
    return output


def get_input(dataset, tokenizer, id, device):
    question = dataset["question"][id]
    context = dataset["context"][id]
    reference = dataset["answers"][id]["text"]
    prompt = context + "\n\n" + question
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device, non_blocking=True)
    return reference, prompt, input_ids


def generate(qid, dataset, tokenizer, model, max_length=40):
    reference, prompt, input_ids = get_input(dataset, tokenizer, qid, model.device)
    prompt_len = len(input_ids[0])

    with torch.no_grad():
        output = model.generate(input_ids, max_length=prompt_len + max_length, do_sample=False, output_scores=True,
                                no_repeat_ngram_size=5, num_return_sequences=1, return_dict_in_generate=True)
    token_ids = output.sequences[0][prompt_len:]
    probs = torch.stack(output.scores, dim=1).softmax(-1)[0]

    generated_text = tokenizer.decode(token_ids, skip_special_tokens=True)

    return generated_text, probs

def normalize_answer(s):
    """Lower text and remove punctuation, articles, and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()


def compute_precision_recall_f1(prediction, ground_truth):
    prediction_tokens = get_tokens(prediction)
    ground_truth_tokens = get_tokens(ground_truth)
    common_tokens = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common_tokens.values())

    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        # If either is no-answer, F1 is 1 if they agree, 0 otherwise
        return (int(prediction_tokens == ground_truth_tokens),) * 3
    if num_same == 0:
        return 0, 0, 0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1


def read_qids_from_file(filename):
    with open(filename, 'r') as file:
        return [int(line.strip()) for line in file][:200]


if __name__ == '__main__':
    print(torch.__version__)
    print(torch.cuda.is_available())
    qid = read_qids_from_file("qidsquadfinal.txt")
    # print(qid[0])
    dataset = datasets.load_dataset("squad_v2", split="validation")
    dataset1 = datasets.load_dataset("squad_v2", split="train")
    # dataset = dataset.filter(lambda x: x['answers']['text'])
    # print(len(dataset))
    # print(dataset['id'][qid[0]])
    dataset = dataset.select(qid)
    print(len(dataset))
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-Math-7B")
    print("before model")

    # model = OPTForCausalLM.from_pretrained("facebook/opt-2.7b", torch_dtype=torch.float16, attn_implementation="flash_attention_2")
    model = Qwen2ForFICausalLM.from_pretrained("Qwen/Qwen2-Math-7B", torch_dtype=torch.float16, device_map="cuda")
    # model.to(device)
    print("model finish")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    # tokenized_datasets = tokenized_datasets.filter(lambda example: example['valid_length'] > 32)
    #tokenized_datasets = tokenized_datasets.filter(lambda example: example['valid_length'] <= 156)
    #tokenized_datasets = tokenized_datasets.filter(lambda example: len(example["answers"]["text"]) == 3)
    #tokenized_datasets["answers"][:]["text"] = tokenized_datasets["answers"][:]["text"][0] 
    # tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    # tokenized_datasets = tokenized_datasets.remove_columns(["attention_mask"])
    # tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    # print(tokenized_datasets)   
    # small_train_dataset = tokenized_datasets["train"].shuffle().select(range(50))
    # print(small_train_dataset['input_ids'])
    train_dataloader = DataLoader(tokenized_datasets, shuffle=False, batch_size=2)
    bs = 2
    l = 512

    print("before train")
    for layer in model.children():
        print(layer)
    # paras = list(model.parameters())
    # np.set_printoptions(suppress=True)
    # print(paras)

    device = torch.device("cuda:0")
    print("device done")
    model.cuda()
    print("cuda done")
    progress_bar = tqdm(range(len(train_dataloader) * 500))
    print("tqdm done")

    length_file = open("1024lengths.txt", "w")
    time_file = open("1024generation_times.txt", "w")
    golden_file = open("1024golden_times.txt", "w")
    total = 0
    totalm = 0
    idx = 0
    precision = 0
    recall = 0
    f1 = 0
    total_pre = 0
    total_re = 0
    total_re2 = 0
    total_f1 = 0
    total_samples = 0
    accuracy = 0
    correct_samples = 0


    global flag
    flag = 0
    global index
    index = 0
    global key_tokens
    with open("squadrandomtokenqwen.txt", "r") as f:
        key_tokens = [int(line.strip()) for line in f]
    #print(key_tokens)

    global min1
    min1 = torch.arange(196, dtype=torch.float16).reshape(7, 28).fill_(1000)
    global max1
    max1 = torch.arange(196, dtype=torch.float16).reshape(7, 28).fill_(-1000)
    min1.to(device)
    max1.to(device)


    handles = []
    for i in range(28):
    #    handle2 = model.model.decoder.layers[i].self_attn.out_proj.register_forward_hook(hook_fn2)
    #    handles.append(handle2)
        hookop = lambda module, input, output, i=i: hook_fn2(module, input, output, l=3, layer=i)
        #handle2 = model.model.layers[i].self_attn.o_proj.register_forward_hook(hookop)
        #handle3 = model.model.decoder.layers[i].self_attn.out_proj.register_forward_hook(hookoc)
        hookkp = lambda module, input, output, i=i: hook_fn2(module, input, output, l=1, layer=i)
        #handle4 = model.model.layers[i].self_attn.k_proj.register_forward_hook(hookkp)
        #handle5 = model.model.decoder.layers[i].self_attn.k_proj.register_forward_hook(hookkc)
        hookqp = lambda module, input, output, i=i: hook_fn2(module, input, output, l=0, layer=i)
        #handle6 = model.model.layers[i].self_attn.q_proj.register_forward_hook(hookqp)
        #handle7 = model.model.decoder.layers[i].self_attn.q_proj.register_forward_hook(hookqc)
        hookvp = lambda module, input, output, i=i: hook_fn2(module, input, output, l=2, layer=i)
        #handle8 = model.model.layers[i].self_attn.v_proj.register_forward_hook(hookvp)
        #handle9 = model.model.decoder.layers[i].self_attn.v_proj.register_forward_hook(hookvc)
        hook1p = lambda module, input, output, i=i: hook_fn2(module, input, output, l=4, layer=i)
        #handle10 = model.model.layers[i].mlp.up_proj.register_forward_hook(hook1p)
        #handle11 = model.model.decoder.layers[i].fc1.register_forward_hook(hook1c)
        hook2p = lambda module, input, output, i=i: hook_fn2(module, input, output, l=5, layer=i)
        #handle12 = model.model.layers[i].mlp.gate_proj.register_forward_hook(hook2p)
        hook3p = lambda module, input, output, i=i: hook_fn2(module, input, output, l=6, layer=i)
        #handle14 = model.model.layers[i].mlp.down_proj.register_forward_hook(hook3p)
        #handle13 = model.model.decoder.layers[i].fc2.register_forward_hook(hook2c)
        #handles.append(handle2)
        #handles.append(handle3)
        #handles.append(handle4)
        #handles.append(handle5)
        #handles.append(handle6)
        #handles.append(handle7)
        #handles.append(handle8)
        #handles.append(handle9)
        #handles.append(handle10)
        #handles.append(handle11)
        #handles.append(handle12)
        #handles.append(handle14)

    #for _ in range(10):
    #    q = random.randint(1, len(dataset1))
    #    _, _ = generate(q, dataset1, tokenizer, model, max_length=50)

    for handle in handles:
        handle.remove()
    handles.clear()
    
    #torch.save(min1, "min5.pt")
    #minnp = min1.numpy()
    #np.savetxt("min5txt", minnp, fmt='%.6e')
    #torch.save(max1, "max5.pt")
    #maxnp = max1.numpy()
    #np.savetxt("max5.txt", maxnp, fmt='%.6e')

    #summ = 0
    rs = [random.randint(1, 2000) for _ in range(1)]
    layer_weights = {
    'v_proj': 1,
    'k_proj': 1,
    'q_proj': 7,
    'out_proj': 7,
    'up_proj': 37,
    'gate_proj': 37,
    'down_proj': 37
    }

    # Calculate total weight
    total_weight = sum(layer_weights.values())
    layers = list(layer_weights.keys())
    weights = [layer_weights[layer]/total_weight for layer in layers]

    for step, batch in enumerate(train_dataloader):
        temp = batch["answers"]
        #print(len(temp))
        # batch = {k: v.to(device) for k, v in batch.items()}
        # batch.remove_columns(["context", "valid_length", "question", "answers", "id"])
        for key in ["title", "context", "valid_length", "question", "id", "answers"]:
            batch.pop(key, None)
        batch = {k: v.to(device) for k, v in batch.items()}
        idx = idx + 1
        # if idx == 1:
        #    print(batch['input_ids'])
        input_length = torch.sum(batch['attention_mask']).item()
        #length_file.write(str(input_length) + "\n")
        prompt_len = len(batch['input_ids'][0])
        answer = temp['text'][:][0]
        max_length = 60

        past_key_values = None
        generated = batch['input_ids']
        attention_mask = batch['attention_mask']
        input_ids = batch['input_ids']
        for i in range(max_length):
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "use_cache": True,
            }

            # Model forward pass
            with torch.no_grad():
                outputs = model(**model_inputs)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)

            input_ids = next_token.unsqueeze(1)
            generated = torch.cat([generated, input_ids], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones(bs, 1).to(device)], dim=1)

            # Update KV Cache
            past_key_values = outputs.past_key_values

        generatedtext1 = tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True,
                                               no_repeat_ngram_size=5,
                                               num_return_sequences=1)
        base_output = generatedtext1

        for _ in range(20):
            handles = []
            b = random.randint(0, 27)
            selected_layer = random.choices(layers, weights=weights)[0]
            if selected_layer == 'v_proj':
                handle1 = model.model.layers[b].self_attn.v_proj.register_forward_hook(hook1_fn)
            elif selected_layer == 'k_proj':
                handle1 = model.model.layers[b].self_attn.k_proj.register_forward_hook(hook1_fn)
            elif selected_layer == 'q_proj':
                handle1 = model.model.layers[b].self_attn.q_proj.register_forward_hook(hook1_fn)
            elif selected_layer == 'out_proj':
                handle1 = model.model.layers[b].self_attn.o_proj.register_forward_hook(hook1_fn)
            elif selected_layer == 'up_proj':
                handle1 = model.model.layers[b].mlp.up_proj.register_forward_hook(hook1_fn)
            elif selected_layer == 'gate_proj':
                handle1 = model.model.layers[b].mlp.gate_proj.register_forward_hook(hook1_fn)
            else:  # fc2
                handle1 = model.model.layers[b].mlp.down_proj.register_forward_hook(hook1_fn)
            #handle1 = model.model.decoder.layers[b].self_attn.v_proj.register_forward_hook(hook1_fn)
            #print(b)
            #handle1.remove()
            #if selected_layer == 'fc2':
            #    summ = summ + 1
            #    print(summ)
            for i in range(28):
                #handle2 = model.model.decoder.layers[i].self_attn.out_proj.register_forward_hook(hookop)
                hookop = lambda module, input, output, i=i: hook_fn2(module, input, output, l=3, layer=i)
                #handle2 = model.model.layers[i].self_attn.o_proj.register_forward_hook(hookop)
                hookoc = lambda module, input, output, i=i: hook_fn(module, input, output, l=3, layer=i)
                #handle3 = model.model.layers[i].self_attn.o_proj.register_forward_hook(hookoc)
                #handle4 = model.model.decoder.layers[i].self_attn.k_proj.register_forward_hook(hookkp)
                hookkp = lambda module, input, output, i=i: hook_fn2(module, input, output, l=1, layer=i)
                #handle4 = model.model.layers[i].self_attn.k_proj.register_forward_hook(hookkp)
                hookkc = lambda module, input, output, i=i: hook_fn(module, input, output, l=1, layer=i)
                #handle5 = model.model.layers[i].self_attn.k_proj.register_forward_hook(hookkc)
                #handle6 = model.model.decoder.layers[i].self_attn.q_proj.register_forward_hook(hookqp)
                hookqp = lambda module, input, output, i=i: hook_fn2(module, input, output, l=0, layer=i)
                #handle6 = model.model.layers[i].self_attn.q_proj.register_forward_hook(hookqp)
                hookqc = lambda module, input, output, i=i: hook_fn(module, input, output, l=0, layer=i)
                #handle7 = model.model.layers[i].self_attn.q_proj.register_forward_hook(hookqc)
                #handle8 = model.model.decoder.layers[i].self_attn.v_proj.register_forward_hook(hookvp)
                hookvp = lambda module, input, output, i=i: hook_fn2(module, input, output, l=2, layer=i)
                #handle8 = model.model.layers[i].self_attn.v_proj.register_forward_hook(hookvp)
                hookvc = lambda module, input, output, i=i: hook_fn(module, input, output, l=2, layer=i)
                #handle9 = model.model.layers[i].self_attn.v_proj.register_forward_hook(hookvc)
                #handle10 = model.model.decoder.layers[i].fc1.register_forward_hook(hook1p)
                hook1p = lambda module, input, output, i=i: hook_fn2(module, input, output, l=4, layer=i)
                #handle10 = model.model.layers[i].mlp.up_proj.register_forward_hook(hook1p)
                hook1c = lambda module, input, output, i=i: hook_fn(module, input, output, l=4, layer=i)
                #handle11 = model.model.layers[i].mlp.up_proj.register_forward_hook(hook1c)
                #handle12 = model.model.decoder.layers[i].fc2.register_forward_hook(hook2p)
                hook2p = lambda module, input, output, i=i: hook_fn2(module, input, output, l=5, layer=i)
                #handle12 = model.model.layers[i].mlp.gate_proj.register_forward_hook(hook2p)
                hook2c = lambda module, input, output, i=i: hook_fn(module, input, output, l=5, layer=i)
                #handle13 = model.model.layers[i].mlp.gate_proj.register_forward_hook(hook2c)
                hook3p = lambda module, input, output, i=i: hook_fn2(module, input, output, l=6, layer=i)
                #handle14 = model.model.layers[i].mlp.down_proj.register_forward_hook(hook2p)
                hook3c = lambda module, input, output, i=i: hook_fn(module, input, output, l=6, layer=i)
                #handle15 = model.model.layers[i].mlp.down_proj.register_forward_hook(hook2c)
                #handles.append(handle2)
                #handles.append(handle3)
                #handles.append(handle4)
                #handles.append(handle5)
                #handles.append(handle6)
                #handles.append(handle7)
                #handles.append(handle8)
                #handles.append(handle9)
                #handles.append(handle10)
                #handles.append(handle11)
                #handles.append(handle12)
                #handles.append(handle13)
                #handles.append(handle14)
                #handles.append(handle15)

            past_key_values = None
            generated = batch['input_ids']
            # current_length = input_ids.size(1)

            # Initial attention_mask
            temp1 = copy.deepcopy(batch)
            attention_mask = batch['attention_mask']
            input_ids = batch['input_ids']
            #global result
            #result = torch.empty(1, dtype=torch.float32, device='cuda')
            model.model.calibration_mode = True
            start_t1 = time.time()
            for i in range(max_length):
                # Prepare model inputs
                #global flag
                #flag = 0
                #global result
                #result = torch.empty(1, dtype=torch.float32, device='cuda')
                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "past_key_values": past_key_values,
                    "use_cache": True,
                }
             
                # Model forward pass
                flag = i
                if i > 0:
                    model.model.calibration_mode = False
                s1 = time.time()
                with torch.no_grad():
                    outputs = model(**model_inputs)
                e1 = time.time()
                # print(e1-s1)
                #flag = i
                # Get logits of the next token
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)

                # Update generated sequence
                # Update input_ids with the new token
                input_ids = next_token.unsqueeze(1)
                generated = torch.cat([generated, input_ids], dim=-1)
                #input_ids = torch.cat([input_ids, input_ids], dim=-1)
                #print(input_ids.shape)
                #print(attention_mask.shape[1])
                # Update attention_mask
                #if attention_mask.shape[1] > prompt_len:
                #    attention_mask = torch.cat([attention_mask[:, :-2], attention_mask[:, -1:]], dim=1)
                #attention_mask = torch.cat([attention_mask, torch.zeros(bs, 1).to(device)], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones(bs, 1).to(device)], dim=1)
                #print(attention_mask.shape[1])
                # attention_mask = torch.tensor([[1, 0]])
                # If you want to implement the previously discussed [0, 1] pattern, you can modify as follows:
                # new_attention_mask = torch.zeros((1, 2), dtype=torch.long)
                # new_attention_mask[0, 1] = 1
                # attention_mask = torch.cat([attention_mask, new_attention_mask], dim=-1)

                # Update KV Cache
                past_key_values = outputs.past_key_values
                #new_past_key_values = tuple(
                #    (
                #        torch.cat([layer_key[:, :, :-2, :], layer_key[:, :, -1:, :]], dim=2),  # Remove the last sequence position
                #        torch.cat([layer_value[:, :, :-2, :], layer_value[:, :, -1:, :]], dim=2)  # Remove the last sequence position
                #    )
                #    for layer_key, layer_value in past_key_values
                #)
                #if past_key_values[0][0].shape[2] > prompt_len:
                #    past_key_values = new_past_key_values
                #print(past_key_values[0][0].shape[2])
            # Update current length

            # current_length += 1

            # mem_before = torch.cuda.memory_allocated()
            end_t1 = time.time()
            generatedtext = tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True, no_repeat_ngram_size=5,
                                             num_return_sequences=1)
            # end_t1 = time.time()
            for handle in handles:
                handle.remove()
            handles.clear()
            handle1.remove()



            golden_time = end_t1 - start_t1
            golden_file.write(str(golden_time) + "\n")
            if idx > 1:
                total += golden_time
            i = 0

            for modify_output in generatedtext:
                #base_output = generatedtext1
                #modify_output = generatedtext
                #print(base_output)
                #print("-------------------------------")
                #print(modify_output)
                #print("-------------------------------")
            # total_f1 += f11
                pre2, re2, f12 = compute_precision_recall_f1(modify_output, answer[i])
            # total_pre2 += pre2
                if re2 < 1:
                    re2 == 0
                total_re2 += re2
            # total_f12 += f12

                total_samples += 1
            #print(result)
                if base_output[i] == modify_output:
                    correct_samples += 1
                i = i + 1
                # max_memory = torch.cuda.max_memory_allocated(device=device)
                # max_memory_mb = max_memory / 1024 / 1024 / 1024

                # print(f"max_memory: {max_memory_mb:.2f} GB")

            progress_bar.update(1)

        index = index + 2

    max_memory = torch.cuda.max_memory_allocated(device=device)
    max_memory_mb = max_memory / 1024 / 1024 / 1024

    print(f"max_memory: {max_memory_mb:.2f} GB")
    accuracy = correct_samples / total_samples
    # precision = total_pre / total_samples
    recall2 = total_re2 / total_samples
    # f1 = total_f1 / total_samples
    print(f"\nAccuracy : {accuracy:.4f}")
    # print(f"\nPrecision : {precision:.4f}")
    print(f"\nRecall2 : {recall2:.4f}")
    # print(f"\nF1 : {f1:.4f}")
    total = total / (idx - 1)
    total = total / 20
    print(total)
    length_file.close()
    time_file.close()
