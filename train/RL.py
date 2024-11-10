import json
import random
import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Sequence
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy
from torch.utils.data import Dataset
import transformers
import copy
from tqdm.auto import tqdm
from peft import LoraConfig, TaskType, get_peft_model

from peifg.model import *

max_length = 720
padding_strategy = "right"
output_dir = "/RL"
device = torch.device("cuda:0")
DEFAULT_EOS_TOKEN = "<|extra_0|>"
DEFAULT_BOS_TOKEN = "<|extra_1|>"
DEFAULT_UNK_TOKEN = "<|extra_2|>"
def preprocess_dataset(feedback_data, score_data, destination_dataset):
    with open(feedback_data, 'r') as file:
        feedback_list = json.load(file)[:2000]
    with open("./dataset/test.json", 'r') as file:
        all_data = json.load(file)   
    distractor_list = []
    for item in all_data:
        for dis in item['new_distractors']:
            distractor_list.append(dis)
    with open(score_data, "r") as file:
        # 逐行读取文件内容
        score_list = []
        for line in file:
            line_num = json.loads(line)
            score_list.append(line_num)

    sen_score_ls = []
    for feedback_sublist, score_sublist in zip(feedback_list, score_list):
        sen_score_subls = []
        for f, s in zip(feedback_sublist, score_sublist):
            sen_score_subls.append([f, s])
        sen_score_ls.append(sen_score_subls)
    better_ls, worse_ls = [], []
    total_better_ls, total_worse_ls = [], []
    distractor_ls = []
    for sen_score_subls, distractor in zip(sen_score_ls, distractor_list):
        for pair in list(itertools.combinations(sen_score_subls, 2)):
            if pair[0][1] > pair[1][1]:
                better_ls += [pair[0][0]]
                worse_ls += [pair[1][0]]
                distractor_ls += [distractor]
            elif pair[0][1] < pair[1][1]:
                better_ls += [pair[1][0]]
                worse_ls += [pair[0][0]]
                distractor_ls += [distractor]
    total_better_ls += better_ls
    total_worse_ls += worse_ls

    destination_data = {
        "type": "text2text",
        "instances": [
            {"prompt":"Distractor:"+ dis +"\nFeedback:", "chosen": ele_better, "rejected": ele_worse}
            for dis, ele_better, ele_worse in zip(distractor_ls,total_better_ls, total_worse_ls)
        ],
    }
    with open(destination_dataset, "w") as f:
        json.dump(destination_data, f)   





# Load the dataset using the HuggingFace dataset library
extensions = "json"
KEY_INSTANCES = "instances"
# 数据集处理
feedback_data = "./RL/feedback.json"
score_data = "./RL/scores.json"
distionation_data = "./RL/path"
preprocess_dataset(feedback_data, score_data, distionation_data)
raw_dataset = load_dataset(
    extensions,
    data_files=[distionation_data],
    field=KEY_INSTANCES,
    split="train",
)


from transformers import AutoTokenizer

model_name ='peifg'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.add_special_tokens(
    {
        "eos_token": DEFAULT_EOS_TOKEN,
        "bos_token": DEFAULT_BOS_TOKEN,
        "unk_token": DEFAULT_UNK_TOKEN,
    }
)
model = peifgQwenForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, device_map=device, trust_remote_code=False)


training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="no",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=32,
    learning_rate=2e-5,
    weight_decay=0,
    num_train_epochs=3,
    warmup_ratio=0,
    logging_strategy="steps",
    logging_first_step=True,
    save_strategy="epoch",
    save_total_limit=3,
    seed=42,
    run_name="wandb",
    load_best_model_at_end=False,
    greater_is_better=False,
    #deepspeed=ds_config,
    log_on_each_node=False,
    logging_steps=1,
    fp16=False,
    lr_scheduler_type="cosine",
    remove_unused_columns=False
)
from trl import SFTTrainer,DPOTrainer


lora_alpha = 16
lora_dropout = 0.1
lora_r = 8


peft_config = LoraConfig(
    target_modules=r'.*\.(c_proj|c_attn)', 
    inference_mode=False, 
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
)
model = get_peft_model(model, peft_config)

dpo_trainer = DPOTrainer(
    model,
    None,
    args=training_args,
    beta=0.1,
    train_dataset=raw_dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
)

dpo_trainer.train()
dpo_trainer.save_model()
print("Saving last checkpoint of the model")
model.save_pretrained(output_dir)




