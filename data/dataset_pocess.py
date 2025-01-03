
import io
import os
import copy
import json
import logging
import torch
import random
import numpy as np

from typing import List, Optional, Tuple, Union, Dict, Sequence
from PIL import Image, ImageFile, ImageDraw, ImageColor, ImageFont 
ImageFile.LOAD_TRUNCATED_IMAGES = True
import transformers 
from peifg.data.base_dataset import BaseDataset
# TODO modified to custom constants
from peifg.utils.constants import *
from peifg.utils import conversation as conversation_lib

COLOR_MASKS = ['red', 'yellow', 'blue', 'green', 'purple', 'white', 'black', 'brown', 'orange', 'pink', 'gold', 'silver','gray']

class MyDataset_Feedback(BaseDataset):
    """Conversation format dataset stage2 fine-tuning."""

    def __init__(self, datasets, tokenizer, multimodal_cfg):
        super(MyDataset_Feedback, self).__init__(datasets, tokenizer, multimodal_cfg)
        # v0 version format conversation
        conversation_lib.default_conversation = conversation_lib.conv_templates["opt"]
        logging.warning("Formatting inputs into conversation type: mpt-fixed")
        logging.warning("Loading data...")

        list_data_dict = []
        list_image_path = [] 
        self.instruct_blip_tokenizer = transformers.AutoTokenizer.from_pretrained("./instructblip-flan-t5-xl/qformer_tokenizer", trust_remote_code=True, padding_side="right", max_length=200)
        with open("./instructions.json", "r") as file:
            self.instructions = json.load(file)
        for name in datasets.split("+"):
            dataset = CONVERSATION_DATA[name]

            data_path = dataset['annotations']
            data = json.load(open(data_path, "r"))

            list_data_dict.extend(data)

            image_path = dataset['images']

            list_image_path.extend([image_path] * len(data))

            logging.warning(f"Data from {data_path} provide {len(data)} conversations.")

        assert len(list_data_dict) == len(list_image_path)
        logging.warning(f"{len(list_data_dict)} conversations in total.")
        a_new_list = list(zip(list_data_dict, list_image_path))
        random.shuffle(a_new_list)
        list_data_dict_new, list_image_path_new = zip(*a_new_list)
        self.list_data_dict = list_data_dict_new
        self.list_image_path = list_image_path_new

        self.im_patch_token = 151859

        self.im_start_token = 151857

        self.im_end_token = 151858
    
    def multimodal_processor(self, sources):

        for source in sources:
            if self.multimodal_cfg['sep_image_conv_front']:
                assert DEFAULT_IMAGE_TOKEN in source[0]['value']
                source[0]['value'] = source[0]['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                source[0]['value'] = DEFAULT_IMAGE_TOKEN + conversation_lib.default_conversation.sep + conversation_lib.default_conversation.roles[0] + ": " + source[0]['value']
            for sentence in source:
                # replace the <image>
                replace_token = DEFAULT_IMAGE_PATCH_TOKEN * self.multimodal_cfg['image_token_len']
                # if self.multimodal_cfg['use_im_start_end']:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                sentence["value"] = str(sentence["value"]).replace(DEFAULT_IMAGE_TOKEN, replace_token)
                # replace expert token
                replace_ex_token = DEFAULT_EXPERT_PAD_TOKEN * self.multimodal_cfg['expert_token_len']
                replace_ex_token = DEFAULT_EXPERT_START_TOKEN + replace_ex_token + DEFAULT_EXPERT_END_TOKEN
                sentence["value"] = str(sentence["value"]).replace(DEFAULT_EXPERT_TOKEN, replace_ex_token)
        return sources

    def _tokenize_fn(self, strings):
        """Tokenize a list of strings."""
        tokenized_list = [
            self.tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            ) for text in strings
        ]
        input_ids = labels = [
            tokenized.input_ids[0] for tokenized in tokenized_list
        ]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(self.tokenizer.pad_token_id).sum().item()
            for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def _mask_targets(self, target, tokenized_lens, speakers):
        # cur_idx = 0
        cur_idx = tokenized_lens[0]
        tokenized_lens = tokenized_lens[1:]
        target[:cur_idx] = IGNORE_INDEX
        for tokenized_len, speaker in zip(tokenized_lens, speakers):
            if speaker.lower() == "human":
                target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
            cur_idx += tokenized_len

    def token_processor(self, sources):
        conv = conversation_lib.conv_templates['opt'].copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]
                # source = "i do not know."

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())


       

        input_ids = self.tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        ).input_ids

        targets = input_ids.clone()
        assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

        # Mask targets
        sep = conv.sep + conv.roles[1]
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(self.tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep)
            re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
            for conv_idx in range(3, len(rounds), 2):
                re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
            cur_len = 0
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(re_rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                round_len = len(self.tokenizer(rou).input_ids) + len(self.tokenizer(conv.sep).input_ids)

                instruction_len = len(self.tokenizer(parts[0]).input_ids)
                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX

            if cur_len < self.tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

        return dict(
            input_ids=input_ids,
            labels=targets,
        )
    
    def __len__(self):
        return len(self.list_data_dict) * 3

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        j = i % 3 - 1
        i = i // 3

        font_size = 15
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",size=font_size)
        data = copy.deepcopy(self.list_data_dict[i])
        filtered_object_boxes = {f"person{i}": [round(num) for num in values[0:4]] for i, (key, values) in enumerate(data["object_boxes"].items()) if "person" in key}
        
        if isinstance(data, dict):
            if 'img_name' in data:
                image_path = self.list_image_path[i]
                image_file = data['img_name']

                try:
                    image = Image.open(image_path + image_file).convert('RGB')
                    # img1 = ImageDraw.Draw(image)
                    draw = ImageDraw.Draw(image, 'RGBA') 
                    # Loop through each bounding box and draw it on the image
                    idx=0
                    for label, box_coordinates in filtered_object_boxes.items():
                        color=ImageColor.getrgb(COLOR_MASKS[idx%len(COLOR_MASKS)]) + (128,)
                        draw.rectangle(box_coordinates, outline=color,width=4)
                        draw.text((box_coordinates[0], box_coordinates[1]), label, fill="teal",font=font)
                        idx=idx+1
                except:
                    print(f'cannot identify image file {image_path + image_file}.')
                    return self.__getitem__(0)

                try:
                    image, image_1 = self.image_processor(image)
                except:
                    print(f'image {image_file} are broken or grayscale! we thus select 0-th sample instead!')
                    return self.__getitem__(0)
            conversations = self.multimodal_processor([
                            [{"from": "human",
                              "value":"<image> <expert>\nquestion: "+ data["question"]+"\nanswer: "+ data["answers"] + "\ndistractor: " + data["new_distractors"][j]},
                             {"from": "gpt",
                              "value": "Eductional level: " + data["new_feedbacks"][j]["level"] + "\nMisconception: "+ data["new_feedbacks"][j]["Misconception"] + "\nExplanation: " + data["new_feedbacks"][j]["Explanation"] }]] )

        else:
            conversations = [data]
        choice = np.random.choice(len(self.instructions))

        text_input = self.instructions[choice].format(answer=data["answers"] , question=data["question"], distractor=data["new_distractors"][j])
        # text_input = ""

        query_input = self.instruct_blip_tokenizer(
            text_input,
            return_tensors="pt",
            padding="longest",
            max_length=200,
            truncation=True,
        )
        
        data_dict = self.token_processor(conversations)
        data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])
        data_dict['query_input_ids'] = query_input["input_ids"][0]
        data_dict['query_attention_mask'] = query_input["attention_mask"][0]
        
        if isinstance(data, dict) and 'img_name' in data:
            data_dict['image'] = [image]
            data_dict['image_high'] = [image_1]
        else:
            crop_size = self.multimodal_cfg['image_processor'].crop_size
            data_dict['image'] = [torch.zeros(3, crop_size['height'], crop_size['width'])]
            data_dict['image_high'] = [torch.zeros(3, 1024, 1024)]
        return data_dict
