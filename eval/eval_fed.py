import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import numpy as np
from peifg.utils.conversation import conv_templates, SeparatorStyle
from peifg.utils.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from peifg.model import *
from peifg.utils.utils import KeywordsStoppingCriteria

from PIL import Image

import os
import json
from tqdm import tqdm
import requests
from PIL import Image
from io import BytesIO
from peifg.model.plug.blip_process import BlipImageEvalProcessor
from transformers import TextStreamer
from peifg.model.plug.transforms import train_transform, test_transform

device = torch.device("cuda:0")
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'

DEFAULT_EXPERT_PAD_TOKEN = '<ref>'
DEFAULT_EXPERT_START_TOKEN = '<quad>'
DEFAULT_EXPERT_END_TOKEN = '</quad>'
DEFAULT_EXPERT_TOKEN = '<expert>'


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def eval_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    instruct_blip_tokenizer = AutoTokenizer.from_pretrained("./pre-trained/instructblip-flan-t5-xl/qformer_tokenizer", trust_remote_code=True, padding_side="right", max_length=150)
    model = peifgQwenForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, device_map=device, trust_remote_code=False)


    model.to(device=device,  dtype=torch.bfloat16)
    model.get_model().initialize_vision_modules(
        vision_tower='clip-vit-large-patch14',
        freeze_vision_tower=True,
        use_im_start_end=True,
        vision_select_layer=-2,
        dtype=torch.bfloat16,
        device='cuda'
    )
    image_processor = CLIPImageProcessor.from_pretrained("./pre-trained/clip-vit-large-patch14", torch_dtype=torch.bfloat16)

    image_processor_high = BlipImageEvalProcessor(image_size=1024)

    use_im_start_end = True

    image_token_len = 256
    expert_token_len = 15
    # expert_token_len = 0

    with open(args.dataset, "r") as file:
        dataset = json.load(file)
    with open("./instructions.json", "r") as file:
        instructions = json.load(file)

    references = []
    predictions = []
    
    for data in tqdm(dataset, desc="process"):
        for i in range(3):
            qs = "question: " + data['question'] + "\nanswer: " + data['answers'] + "\ndistractor: " + data['new_distractors'][i]
            # TODO no instructions
            choice = np.random.choice(len(instructions))
            text_input = instructions[choice].format(answer=data["answers"] , question=data["question"], distractor=data['new_distractors'][i])

            if use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN*image_token_len + DEFAULT_IM_END_TOKEN  + ' ' + \
                    DEFAULT_EXPERT_START_TOKEN + DEFAULT_EXPERT_PAD_TOKEN * expert_token_len + DEFAULT_EXPERT_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs


            

            conv_mode = "opt"
            args.conv_mode = conv_mode

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            inputs = tokenizer([prompt])
            query_input = instruct_blip_tokenizer([text_input]) # add query_input_id


            image = load_image(args.image_file + data['img_name'])
            
            image_1 = image.copy()
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            image_tensor_1 = image_processor_high(image_1)

            input_ids = torch.as_tensor(inputs.input_ids).cuda(device)
            # TODO add query_input_ids and query_attentation_mask
            query_input_ids = torch.as_tensor(query_input.input_ids).cuda(device)
            query_attention_mask = torch.as_tensor(query_input.attention_mask).cuda(device)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

                
            try:
                output_ids = model.generate(
                    input_ids,
                    images=[(image_tensor.unsqueeze(0).to(device=device, dtype=torch.bfloat16), image_tensor_1.unsqueeze(0).to(device=device, dtype=torch.bfloat16))],
                    query_input_ids=query_input_ids,
                    # query_attention_mask=query_attention_mask,
                    do_sample=True,
                    # top_k=0,
                    temperature=0.8,
                    top_p=0.95,
                    # remove_invalid_values=True,
                    # temperature=0.2,
                    # streamer=streamer,
                    max_new_tokens=70,
                    stopping_criteria=[stopping_criteria]
                    )
                outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

            except KeyError:
                print("error!!!")
                outputs = "error!!!"
            
            # conv.messages[-1][-1] = outputs
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            print(outputs)
            predictions.append(outputs)
    return  predictions


if __name__ == "__main__":
    step = 2400
    for i in range(1):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model-name", type=str, default=f"./peifg-feedback")
        parser.add_argument("--image-file", type=str, default="./vcr1images/")
        parser.add_argument("--conv-mode", type=str, default=None)
        parser.add_argument("--dataset", type=str, default="./dataset/test.json")
        args = parser.parse_args()

        predictions = eval_model(args)
        predictions_folder_path = f"./peifg-feedback/path"
        def create_folder_if_not_exists(folder_path):
            # 检查文件夹是否存在
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                print(f"'{folder_path}' success")
            else:
                print(f"'{folder_path}' fail")
        create_folder_if_not_exists(os.path.dirname(predictions_folder_path))
        
        with open(predictions_folder_path, 'w') as jsonl_file:
            json.dump(predictions, jsonl_file)
            jsonl_file.write('\n')
