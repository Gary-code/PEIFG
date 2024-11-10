import logging
import pathlib
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_config, get_peft_model,PeftModel


from peifg.model import *
from peifg.data import make_supervised_data_module
from peifg.utils.arguments import *
from peifg.utils.utils import smart_tokenizer_and_embedding_resize
from peifg.model.vision_encoder.sam import build_sam_vit_b


base_model = peifgQwenForCausalLM.from_pretrained(
        f'peifg/',
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda"},
)
lora_model = PeftModel.from_pretrained(
        base_model,
        f'peifg-feedback/',
        device_map={"": "cuda"},
        torch_dtype=torch.bfloat16,
)

model = lora_model.merge_and_unload()

lora_model.train(False)

tokenizer = AutoTokenizer.from_pretrained('peifg/',trust_remote_code=True)
model.save_pretrained(f"peifg/new")
tokenizer.save_pretrained(f"peifg/new")
