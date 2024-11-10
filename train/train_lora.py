import numpy as np
import logging
import pathlib
import torch
import transformers
from peft import LoraConfig, get_peft_config, get_peft_model


from peifg.train.trainer_vit_fixlr import peifgTrainer
from peifg.model import *
from peifg.data import make_supervised_data_module
from peifg.utils.arguments import *
from peifg.utils.utils import smart_tokenizer_and_embedding_resize
from peifg.model.vision_encoder.sam import build_sam_vit_b
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# device = torch.device("cuda:7")




def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, padding_side="right", model_max_length=training_args.model_max_length)

    model = peifgQwenForCausalLM.from_pretrained(model_args.model_name_or_path, low_cpu_mem_usage=True)



    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token='<|endoftext|>'),
        tokenizer=tokenizer,
        model=model,
        )

    dtype = torch.float16
    if training_args.fp16:
        dtype = torch.float16
    if training_args.bf16:
        dtype = torch.bfloat16

    vision_tower_dict = model.get_model().initialize_vision_modules(
        vision_tower=model_args.vision_tower,
        pretrained_stage1_model=model_args.pretrained_stage1_model,
        freeze_vision_tower=model_args.freeze_vision_tower,
        use_im_start_end=model_args.use_im_start_end,
        vision_select_layer=model_args.vision_select_layer,
        dtype=dtype,
        device=training_args.device,
        is_train=True
    )

    model.initialize_vision_tokenizer(
        tokenizer=tokenizer, 
        freeze_lm_model=model_args.freeze_lm_model, 
        pretrained_stage1_model=model_args.pretrained_stage1_model,
        device=training_args.device,
    )




    model.to(dtype=dtype, device=training_args.device)
    model.enable_input_require_grads()
    
    peft_config = LoraConfig(
            # target_modules=r'.*language_model.*\.(q_proj|v_proj)', 
            target_modules=r'.*\.(c_proj|c_attn)', 
            inference_mode=False, 
            r=training_args.lora_r, 
            lora_alpha=training_args.lora_alpha, 
            lora_dropout=training_args.lora_dropout
        )
    
    model = get_peft_model(model, peft_config)


    # setting training param required grad
    for name, param in model.named_parameters():
        if "lora" in name or "img_projector" in name or "expert" in name or "query_tokens" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # print leanable parameters
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print(name)
    
    model.print_trainable_parameters()

    data_args.image_token_len = 256
    data_args.image_processor = vision_tower_dict['image_processor']
    data_args.image_processor_high = vision_tower_dict['image_processor_high']
    data_args.use_im_start_end = model_args.use_im_start_end


                
    params_grad = [p.numel() for n, p in model.named_parameters() if p.requires_grad]
    print(f"Number of Mapping Trainable Parameters: {sum(params_grad) / (1 << 20):.2f} M")
  

    data_module = make_supervised_data_module(
        interleave=training_args.interleave, 
        with_box=training_args.with_box, 
        tokenizer=tokenizer, 
        data_args=data_args
    )

    trainer = peifgTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        # trainer.train(resume_from_checkpoint=True)
        trainer.train()
    else:
        trainer.train()
    trainer.save_state()
    trainer._safe_save(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
