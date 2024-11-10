from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-350m")
    use_cache: bool = field(default=False)
    vision_tower: Optional[str] = field(default="./pre-trained/clip-vit-large-patch14")
    freeze_vision_tower: bool = field(default=False)
    freeze_lm_model: bool = field(default=False)
    pretrained_stage1_model: Optional[str] = field(default=None) # mlp &/ vision tower
    vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    use_im_start_end: bool = field(default=False)


@dataclass
class DataArguments:
    datasets: str = field(default=None, metadata={"help": "combinations of the training data."})
    datasets_eval: str = field(default=None, metadata={"help": "combinations of the eval data."})
    sep_image_conv_front: bool = False
    image_token_len: int = 256
    expert_token_len: int = 15
    image_aspect_ratio: str = 'square'
    conversation_version: str = 'mpt'  
    box_limit: int = 0


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    interleave: bool = field(default=False)
    with_box: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    lora_enable: bool = True
    lora_r: int = 8
    lora_alpha: int = 256
    lora_dropout: float = 0.5
    lora_weight_path: str = ""
    lora_bias: str = "none"
    do_eval: bool = False
