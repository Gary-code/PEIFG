
import torch
import transformers
from dataclasses import dataclass, field
from peifg.data.dataset_pocess import MyDataset, MyDataset_Feedback, MyDataset_Inference
from peifg.utils.constants import *


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    instruct_tokenizer = transformers.AutoTokenizer.from_pretrained("./pre-trained/instructblip-flan-t5-xl/qformer_tokenizer", trust_remote_code=True, padding_side="right", max_length=150)
    def __call__(self, instances):
        
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        images = [torch.stack(instance['image']) for instance in instances]
        query_input_ids, query_attention_mask = tuple([instance[key] for instance in instances] for key in ("query_input_ids", "query_attention_mask"))
        images_high = [torch.stack(instance['image_high']) for instance in instances]

        images = list(zip(images, images_high))


        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
            
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        
        query_input_ids = torch.nn.utils.rnn.pad_sequence(
            query_input_ids,
            batch_first=True,
            padding_value=self.instruct_tokenizer.pad_token_id)
        
        query_attention_mask = torch.nn.utils.rnn.pad_sequence(
            query_attention_mask,
            batch_first=True,
            padding_value=self.instruct_tokenizer.pad_token_id)
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            images=images,
        )
        return batch


def make_supervised_data_module(interleave, with_box, tokenizer, data_args):


    if data_args.conversation_version == 'opt':
        dataset_cls = MyDataset_Feedback


    train_dataset = dataset_cls(
        tokenizer=tokenizer,
        datasets=data_args.datasets,
        multimodal_cfg=dict(
            sep_image_conv_front=data_args.sep_image_conv_front,
            expert_token_len=data_args.expert_token_len,
            image_token_len=data_args.image_token_len,
            image_aspect_ratio=data_args.image_aspect_ratio,
            use_im_start_end=data_args.use_im_start_end,
            image_processor=data_args.image_processor,
            image_processor_high = data_args.image_processor_high,
            box_limit=data_args.box_limit,
        )
    )
    eval_dataset = dataset_cls(
        tokenizer=tokenizer,
        datasets=data_args.datasets_eval,
        multimodal_cfg=dict(
            sep_image_conv_front=data_args.sep_image_conv_front,
            expert_token_len=data_args.expert_token_len,
            image_token_len=data_args.image_token_len,
            image_aspect_ratio=data_args.image_aspect_ratio,
            use_im_start_end=data_args.use_im_start_end,
            image_processor=data_args.image_processor,
            image_processor_high = data_args.image_processor_high,
            box_limit=data_args.box_limit,
        )
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)