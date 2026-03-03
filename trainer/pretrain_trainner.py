# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

torch.backends.cuda.enable_cudnn_sdp(False)
from transformers import AutoTokenizer, TrainingArguments,Trainer
from datasets import load_dataset

from model import BedRockV3Config,BedRockV3ForCausalLM
from dataset import PretrainDataset, PretrainDataCollator

os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "0"
if __name__ == '__main__':
    
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/home/jx/experiment/BedRock-llm/tokenizer/tokenizer_modify")

    # model
    config = BedRockV3Config()
    model = BedRockV3ForCausalLM(config)


    print("Model parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad)/10**9)

    # dataset
    chinese_c4 = load_dataset("/mnt/sdb1/lijiaxin/Chinese-c4", split="train")
    chinese_c4_train_test = chinese_c4.train_test_split(test_size=0.004, seed=42, shuffle=True)
    chinese_c4_train_test['val'] = chinese_c4_train_test.pop('test')

    train_dataset = PretrainDataset(chinese_c4_train_test['train'], tokenizer)
    val_dataset = PretrainDataset(chinese_c4_train_test['val'], tokenizer)

    # Train Arguments
    training_args = TrainingArguments(
        output_dir="./result",
        # overwrite_output_dir=True, # 每次清空输出目录
        seed=42,
        # 学习率
        learning_rate=1e-4,       # 对小型 LLM 可用，若模型更大可调小到 1e-4 或 5e-5
        lr_scheduler_type="cosine",
        warmup_steps=1000,             # warmup_steps=0 表示不使用 warmup
        weight_decay=0.01,        # Transformer 预训练常用

        # Batch size 设置
        per_device_train_batch_size=24,   # 每张 GPU micro batch，4090 24GB 适合 2~4
        gradient_accumulation_steps=1,   # 累积梯度，使 effective batch size = 2*8*2 = 32
                                        # 这个值可根据显存和模型调大或调小
        per_device_eval_batch_size=4,    # eval batch 也小点，避免显存爆掉

        # 总 batch size
        # effective batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus

        num_train_epochs=1,           # 根据 dataset 调整
        max_grad_norm=1.0,           # 防止梯度爆炸
        adam_beta1=0.9,
        adam_beta2=0.95,

        # 评估和保存
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        load_best_model_at_end=True,
        logging_steps=100,
        eval_accumulation_steps=1, # eval 累积梯度，避免显存爆掉

        # 精度设置
        fp16=True,                    # mixed precision，4090 支持 Tensor Cores
        bf16=False,                   # 不使用 bf16

        # --- DeepSpeed 核心配置 ---
        deepspeed="/home/jx/experiment/BedRock-llm/trainer/ds_config_zero2.json", 

        torch_compile=False, # 与deepspeed互斥
        # torch_compile_options={"dynamic": True},  # 启用动态形状，允许 sequence length 变化

        # 其他
        prediction_loss_only=True, # 只返回 loss，不返回其他指标
        save_total_limit=5,
        dataloader_num_workers=4,
        report_to="tensorboard",
        ddp_find_unused_parameters=False,

    )

    # data_collator
    data_collator = PretrainDataCollator()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    trainer.train(resume_from_checkpoint=True) 