from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset

import argparse
import os


def main(params) -> None:
    # 1. 设置tokenizer
    tokenizer = AutoTokenizer.from_pretrained(params.model_dir)

    # 2. 构建数据集
    if os.path.isfile(params.dataset):
        ds = load_dataset("text", data_files={"train": params.dataset})  # 支持本地的txt文件形式
    else:
        ds = load_dataset(params.dataset)

    def tokenize_func(x):
        """
        要传入datasets.map()函数里的函数，代表了我们要对输入实例x做的变化。
        在这里只有text部分需要被tokenize。
        """
        return tokenizer(x["text"], return_special_tokens_mask=True)

    # batched=True保证了ds真正被构建成类似于dataloader的形式
    tokenized_ds = ds.map(tokenize_func, batched=True, remove_columns=ds["train"].column_names)

    def group_func(x):
        """
        也是要传入map里的函数，这里把所有训练文本合并气起来再按照seq_len切分，这样使得不用在训练阶段补那么多0。
        另外因为训练的是next token的预测所以label只需要向后移一位即可。
        """
        concat = {k : sum(x, []) for k in x.keys()}
        total_len = len(concat[list(x.keys())[0]])
        if total_len > int(params.seq_len):
            total_len = (total_len // int(params.seq_len)) * int(params.seq_len)
        result = {
                k: [t[i : i + params.seq_len] for i in range(0, total_len, params.seq_len)]
                for k, t in concat.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    final_ds = tokenized_ds.map(group_func, batched=True)

    # 3. 创建模型
    model = AutoModelForCausalLM.from_pretrained(params.model_dir)

    # 4. 创建datacollector 这个相当于dataloader 反正是trainer需要的对dataset的封装
    datacollector = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)

    # 5. 构建trainer开始训练
    train_args = TrainingArguments(
            output_dir=params.output_dir,
            per_device_train_batch_size=params.batch_size,
            num_train_epochs=params.epochs,
            logging_steps=10,
            save_total_limit=2,
            fp16=False,
            gradient_accumulation_steps=1
    )

    trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=final_ds,
            data_collator=datacollector
    )
    trainer.train()
    trainer.save_model(params.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", default="gpt2", help="模型目录")
    parser.add_argument("--dataset", required=True, help="数据集")
    parser.add_argument("--output_dir", default="./pretrained", help="保存目录")
    parser.add_argument("--seq_len", default=1024, help="输入序列的长度")
    parser.add_argument("--batch_size", default=16, help="batch大小")
    parser.add_argument("--epochs", default=100, help="训练轮次")

    args = parser.parse_args()

    main(args)