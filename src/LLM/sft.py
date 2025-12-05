from transformers import AutoTokenizer, AutoModelForVision2Seq, Trainer, TrainingArguments

import datasets
import argparse


def main(params) -> None:
    # 1. tokenizer初始化
    tokenizer = AutoTokenizer.from_pretrained(params.model_dir)

    # 2. 处理dataset 一般dataset为jsonl格式 instruction fine-tuning
    ds = datasets.load_dataset(params.dataset)

    def fit_func(x):
        """
        给prompt的部分盖上-100的帽子
        """
        inputs = []
        labels = []
        for p, r in zip(x["prompt"], x["response"]):
            full = p + "\n" + r
            tokenized_ds = tokenizer(full, truncation=True, max_length=int(params.seq_len))
            input_ids = tokenized_ds["input_ids"]

            prompt_ds = tokenizer(p, truncation=True, max_lenght=int(params.seq_len))
            prompt_len = len(prompt_ds["input_ids"])

            labels_ids = [-100] * prompt_len + input_ids[prompt_len:]
            labels_ids = labels_ids[:int(params.seq_len)]
            if len(labels_ids) < int(params.seq_len):
                labels_ids += [-100] * (int(params.seq_len) - len(labels_ids))

            inputs.append(input_ids)
            labels.append(labels_ids)

        return {'input_ids': inputs, 'labels': labels, 'attention_mask': [[1 if id != tokenizer.pad_token_id else 0 for id in seq] for seq in inputs]}

    final_ds = ds.map(fit_func, batched=True)

    # 3. 初始化模型并训练
    model = AutoModelForVision2Seq.from_pretrained(params.model_dir)

    training_args = TrainingArguments(
            output_dir=params.output_dir,
            per_device_train_batch_size=2,
            num_train_epochs=1,
            logging_steps=10,
            save_total_limit=2,
            fp16=False,
    )

    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=final_ds,
    )

    trainer.train()
    trainer.save_model(params.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", default="./Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset", default="./sft_dataset")
    parser.add_argument("--seq_len", default=512)
    parser.add_argument("--output_dir", default="./sft")

    args = parser.parse_args()

    main(args)
