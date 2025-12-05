from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, Trainer, TrainingArguments
from trl import PPOConfig, PPOTrainer
from datasets import Dataset

import argparse


def main(params) -> None:
    if params.reward_dir is None:
        train_reward_model("./model", "./reward_dataset", params.out_dir + "reward")

    run_ppo(params.model_dir, params.out_dir + "reward", params.dataset, params.out_dir, int(params.epochs))

def run_ppo(p_model, r_model, dataset, out_dir, epochs):
    tokenizer = AutoTokenizer.from_pretrained(p_model)

    policy = AutoModelForCausalLM.from_pretrained(p_model)
    reward = AutoModelForSequenceClassification.from_pretrained(r_model)

    ppo_config = PPOConfig(
            model_adapter_name=p_model,
            learning_rate=1e-4,
            batch_size=16,
            num_ppo_epochs=epochs
    )
    ppo_trainer = PPOTrainer(
            args = ppo_config,
            model=policy,
            reward_model=reward,
            ref_model=None,
            train_dataset=dataset,
            value_model=policy,
            processing_class=tokenizer,
    )
    ppo_trainer.train()


def train_reward_model(train_dir: str, dataset, out_dir):
    tokenizer = AutoTokenizer.from_pretrained(train_dir)

    ds = Dataset.from_list(dataset)
    def map_func(x):
        return tokenizer(ds["inputs"], truncation=True, padding=True)
    final_ds = ds.map(map_func, batched=True)

    model = AutoModelForCausalLM.from_pretrained(train_dir)

    train_args = TrainingArguments(output_dir=out_dir, per_device_train_batch_size=16, num_train_epochs=100)
    trainer = Trainer(model=model, args=train_args, train_dataset=final_ds)
    trainer.train()
    trainer.save_model(out_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("__model_dir", default="./Qwen2.5-VL-7B-Instruct", help="大模型路径")
    parser.add_argument("__reward_dir", default="./reward", help="奖励模型路径")
    parser.add_argument("--epochs", default=100, help="PPO训练的轮次")
    parser.add_argument("--out_dir", default="./output_dir", help="训练后的大模型")

    args = parser.parse_args()

    main(args)