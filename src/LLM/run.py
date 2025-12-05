from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
from argparse import ArgumentParser

import requests


def main(params) -> None:
    # 1. 加载模型
    model_dir = params.model_dir  # 一般符合hf的模型直接把目录放在这里就可以了

    model = AutoModelForVision2Seq.from_pretrained(model_dir, dtype="auto", device_map="auto")  # auto比较推荐，也可以自己指定

    # 2. 加载模型处理器
    processor = AutoProcessor.from_pretrained(model_dir)
    #    - AutoTokenizer: 如果只是纯文本模型 (如 BERT, GPT)，只需要 Tokenizer。
    #    - AutoImageProcessor: 如果只是纯视觉模型 (如 ViT)，只需要 ImageProcessor。
    #    - AutoProcessor: 多模态模型通常包含一个 Tokenizer 和一个 ImageProcessor。AutoProcessor 是一个包装器，把这两者结合在一起，方便同时处理文本和图像输入。

    # 3. 构造输入消息
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "描述一下这张图片。"},
            ],
        }
    ]  # 这个message的格式和模板会提供在config.json中的chat_template中
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image = Image.open(requests.get("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg", stream=True).raw)

    inputs = processor(
            text=[text],
            images=image,
            padding=True,  # 补齐token至与整个batch中最长的seq的token一致
            return_tensors="pt"
    ).to("cuda")  # 这玩意要放到GPU上

    # 4. 推理生成
    output_all = model.generate(**inputs, max_new_tokens=128)  # reutrn: list[Tensor], Tensor.shape: [batch_size, seq_len], 其中包含了: 输入 + 输出
    output_answer = [opt_all[len(ips):] for ips, opt_all in zip(inputs, output_all)]
    output_text = processor.batch_decode(output_answer, skip_special_token=True, clean_up_tokenization_spaces=False)  # 一般都这样设置就可以 没有影响
    print(output_text[0])


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--model_dir", default="./Qwen2.5-VL-7B-Instruct", help="模型目录")

    args = parser.parse_args()

    main(args)




