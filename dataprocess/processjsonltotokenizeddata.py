import json
from datasets import Dataset


def load_medical_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data


def processjsontotokenized_dataset(train_path, eval_path, output_path):
    train_data = load_medical_data(train_path)
    eval_data = load_medical_data(eval_path)
    train_dataset = Dataset.from_list(train_data)
    valid_dataset = Dataset.from_list(eval_data)
    train_dataset.save_to_disk(f"{output_path}/train")
    valid_dataset.save_to_disk(f"{output_path}/eval")


if __name__ == "__main__":
    processjsontotokenized_dataset(r"D:\cjq\project\python\qwen3_Lora_sft_project\data/medical_train.jsonl",
                                   r"D:\cjq\project\python\qwen3_Lora_sft_project\data/medical_eval.jsonl",
                                   r"D:\cjq\project\python\qwen3_Lora_sft_project\data\tokenized")
