# 首先将json格式数据转换成标注格式jsonl格式文件（此处转换需要将json格式中的key值设置成相应的key值），然后将jsonl格式的文件转换成tokenizer格式文件
# json文件下载
# https://huggingface.co/datasets/whalning/Chinese-medical-QA

import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer, \
    DataCollatorForLanguageModeling
from .dataset_loader import load_tokenized_dataset
import torch


class LORASFTtrainModel(object):
    def __init__(self, sftcfg, loracfg):
        # === Load Configs ===
        with open(sftcfg, "r", encoding="utf-8") as f:
            self.sft_cfg = yaml.safe_load(f)
        with open(loracfg, "r", encoding="utf-8") as f:
            self.lora_cfg = yaml.safe_load(f)

        # === Load Model + LoRA ===
        self.model = AutoModelForCausalLM.from_pretrained(self.sft_cfg["model_name_or_path"], device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.sft_cfg["model_name_or_path"], trust_remote_code=True)
        self.peft_config = LoraConfig(**self.lora_cfg)
        self.model = get_peft_model(self.model, self.peft_config)

    def Update(self):
        # === Load Dataset ===
        train_dataset = load_tokenized_dataset(self.sft_cfg["model_name_or_path"], self.sft_cfg["train_data_path"])
        eval_dataset = load_tokenized_dataset(self.sft_cfg["model_name_or_path"], self.sft_cfg["eval_data_path"])
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        # === Training Args ===
        args = TrainingArguments(
            output_dir=self.sft_cfg["output_dir"],
            per_device_train_batch_size=int(self.sft_cfg["per_device_train_batch_size"]),
            num_train_epochs=int(self.sft_cfg["num_train_epochs"]),
            learning_rate=float(self.sft_cfg["learning_rate"]),
            logging_steps=int(self.sft_cfg["logging_steps"]),
            save_strategy=self.sft_cfg["save_strategy"],
            report_to=["tensorboard"],
        )
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        trainer.train()
        self.model.save_pretrained(self.sft_cfg["output_dir"])


class LORASFTinferenceModel(object):
    def __init__(self, model_dir):
        # 1️⃣ 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        # 2️⃣ 加载模型（自动选择 GPU/CPU，使用半精度加速）
        self.model = AutoModelForCausalLM.from_pretrained(model_dir,
                                                          torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                                                          device_map="auto", trust_remote_code=True)
        self.model.eval()

    def infernece(self, prompt):
        # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)#会把输入也解码出来
        # 3️⃣ 构造输入（Qwen3 推荐使用 apply_chat_template）
        # 如果是多轮对话，可传入 messages=[{"role":"user","content":prompt}]
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt").to(self.model.device)
        attention_mask = (inputs != self.tokenizer.pad_token_id).to(self.model.device)
        # 4️⃣ 生成输出（加上 temperature、top_p 控制生成多样性）
        outputs = self.model.generate(inputs, attention_mask=attention_mask, max_new_tokens=1024,
                                      temperature=0.7, top_p=0.9, do_sample=True,
                                      pad_token_id=self.tokenizer.eos_token_id)
        # 5️⃣ 解码输出（取生成的新内容部分）
        response = self.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        print("🩺 模型回复：", response)
        return response

