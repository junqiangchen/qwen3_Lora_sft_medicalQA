import json

# 输入和输出文件路径
input_path = r"D:\cjq\project\python\qwen3_Lora_sft_project\data\medical_eval.json"  # 原始 JSON 文件（带 [ ]）
output_path = r"D:\cjq\project\python\qwen3_Lora_sft_project\data\medical_eval.jsonl"  # 目标 JSONL 文件（一行一个对象）


def to_str(value):
    """安全地将值转换为字符串"""
    if isinstance(value, list):
        # 如果是列表，拼接为一个字符串（用空格连接）
        return " ".join(map(str, value))
    elif isinstance(value, dict):
        # 如果是字典，可以转成 JSON 字符串
        return json.dumps(value, ensure_ascii=False)
    elif value is None:
        return ""
    else:
        return str(value)


# 读取整个 JSON 文件
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(output_path, "w", encoding="utf-8") as f:
    for item in data:
        # ✅ 修改 key 名
        new_item = {
            "input": to_str(item.get("patient_question", "")),  # 把 patient_question 改成 input
            "output": to_str(item.get("doctor_answer", "")),  # 把 doctor_answer 改成 output
        }
        json_line = json.dumps(new_item, ensure_ascii=False)
        f.write(json_line + "\n")

print(f"✅ 已成功转换并修改 key 为 JSONL：{output_path}")
