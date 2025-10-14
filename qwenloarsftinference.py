import os
from model import LORASFTtrainModel, LORASFTinferenceModel

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def lorasfttrain():
    sftcfg = r"D:\cjq\project\python\qwen3_Lora_sft_project\config\sft_config.yaml"
    loracfg = r"D:\cjq\project\python\qwen3_Lora_sft_project\config\lora_config.yaml"
    loarasft = LORASFTtrainModel(sftcfg, loracfg)
    loarasft.Update()


def lorasftinfer():
    model_dir = r""
    lorasft_infer = LORASFTinferenceModel(model_dir)
    input_promot = "CT 报告写：肝右叶有 1.2 cm 强化灶，建议进一步检查，我该怎么做？"
    lorasft_infer.infernece(input_promot)


if __name__ == "__main__":
    lorasfttrain()
    # lorasftinfer()
