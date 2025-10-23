#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import torch
from transformers import AutoModel, AutoTokenizer
from transformers import LlamaTokenizer, LlamaModel

# 设为全局变量，避免每次调用都要重复加载
_TOKENIZER = None
_MODEL = None
_DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

##########################################################
# 占位函数：调用 LLaMA3-8B 获取文本embedding
# 真实项目中可替换为具体的调用，如 huggingface/transformers 或 其余API
##########################################################
def _init_llama3_8b_model(model_name="/data/zxlei/checkpoints_repository/llm/Llama-3.1-8B"):
    """
    内部函数: 初始化 tokenizer 和 model, 存到全局变量。
    如果已经初始化过，将直接返回。
    """
    global _TOKENIZER, _MODEL
    if _TOKENIZER is None or _MODEL is None:
        print(f"Loading tokenizer and model: {model_name} ...")
        # _TOKENIZER = LlamaTokenizer.from_pretrained(model_name)
        _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        # 如果环境支持GPU，可指定 device_map="auto" 等；没有GPU则在CPU上加载
        _MODEL = AutoModel.from_pretrained(model_name, device_map="balanced_low_0")
        _MODEL.eval()

def llama3_8b_encode(text: str) -> torch.Tensor:
    """
    调用 LLaMA-3.1-8B 模型，对输入文本进行编码，返回一个 1 x hidden_dim 的向量。
    这里采用简单的“最后隐层取平均”来得到句子级表示。

    Args:
        text (str): 待编码的文本
    Returns:
        embedding (torch.Tensor): shape [hidden_dim], 即文本的向量表示
    """
    # 确保初始化
    _init_llama3_8b_model()

    # 分词, 得到输入张量
    encoded = _TOKENIZER(text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(_DEVICE)
    attention_mask = encoded["attention_mask"].to(_DEVICE)

    with torch.no_grad():
        outputs = _MODEL(input_ids=input_ids, attention_mask=attention_mask)
        # outputs.last_hidden_state: [batch_size, seq_len, hidden_size]
        last_hidden = outputs.last_hidden_state  # B x T x D

        # 简单平均池化: 对序列维度(seq_len)做 mean
        # 注意：也可以只取 [CLS] 或者其他方式
        # 这里 batch_size=1，所以直接 squeeze(0) 变成 [T, D] 再 mean(0) => [D]
        embedding = last_hidden.squeeze(0).mean(dim=0)

    return embedding


def extract_subgoals_from_bddl(bddl_file: str):
    """
    读取 bddl 文件，解析所有 (:subgoal...) 块，并提取子目标信息。
    返回字典 { subgoalN: {"instruction": "...", "embedding": tensor(...) } }
    """
    with open(bddl_file, "r", encoding="utf-8") as f:
        text = f.read()

    # 使用正则，匹配所有 (:subgoalN ... ) 块（包含跨行）。
    # DOTALL 允许 . 匹配换行
    pattern = r"\(:subgoal\d+.*?\)"
    matches = re.findall(pattern, text, flags=re.DOTALL)

    subgoal_data = {}

    for match in matches:
        # match 示例:
        # (:subgoal10 Grasp task: Grasp legal_document.n.01_3.
        #    (isgrasping ?agent.n.01_1 ?legal_document.n.01_3)
        # )
        # 1) 提取 subgoal 编号
        subgoal_id_match = re.search(r"\(:subgoal(\d+)", match)
        if not subgoal_id_match:
            continue
        subgoal_id = subgoal_id_match.group(1)  # e.g. "10"

        # 2) 提取指令行，如 "Grasp task: Grasp legal_document.n.01_3."
        #   这里有多种策略，最简单的：抓取 `:subgoalN` 后面的整行（或直到换行处）
        #   也可以用正则更严格匹配 "Navigation task: ...." 这类文字。
        #   示例：在 (:subgoalX ... ) 中找 "task: ...\.)"

        instr_match = re.search(r'(\w+\s+task:\s+[\w\s\._]+)\.', match)
        if instr_match:
            instruction_line = instr_match.group(1).strip()
        else:
            # 如果没匹配到，兜底也可以直接从 :subgoal 行截一部分
            fallback_pat = r"\:subgoal\d+\s+(.*)"
            fallback_match = re.search(fallback_pat, match)
            if fallback_match:
                instruction_line = fallback_match.group(1).strip()
                # 去掉可能的后续括号部分
                instruction_line = instruction_line.split("(")[0].strip()

        # 3) 去除 .n.01_x 这样的后缀
        #    将 "legal_document.n.01_3" -> "legal_document"
        #    例如 "Grasp task: Grasp legal_document.n.01_3." -> "Grasp task: Grasp legal_document."
        # cleaned_instruction = re.sub(r"\.n\.01_\d+", "", instruction_line)
        cleaned_instruction = instruction_line

        # 进一步可去掉多余空白
        cleaned_instruction = cleaned_instruction.replace("\n", " ")
        cleaned_instruction = cleaned_instruction.strip()

        # 4) 调用LLAMA3-8B获取embedding
        embedding_tensor = llama3_8b_encode(cleaned_instruction).to("cpu")

        # 5) 存入字典
        key = f"subgoal{subgoal_id}"
        subgoal_data[key] = {
            "instruction": cleaned_instruction,
            # tensor无法直接json序列化，稍后转成list或其他形式再dump
            "embedding": embedding_tensor,  # 转成 python list
        }

    return subgoal_data


def main():
    # 假定 BDDL 文件路径固定，也可通过sys.argv参数传入
    bddl_file = "/data/zxlei/embodied/embodied-bench/bddl_subgoals/bddl_subgoals/activity_definitions/recycling_office_papers/problem0.bddl"

    if not os.path.exists(bddl_file):
        print(f"BDDL file not found: {bddl_file}")
        return

    # 解析subgoal
    subgoals = extract_subgoals_from_bddl(bddl_file)

    # 构造输出路径，与 bddl 同级目录
    out_dir = os.path.dirname(bddl_file)
    out_file = os.path.join(out_dir, "problem0_subgoals.pt")

    # 保存为pt文件：
    torch.save(subgoals, out_file)

    print(f"Subgoal data saved to {out_file}")


if __name__ == "__main__":
    main()
