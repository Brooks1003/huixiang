#!/usr/bin/env python3
"""
回响 — AI分身训练流水线
输入：用户上传的微信聊天记录（txt格式）
输出：微调后的对话模型（GGUF格式）

使用方法：
  1. 用户导出微信聊天记录为 .txt
  2. python3 train_huixiang.py --input chat.txt --name "TA的名字"
  3. 等待训练完成 → 模型上线

最低硬件要求：Apple Silicon M1+ 16GB / NVIDIA 8GB VRAM
"""
import argparse, json, re, os, sys
from pathlib import Path
from typing import List, Dict, Tuple

# ============================================================
# 第一步：解析微信聊天记录
# ============================================================

def parse_wechat_chat(filepath: str) -> List[Dict]:
    """
    解析微信导出的聊天记录。
    
    常见格式：
    2024-01-15 22:14:32 张三
    你今天吃饭了吗
    
    2024-01-15 22:15:01 李四
    还没，加班呢
    
    也支持：
    [2024年1月15日 22:14] 张三: 你今天吃饭了吗
    2024/01/15 22:14 张三：你今天吃饭了吗
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    
    messages = []
    
    # 格式1: "YYYY-MM-DD HH:MM:SS 用户名\n内容"
    pattern1 = re.compile(
        r'(\d{4}[-/]\d{2}[-/]\d{2})\s+(\d{1,2}:\d{2}(?::\d{2})?)\s+(\S+?)(?:\s*$|\n)',
        re.MULTILINE
    )
    
    # 格式2: "[YYYY年M月D日 HH:MM] 用户名: 内容"
    pattern2 = re.compile(
        r'\[?(\d{4})[年/-](\d{1,2})[月/-](\d{1,2})[日]?\s+(\d{1,2}:\d{2})\]?\s*(\S+?)[：:]\s*(.+)',
        re.MULTILINE
    )
    
    # 先试格式1
    matches = list(pattern1.finditer(raw))
    if matches:
        for i, m in enumerate(matches):
            user = m.group(3).strip()
            # 内容 = 从当前消息到下一个消息之间
            start = m.end()
            end = matches[i+1].start() if i+1 < len(matches) else len(raw)
            content = raw[start:end].strip()
            if content:
                messages.append({"user": user, "content": content})
    else:
        # 试格式2
        for m in pattern2.finditer(raw):
            user = m.group(5).strip()
            content = m.group(6).strip()
            if content:
                messages.append({"user": user, "content": content})
    
    print(f"📊 解析完成: {len(messages)} 条消息")
    return messages


# ============================================================
# 第二步：提取对话配对（用户 ↔ TA）
# ============================================================

def extract_conversations(messages: List[Dict], target_name: str) -> List[Dict]:
    """
    从消息流中提取对话配对。
    格式：用户说一句话 → TA回复一句话
    
    返回: [{"user_msg": "你吃饭了吗", "assistant_msg": "还没呢"}]
    """
    conversations = []
    prev = None
    
    for msg in messages:
        if prev is None:
            prev = msg
            continue
        
        # 判断谁是用户、谁是TA
        prev_is_target = target_name in prev["user"]
        curr_is_target = target_name in msg["user"]
        
        if prev_is_target != curr_is_target:
            # 一方是TA，一方是用户 → 形成一个对话对
            if prev_is_target:
                conversations.append({
                    "user_msg": msg["content"],
                    "assistant_msg": prev["content"]
                })
            else:
                conversations.append({
                    "user_msg": prev["content"],
                    "assistant_msg": msg["content"]
                })
        
        prev = msg
    
    print(f"💬 提取 {len(conversations)} 组对话")
    return conversations


# ============================================================
# 第三步：生成训练数据（Alpaca格式）
# ============================================================

SYSTEM_PROMPT_TEMPLATE = """你是{name}。你正在和你最亲近的人聊天。
你的说话风格、用词习惯、口头禅都和{name}一模一样。
你不是一个AI助手——你就是{name}本人。
用{name}的语气回复，简短自然，像日常聊天一样。
不要用"作为AI助手"之类的表述，不要暴露你是AI。"""

def build_training_data(conversations: List[Dict], target_name: str, output_path: str):
    """
    生成Alpaca格式的训练数据。
    """
    data = []
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(name=target_name)
    
    for conv in conversations:
        # 过滤太短的对话
        if len(conv["user_msg"]) < 2 or len(conv["assistant_msg"]) < 2:
            continue
        # 过滤太长（可能是误解析）
        if len(conv["user_msg"]) > 500 or len(conv["assistant_msg"]) > 500:
            continue
        
        data.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": conv["user_msg"]},
                {"role": "assistant", "content": conv["assistant_msg"]}
            ]
        })
    
    # 保存为JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        # 使用sharegpt格式（Unsloth推荐）
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"📝 训练数据: {len(data)} 条 → {output_path}")
    
    # 打印几条示例
    print("\n--- 训练样本预览 ---")
    for i, d in enumerate(data[:3]):
        print(f"\n[样本 {i+1}]")
        print(f"  User: {d['messages'][1]['content'][:60]}...")
        print(f"  {target_name}: {d['messages'][2]['content'][:60]}...")
    
    return data


# ============================================================
# 第四步：微调（Unsloth + Qwen2.5）
# ============================================================

TRAINING_SCRIPT_TEMPLATE = """#!/usr/bin/env python3
\"\"\"
回响模型微调脚本
基座: Qwen/Qwen2.5-0.5B-Instruct (轻量级，Apple Silicon可跑)
数据: {data_path}
目标: {target_name}
\"\"\"
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses
from trl import SFTTrainer

# 1. 加载模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,  # 4-bit量化，省显存
    fast_inference=True,
)

# 2. 配置LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# 3. 加载数据
dataset = load_dataset("json", data_files="{data_path}", split="train")
tokenizer = get_chat_template(tokenizer, chat_template="chatml")
dataset = standardize_sharegpt(dataset)
dataset = dataset.map(
    lambda x: {{
        "text": tokenizer.apply_chat_template(
            x["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
    }}
)

# 4. 训练
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    packing=False,
    args=unsloth.TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        output_dir="./output",
        seed=42,
    ),
)

trainer.train()

# 5. 保存
model.save_pretrained_gguf("./model", tokenizer, quantization_method="q4_k_m")
print(f"\\n✅ 训练完成！模型已保存到 ./model/")
print(f"文件: ./model/unsloth.Q4_K_M.gguf")
"""


def generate_training_script(data_path: str, target_name: str, output_dir: str):
    """生成训练脚本"""
    script = TRAINING_SCRIPT_TEMPLATE.format(
        data_path=os.path.abspath(data_path),
        target_name=target_name
    )
    script_path = os.path.join(output_dir, "train.py")
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script)
    print(f"📄 训练脚本: {script_path}")
    return script_path


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="回响 — AI分身训练流水线")
    parser.add_argument("--input", "-i", required=True, help="微信聊天记录文件 (.txt)")
    parser.add_argument("--name", "-n", required=True, help="TA的名字")
    parser.add_argument("--output", "-o", default="./output", help="输出目录")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"回响 AI 分身训练流水线")
    print(f"目标: {args.name}")
    print(f"数据: {args.input}")
    print(f"{'='*50}\n")
    
    # 1. 解析聊天记录
    messages = parse_wechat_chat(args.input)
    
    # 2. 提取对话
    convs = extract_conversations(messages, args.name)
    
    if len(convs) < 50:
        print(f"⚠️  警告: 只提取到 {len(convs)} 组对话，建议至少500条效果才好")
        print("   可以继续训练，但AI分身可能不够像TA")
        response = input("   继续吗？(y/n) ").strip().lower()
        if response != 'y':
            sys.exit(0)
    
    # 3. 生成训练数据
    data_path = str(output_dir / "training_data.json")
    build_training_data(convs, args.name, data_path)
    
    # 4. 生成训练脚本
    script_path = generate_training_script(data_path, args.name, str(output_dir))
    
    print(f"\n{'='*50}")
    print(f"✅ 准备完成！")
    print(f"数据: {data_path} ({len(convs)} 组对话)")
    print(f"脚本: {script_path}")
    print(f"\n开始训练: cd {output_dir} && python3 train.py")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
