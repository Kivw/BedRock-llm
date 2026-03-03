from transformers import AutoTokenizer, GenerationConfig
from model import BedRockV3ForCausalLM
import torch

# ================= 路径设置 =================
tokenizer_path = "/home/jx/experiment/BedRock-llm/tokenizer/tokenizer_modify"
model_path = "/home/jx/experiment/BedRock-llm/result/checkpoint-104000"

# ================= 加载模型 =================
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
model = BedRockV3ForCausalLM.from_pretrained(model_path, trust_remote_code=True)

# 显存与评估模式设置
model.config.use_cache = True
model.to("cuda")
model.eval()

# 确保 pad_token 有值
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ================= 生成配置 =================
# 预训练模型更容易复读或发散，建议加强约束
gen_config = GenerationConfig(
    max_new_tokens=512,
    min_new_tokens=1,
    do_sample=True,
    temperature=0.9,
    top_p=0.7,              # 适当调大，增加多样性
    # top_k=50,               # 开启 top_k 防止低概率词，比 0 更稳定
    repetition_penalty=1.5, # 预训练模型容易复读，加重惩罚
    # no_repeat_ngram_size=3, # 防止重复生成
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    use_cache=True,
    num_beams=1,
)

# ================= 推理函数 =================
def inference_pretrain(prompt_text):
    """
    预训练模型推理：直接进行文本续写
    prompt_text: str, 例如 "问题：... \n 答案："
    """
    # 1️⃣ Tokenize 纯文本 (不再使用 apply_chat_template)
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )
    
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    input_length = input_ids.shape[1]
    
    print(f"输入提示词:\n{prompt_text}")
    print("-" * 50)
    
    # 2️⃣ 推理
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=gen_config,
        )
    
    # 3️⃣ 截取生成部分 (去掉输入部分)
    generated_ids = outputs[:, input_length:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response

# ================= 主函数 =================
def main():
    # --- 预训练模型的 Prompt 工程 ---
    # 预训练模型不懂"角色"，只懂"续写"，需要用自然语言引导
    question = "中国的首都是北"

    # 选择模版 A (问答式)
    prompt = f"你知道吗，{question}京，它是"
    
    # 方案 B: 如果是代码预训练为主，可用代码注释风格
    # prompt = f"# Python code to open and display an image using opencv:\n"
    
    # 方案 C: 简单问答风格
    # prompt = f"问：{question}\n答："
    
    answer = inference_pretrain(prompt)
    
    print("模型生成结果:")
    print(answer)

if __name__ == "__main__":
    main()
