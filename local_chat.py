import os

# 减少显存碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoConfig,
    GenerationConfig,
)


MODEL_CONFIGS = {
    "GLM": {
        "path": "/mnt/workspace/chatglm3-6b",
        "type": "glm",
    },
    "Qwen": {
        "path": "/mnt/workspace/Qwen-7B-Chat",
        "type": "qwen",
    },
}


MAX_HISTORY_ROUNDS = 5
MAX_NEW_TOKENS = 256
GLM_MAX_LENGTH = 4096

current_model_name = None
current_model_type = None
tokenizer = None
model = None

histories = {
    "GLM": [],
    "Qwen": [],
}


def check_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境没有检测到 GPU，请先确认 nvidia-smi 是否正常。")

    print("GPU 可用")
    print("GPU 名称:", torch.cuda.get_device_name(0))
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
    print(f"GPU 显存: {total_mem:.2f} GB\n")


def show_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024
        reserved = torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024
        print(f"[显存] 已分配: {allocated:.2f} GB, 已保留: {reserved:.2f} GB")


def unload_model():
    global tokenizer, model, current_model_name, current_model_type

    if model is not None:
        del model
        model = None

    if tokenizer is not None:
        del tokenizer
        tokenizer = None

    current_model_name = None
    current_model_type = None

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def trim_history(history):
    if not history:
        return []

    # Qwen-7B-Chat 通常是 [(query, response), ...]
    if isinstance(history[0], tuple):
        return history[-MAX_HISTORY_ROUNDS:]

    # ChatGLM3 通常是 [{"role": "...", "content": "..."}, ...]
    if isinstance(history[0], dict):
        return history[-MAX_HISTORY_ROUNDS * 2:]

    return history[-MAX_HISTORY_ROUNDS:]


def load_model(model_name):
    global tokenizer, model, current_model_name, current_model_type

    if model_name not in MODEL_CONFIGS:
        print("没有这个模型编号。可用模型：GLM / Qwen")
        return

    if current_model_name == model_name and model is not None:
        print(f"当前已经是模型 {model_name}")
        return

    unload_model()

    cfg = MODEL_CONFIGS[model_name]
    model_path = cfg["path"]
    model_type = cfg["type"]

    print(f"\n正在加载模型 {model_name}: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    if model_type == "glm":
        glm_config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        if not hasattr(glm_config, "max_length"):
            glm_config.max_length = getattr(glm_config, "seq_length", GLM_MAX_LENGTH)

        # ChatGLM3 不建议用 device_map='auto'，直接 half().cuda() 更稳
        model = AutoModel.from_pretrained(
            model_path,
            config=glm_config,
            trust_remote_code=True
        ).half().cuda()

    elif model_type == "qwen":
        # Qwen 直接放到 0 号 GPU，避免部分 offload 到 CPU 导致变慢
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map={"": 0}
        )

        try:
            model.generation_config = GenerationConfig.from_pretrained(
                model_path,
                trust_remote_code=True
            )
        except Exception:
            pass

        if hasattr(model, "generation_config"):
            model.generation_config.max_new_tokens = MAX_NEW_TOKENS
            model.generation_config.do_sample = True
            model.generation_config.top_p = 0.8
            model.generation_config.temperature = 0.7
            model.generation_config.repetition_penalty = 1.05

            if model.generation_config.pad_token_id is None:
                model.generation_config.pad_token_id = model.generation_config.eos_token_id

    else:
        raise ValueError("未知模型类型")

    model.eval()

    current_model_name = model_name
    current_model_type = model_type

    print(f"模型 {model_name} 加载完成。")
    show_gpu_memory()
    print()


def chat_once(user_input):
    global histories

    if model is None:
        print("当前没有加载模型，请先使用 /switch GLM 或 /switch Qwen")
        return

    history = histories[current_model_name]
    history = trim_history(history)

    print(f"\n{current_model_name}: ", end="", flush=True)

    start = time.time()

    with torch.inference_mode():
        if current_model_type == "glm":
            response, new_history = model.chat(
                tokenizer,
                user_input,
                history=history,
                do_sample=True,
                top_p=0.8,
                temperature=0.7,
            )

        elif current_model_type == "qwen":
            response, new_history = model.chat(
                tokenizer,
                user_input,
                history=history,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                top_p=0.8,
                temperature=0.7,
            )

        else:
            response = "未知模型类型"
            new_history = history

    torch.cuda.synchronize()

    histories[current_model_name] = trim_history(new_history)

    end = time.time()

    print(response)
    print(f"\n[本轮耗时: {end - start:.2f} 秒]")
    show_gpu_memory()
    print()


def show_models():
    print("\n当前可用模型：")
    for name, cfg in MODEL_CONFIGS.items():
        flag = " ← 当前使用" if name == current_model_name else ""
        print(f"{name}: {cfg['path']}{flag}")
    print()


def clear_history():
    if current_model_name is None:
        print("当前没有加载模型。")
        return

    histories[current_model_name] = []
    print(f"{current_model_name} 的对话历史已清空。")


def clear_all_history():
    for name in histories:
        histories[name] = []
    print("所有模型的对话历史已清空。")


def show_history():
    if current_model_name is None:
        print("当前没有加载模型。")
        return

    print(f"\n========== {current_model_name} 对话历史 ==========")
    for item in histories[current_model_name]:
        print(item)
    print("====================================\n")


def show_help():
    print("""
可用命令：

/models              查看所有模型
/switch GLM          切换到 GLM
/switch Qwen         切换到 Qwen
/clear               清空当前模型对话历史
/clear_all           清空所有模型对话历史
/history             查看当前模型历史
/help                查看帮助
/exit                退出程序

说明：
1. 每次只加载一个模型，切换模型时会自动释放上一个模型的显存。
2. Qwen-7B-Chat 和 ChatGLM3-6B 都会使用 GPU 半精度推理。
3. 如果显存不够，把 MAX_NEW_TOKENS 调小，比如 128。
""")


def main():
    check_gpu()

    print("本地 GPU 多模型对话程序启动")
    show_models()
    show_help()

    while True:
        try:
            user_input = input("你: ").strip()
        except KeyboardInterrupt:
            print("\n已中断。")
            break

        if not user_input:
            continue

        if user_input == "/exit":
            print("已退出。")
            break

        elif user_input == "/help":
            show_help()

        elif user_input == "/models":
            show_models()

        elif user_input.startswith("/switch"):
            parts = user_input.split()

            if len(parts) != 2:
                print("用法：/switch GLM 或 /switch Qwen")
                continue

            load_model(parts[1])

        elif user_input == "/clear":
            clear_history()

        elif user_input == "/clear_all":
            clear_all_history()

        elif user_input == "/history":
            show_history()

        else:
            chat_once(user_input)


if __name__ == "__main__":
    main()