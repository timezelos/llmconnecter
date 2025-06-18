#!/usr/bin/env python3
"""
与 Ollama 本地模型对话（支持多日志、style_log、system_prompt、tail_prompt）
"""
import requests, yaml, re, datetime, pathlib, argparse, sys, os

# ──────────── 基本配置 ────────────
CFG_PATH = "config.yaml"               # YAML 配置文件
LOG_DIR  = "logs"                      # 聊天日志目录
os.makedirs(LOG_DIR, exist_ok=True)
# ──────────────────────────────────

LOG_RE = re.compile(r"\[.*?\]\s*(\w+):\s*(.*)", re.S)
now = lambda: datetime.datetime.now().isoformat(timespec="seconds")

def append_log(path: pathlib.Path, role: str, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{now()}] {role}: {content}\n")

def parse_log(path: pathlib.Path):
    if not path.exists(): return []
    role_map = {"USER": "user", "LLM": "assistant"}
    msgs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = LOG_RE.match(line.strip())
        if m and m.group(1) in role_map:
            msgs.append({"role": role_map[m.group(1)], "content": m.group(2)})
    return msgs

def parse_style(path: pathlib.Path):
    if not path or not path.exists(): return []
    role_map = {"SYS": "system", "SYSTEM": "system",
                "USER": "user",   "LLM": "assistant"}
    msgs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = LOG_RE.match(line.strip())
        if m:
            msgs.append({"role": role_map.get(m.group(1).upper(), "system"),
                         "content": m.group(2)})
    return msgs

def choose_log():
    logs = sorted(pathlib.Path(LOG_DIR).glob("*.log"))
    if not logs:
        return pathlib.Path(LOG_DIR) / f"{now().replace(':','-')}.log"
    print("选择要加载的聊天记录：")
    for i, fp in enumerate(logs, 1):
        print(f"  {i}. {fp.name}")
    print("  0. 新建对话")
    while True:
        sel = input("编号> ").strip()
        if sel == "0":
            return pathlib.Path(LOG_DIR) / f"{now().replace(':','-')}.log"
        if sel.isdigit() and 1 <= int(sel) <= len(logs):
            return logs[int(sel) - 1]
        print("无效编号，请重试。")

# ──────────── ① load_cfg：读出 tail_prompt ────────────
def load_cfg(path: str):
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    ollama = cfg["character_config"]["agent_config"]["llm_configs"]["ollama_llm"]

    prompt_sec = cfg.get("prompt_config", {})
    base_url  = ollama["base_url"].rstrip("/") + "/chat/completions"
    model     = ollama["model"]
    temp      = ollama.get("temperature", 1.0)
    keep_alive= ollama.get("keep_alive", -1)
    sys_prompt= prompt_sec.get("system_prompt", "")
    style_log = prompt_sec.get("style_log", "")
    # >>> 新增 (tail_prompt) ---------------------------
    tail_prompt = prompt_sec.get("tail_prompt", "")
    # <<<----------------------------------------------
    return base_url, model, temp, keep_alive, sys_prompt, style_log, tail_prompt
# ─────────────────────────────────────────────────────

def ask_ollama(url, model, temp, keep_alive, messages):
    payload = dict(model=model, temperature=temp,
                   keep_alive=keep_alive, messages=messages, stream=False)
    r = requests.post(url, json=payload, timeout=100000)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", help="指定日志文件")
    ap.add_argument("--new", action="store_true", help="忽略旧记录新建对话")
    args = ap.parse_args()

    log_fp = (pathlib.Path(args.log) if args.log and not args.new else
              pathlib.Path(LOG_DIR) / f"{now().replace(':','-')}.log"
              if args.new else choose_log())

    # ──────────── ② main 接收 tail_prompt ────────────
    (endpoint, model, temp, keep_alive,
     sys_prompt, style_path, tail_prompt) = load_cfg(CFG_PATH)
    # ────────────────────────────────────────────────

    style_msgs = parse_style(pathlib.Path(style_path))
    history    = parse_log(log_fp)

    print(f"✅ 模型: {model} | 日志: {log_fp.name}")
    if style_msgs: print(f"🎨 style_log 行数: {len(style_msgs)}")
    if sys_prompt: print("📌 system_prompt 将在首轮注入")
    if tail_prompt:print("🔄 tail_prompt 将在每轮末尾注入")

    first_turn = True
    while True:
        user = input("你: ").strip()
        if user.lower() in {"exit", "quit"}: break

        history.append({"role": "user", "content": user})
        append_log(log_fp, "USER", user)

        # 组装 messages
        if first_turn:
            messages = style_msgs.copy()
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            messages += history
        else:
            messages = history

        # ──────────── ③ 每轮末尾插 tail_prompt ───────────
        if tail_prompt:
            messages_to_send = messages + [{"role": "system",
                                            "content": tail_prompt}]
        else:
            messages_to_send = messages
        # ───────────────────────────────────────────────

        try:
            answer = ask_ollama(endpoint, model, temp,
                                keep_alive, messages_to_send)
        except Exception as e:
            print("⚠️  请求失败:", e)
            history.pop(); continue

        print("LLM:", answer, "\n")
        history.append({"role": "assistant", "content": answer})
        append_log(log_fp, "LLM", answer)
        first_turn = False

    print("👋 对话结束，日志已保存 ->", log_fp.resolve())

if __name__ == "__main__":
    main()
