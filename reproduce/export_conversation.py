#!/usr/bin/env python3
"""把当前 Claude Code 会话的对话(用户消息 + 助手回复文本)导出成 Markdown。
数据源:~/.claude/projects/-home-guoj0f-repos/ 下最近修改的 .jsonl(即当前会话)。
输出:/home/guoj0f/repos/ADiT/claude_conversation.md
工具调用/工具结果/思考过程不导出,只保留"聊天"内容。
"""
import json, glob, os, datetime

PROJ = os.path.expanduser("~/.claude/projects/-home-guoj0f-repos")
OUT = "/home/guoj0f/repos/ADiT/claude_conversation.md"

# 这些是注入到对话里的非"聊天"噪声,跳过
NOISE_MARKERS = (
    "<system-reminder", "<task-notification", "<local-command",
    "Caveat: The messages below", "<command-name>", "<command-message>",
    "[SYSTEM NOTIFICATION", "## Exited Plan Mode",
    # 自动轮询 / 定时唤醒的提示词(都出现在消息开头),整段视为噪声并抑制其后的轮询回复
    "推进 ADiT", "完成 ADiT", "ADiT LBA 复现最后一步", "每15分钟进度汇报",
)


# 钉死到“我们这个会话”的 transcript,避免并发会话(另开 terminal)时被“选最新”误选。
SESSION_ID = "6111c5bc-5521-4735-b974-4edcdb1a4587"

def latest_transcript():
    pinned = os.path.join(PROJ, SESSION_ID + ".jsonl")
    if os.path.exists(pinned):
        return pinned
    # 兜底:钉定文件不在时才退回“最近修改的 jsonl”
    files = glob.glob(os.path.join(PROJ, "*.jsonl"))
    return max(files, key=os.path.getmtime) if files else None


def text_of(content):
    """只抽取 text 块(忽略 tool_use / tool_result / thinking)。"""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for b in content:
        if isinstance(b, dict) and b.get("type") == "text":
            parts.append(b.get("text", ""))
    return "\n".join(parts)


def is_noise(t):
    s = t.lstrip()
    return any(m in s[:200] for m in NOISE_MARKERS)


def main():
    tp = latest_transcript()
    out = []
    out.append("# Claude Code 对话记录 — ADiT 复现")
    out.append("")
    out.append(f"> 数据源:`{tp}`  ")
    out.append(f"> 最近更新:{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}（每约 20 分钟自动刷新）")
    out.append("")
    # 先做门控 + 合并:得到干净的(角色,文本)事件序列;同一次回答的多段合并为一段
    events = []   # [(role, text)]
    keep = False  # 是否处于“真实提问”的回合
    if tp and os.path.exists(tp):
        with open(tp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("type") not in ("user", "assistant"):
                    continue
                msg = d.get("message")
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role")
                t = text_of(msg.get("content")).strip()
                if role == "user":
                    if not t:
                        continue          # tool_result 等无文本的 user 记录:不改变回合状态
                    if is_noise(t):
                        keep = False      # 系统提醒 / 自动轮询提示:抑制其后的助手回复
                        continue
                    keep = True           # 真正的用户提问
                    events.append(["user", t])
                elif role == "assistant":
                    if not keep or not t:  # 只保留“真实提问之后”的助手回复
                        continue
                    if events and events[-1][0] == "assistant":
                        events[-1][1] += "\n\n" + t     # 合并同一次回答被工具切开的多段
                    else:
                        events.append(["assistant", t])
    nu = sum(1 for r, _ in events if r == "user")
    na = sum(1 for r, _ in events if r == "assistant")
    for role, t in events:
        if role == "user":
            out.append("\n---\n")
            out.append("### 👤 用户\n")
            out.append(t)
        else:
            out.append("\n### 🤖 Claude\n")
            out.append(t)
    out.append("\n\n---")
    out.append(f"_共 {nu} 条用户消息 / {na} 条助手回复_")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {OUT}: user={nu} assistant={na} from {os.path.basename(tp) if tp else None}")


if __name__ == "__main__":
    main()
