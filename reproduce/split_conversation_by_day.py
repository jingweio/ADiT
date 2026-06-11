#!/usr/bin/env python3
"""把当前 Claude Code 会话的对话按【天】拆成多个 Markdown 文件。

复用 export_conversation.py 的门控 + 合并逻辑(只保留真实用户提问 + 其后的助手回复文本,
跳过 system-reminder / 自动轮询 / 工具调用 / thinking)。每条事件用其来源记录的 timestamp
(UTC) 转成 KAUST 本地时区 (UTC+3) 的日期;一个 Q&A 对统一归到【提问那天】,避免跨午夜被拆开。

输出:/home/guoj0f/repos/ADiT/claude-conversation-records/YYYY-MM-DD.md(每天一个)
      + 同目录 INDEX.md(总览)
"""
import json, glob, os, datetime
from collections import defaultdict

PROJ = os.path.expanduser("~/.claude/projects/-home-guoj0f-repos")
OUTDIR = "/home/guoj0f/repos/ADiT/claude-conversation-records"
SESSION_ID = "6111c5bc-5521-4735-b974-4edcdb1a4587"
TZ = datetime.timezone(datetime.timedelta(hours=3))  # KAUST / AST (UTC+3, 无 DST)

NOISE_MARKERS = (
    "<system-reminder", "<task-notification", "<local-command",
    "Caveat: The messages below", "<command-name>", "<command-message>",
    "[SYSTEM NOTIFICATION", "## Exited Plan Mode",
    "推进 ADiT", "完成 ADiT", "ADiT LBA 复现最后一步", "每15分钟进度汇报",
)
# 注:不过滤自动监控 prompt —— 它们后面跟着的是真实的 OOD 进度/结果汇报(assistant),
# 不能为了去噪把内容删掉。靠下面 esc_user() 的转义来防止排版被带歪(改格式,不删内容)。


def esc_user(t):
    """转义用户消息里的 markdown/HTML 特殊字符,使其按字面渲染——
    避免裸 `*`/`_`(强调)、`<i>`(HTML 斜体标签)、反引号等把整篇排版带歪。
    用户消息是原始输入、本就不含预期 markdown,所以全部转义是安全的。"""
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    out = []
    for ch in t:
        if ch in "\\`*_~#|":
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


def latest_transcript():
    pinned = os.path.join(PROJ, SESSION_ID + ".jsonl")
    if os.path.exists(pinned):
        return pinned
    files = glob.glob(os.path.join(PROJ, "*.jsonl"))
    return max(files, key=os.path.getmtime) if files else None


def text_of(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(b.get("text", "") for b in content
                     if isinstance(b, dict) and b.get("type") == "text")


def is_noise(t):
    s = t.lstrip()
    return any(m in s[:200] for m in NOISE_MARKERS)


def local_date(ts):
    """'2026-05-30T22:10:00.000Z' -> KAUST 本地日期 'YYYY-MM-DD'。"""
    if not ts:
        return None
    try:
        dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.astimezone(TZ).strftime("%Y-%m-%d")
    except Exception:
        return None


def main():
    tp = latest_transcript()
    if not tp or not os.path.exists(tp):
        print("no transcript found"); return

    # 门控 + 合并,带上日期;Q&A 对归到提问那天(cur_date)
    events = []      # [(role, text, date)]
    keep = False
    cur_date = None
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
            ld = local_date(d.get("timestamp"))
            if role == "user":
                if not t:
                    continue
                if is_noise(t):
                    keep = False
                    continue
                keep = True
                cur_date = ld or cur_date
                events.append(["user", t, cur_date])
            elif role == "assistant":
                if not keep or not t:
                    continue
                if events and events[-1][0] == "assistant":
                    events[-1][1] += "\n\n" + t          # 合并被工具切开的多段
                else:
                    events.append(["assistant", t, cur_date])  # 归到提问那天

    # 按天分组
    by_day = defaultdict(list)
    for role, t, dd in events:
        by_day[dd or "unknown"].append((role, t))

    os.makedirs(OUTDIR, exist_ok=True)
    index = ["# Claude Code 对话记录 — ADiT(按天)", "",
             f"> 数据源:`{tp}`  ", f"> 生成:{datetime.datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S')} (UTC+3)",
             "", "| 日期 | 用户消息 | 助手回复 | 文件 |", "|---|---|---|---|"]
    for day in sorted(by_day):
        evs = by_day[day]
        nu = sum(1 for r, _ in evs if r == "user")
        na = sum(1 for r, _ in evs if r == "assistant")
        out = [f"# Claude Code 对话记录 — ADiT — {day}", "",
               f"> 数据源:`{tp}`(按 KAUST UTC+3 本地日期切分)  ",
               f"> 本日:{nu} 条用户消息 / {na} 条助手回复", ""]
        for role, t in evs:
            if role == "user":
                out.append("\n---\n")
                out.append("### 👤 用户\n")
                out.append(esc_user(t))
            else:
                out.append("\n### 🤖 Claude\n")
                out.append(t)
        fname = f"{day}.md"
        with open(os.path.join(OUTDIR, fname), "w", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")
        index.append(f"| {day} | {nu} | {na} | [{fname}]({fname}) |")
        print(f"wrote {fname}: user={nu} assistant={na}")
    with open(os.path.join(OUTDIR, "INDEX.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(index) + "\n")
    print(f"wrote INDEX.md ; {len(by_day)} day-files in {OUTDIR}")


if __name__ == "__main__":
    main()
