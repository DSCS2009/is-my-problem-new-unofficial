#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按字典序分配题目编号，uid 固定为 洛谷/编号
python3 assign_id.py
"""

import os
import re
import json
from typing import List, Dict, Optional

LANG_PRIORITY = ('en', 'zh', 'ja')

# ---------- 工具 ----------
def extract_first_group(pattern: str, text: str) -> Optional[str]:
    m = re.search(pattern, text, re.I)
    return m.group(1).strip() if m else None

def choose_language(sections: Dict[str, str]) -> str:
    # print(sections)
    for prefix in LANG_PRIORITY:
        for k, v in sections.items():
            # print('check', k)
            if k.lower().startswith(prefix):
                return v
    return next(iter(sections.values())) if sections else ""

# ---------- 单文件解析 ----------
def parse_one_file(md_path: str) -> Optional[Dict]:
    try:
        text = open(md_path, encoding='utf-8').read()
    except Exception as e:
        print(f"[WARN] 读取 {md_path} 失败：{e}")
        return None

    # 1. 语言分段
    lang_chunks = re.split(
        r'^\s*###\s*\[([^\]]+)\]\s*:\s*(.+?)(?=^\s*###|^\s*-----|\Z)',
        text, flags=re.M | re.S
    )
    sections = {}
    for i in range(1, len(lang_chunks), 3):
        lang, title = lang_chunks[i].strip(), lang_chunks[i + 1].strip()
        if lang and title:
            sections[lang] = title

    chosen = choose_language(sections)
    if not chosen:
        return None

    # 2. 标题——安全版
    title_line = extract_first_group(r'^(.+?)$', chosen)
    title = title_line if title_line else os.path.basename(md_path).replace('_题目信息.md', '')

    # 3. 来源
    source = extract_first_group(r'^\s*\*\s*\*\*来源\*\*\s*[:：]\s*(.+?)\s*$', text) or \
             extract_first_group(r'^\s*\*\s*\*\*类型\*\*\s*[:：]\s*(.+?)\s*$', text) or "unknown"

    # 4. 题号
    pid = extract_first_group(r'^\s*\*\s*\*\*题目编号\*\*\s*[:：]\s*(.+?)\s*$', text)
    if pid is None:
        pid = os.path.basename(md_path).replace('_题目信息.md', '')
        # print(f"[DEBUG] 文件 {os.path.basename(md_path)} 未找到**题目编号**，退而用文件名 -> {pid}")

    # 5. URL - 修改第二个正则表达式，添加捕获组
    url = extract_first_group(r'\[problemUrl\]:\s*(https?://\S+)', text) or \
          extract_first_group(r'(https?://[^\s\)]+)', text) or ""

    # 6. 标签
    tag_line = extract_first_group(r'^\s*\*\s*\*\*标签\*\*\s*[:：]\s*(.+?)\s*$', text) or ""
    tags = [t.strip() for t in re.split(r'[，,、\s]+', tag_line) if t.strip()]

    # 7. 题目描述——判空
    stmt_match = re.search(r'####\s*题目描述\s*(.+?)(?=\n####|\n-----|\Z)', text, re.S)
    statement = stmt_match.group(1).strip().replace('\n', '\\n') if stmt_match else ""

    return {
        "uid": f"{source}/{pid}",
        "url": url,
        "tags": tags,
        "title": title,
        "statement": statement,
        "source": "洛谷",
        "vjudge": False
    }

# ---------- 主流程 ----------
def collect_all(root_dir: str = '.') -> List[Dict]:
    res = []
    for dirpath, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith('_题目信息.md'):
                full = os.path.join(dirpath, f)
                try:
                    data = parse_one_file(full)
                    if data:
                        res.append(data)
                except Exception as e:
                    print(f"[WARN] 跳过 {full} : {e}")
    return res

def assign_and_save(problems: List[Dict], start: int = 1000):
    # 按 uid 字典序排序
    problems.sort(key=lambda x: x["uid"])
    for idx, p in enumerate(problems, start):
        # 覆写 uid 为“洛谷/编号”
        p["uid"] = f"洛谷/{idx}"
        fname = f"{idx}.json"
        with open(fname, 'w', encoding='utf-8') as f:
            f.write(json.dumps(p, ensure_ascii=False))
    print(f"分配完成，共 {len(problems)} 道题，编号 {start} ~ {start + len(problems) - 1}")

if __name__ == '__main__':
    problems = collect_all()
    print(len(problems))
    assign_and_save(problems)