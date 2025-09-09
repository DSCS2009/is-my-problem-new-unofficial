import asyncio, json, hashlib, aiohttp, orjson, os
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from concurrent.futures import ThreadPoolExecutor
import aiofiles

# ---------- 配置 ----------
with open("settings.json", encoding="utf-8") as f:
    settings = json.load(f)

OLLAMA_BASE = "http://localhost:11435/v1"
MODEL_NAME  = "/root/Qwen3/Qwen3-0.6B-Q8_0.gguf"
WORKERS     = 1
TIMEOUT     = aiohttp.ClientTimeout(total=60)

# 线程池：用于同步文件 IO
io_pool = ThreadPoolExecutor(max_workers=10)

# ---------- 工具 ----------
def read_problem_sync(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def dump_json_sync(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

async def read_problem(path):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(io_pool, read_problem_sync, path)

async def save_problem(data, path):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(io_pool, dump_json_sync, data, path)

def check_processed(p, template):
    orig = p["statement"]
    prompt = template.replace("[[ORIGINAL]]", orig).strip() + " /no_think"
    md5 = hashlib.md5(prompt.encode()).hexdigest()[:8]
    return any(f["prompt_md5"][:8] == md5 for f in p.get("processed", []))

# ---------- 业务 ----------
async def process_one(problem_file, session, pbar):
    p = await read_problem(problem_file)
    p["processed"] = p.get("processed", [])

    changed = False
    for tpl in settings["TEMPLATES"]:
        if check_processed(p, tpl):
            continue

        prompt = tpl.replace("[[ORIGINAL]]", p["statement"]).strip() + " /no_think"
        prompt_md5 = hashlib.md5(prompt.encode()).hexdigest()[:8]
        tpl_md5    = hashlib.md5(tpl.encode()).hexdigest()[:8]

        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.1, "top_p": 0.9}
        }

        async with session.post(f"{OLLAMA_BASE}/chat/completions", json=payload) as resp:
            if resp.status != 200:
                txt = await resp.text()
                print(f"[{problem_file}] HTTP {resp.status}: {txt}")
                continue
            body = orjson.loads(await resp.read())
            result = body["choices"][0]["message"]["content"].strip()

        p["processed"].append({
            "prompt_md5": prompt_md5,
            "template_md5": tpl_md5,
            "result": result
        })
        changed = True

    if changed:
        await save_problem(p, problem_file)
    pbar.update(1)

# ---------- 新增 ----------
import aiofiles, signal, atexit
from pathlib import Path

DONE_LOG = Path(__file__).with_name("done.log")   # 和脚本同目录
done_set = set()                                  # 内存去重

async def load_done():
    """启动时载入已完成列表"""
    if DONE_LOG.exists():
        async with aiofiles.open(DONE_LOG, "r") as f:
            async for line in f:
                done_set.add(line.strip())
    return done_set

async def mark_done(path: str):
    """异步追加到 done.log"""
    async with aiofiles.open(DONE_LOG, "a") as f:
        await f.write(f"{path}\n")
    done_set.add(path)

import os
# ---------- 主流程微调 ----------
async def main():
    # 1. 载入历史完成记录
    await load_done()

    # 2. 扫描待做文件（跳过已完成的）
    from .utils import problem_filenames
    all_files = sorted(problem_filenames())
    todo = []
    for f in all_files:
        if f in done_set:                       # 关键：跳过已完成
            continue
        p = await read_problem(f)
        if "https://cdn.luogu.com.cn/upload/vjudge_pic/UVA" in p["statement"].strip():
            # os.system('rm ' + f)
            print(f)
            continue
        if any(not check_processed(p, t) for t in settings["TEMPLATES"]):
            todo.append(f)

    if not todo:
        print("All problems already processed.")
        return

    print(f"Total to process this run: {len(todo)}")

    # 3. 正常并发处理
    conn = aiohttp.TCPConnector(limit_per_host=WORKERS)
    async with aiohttp.ClientSession(connector=conn, timeout=TIMEOUT) as session:
        with tqdm_asyncio(total=len(todo), desc="Processing") as pbar:
            semaphore = asyncio.Semaphore(WORKERS)

            async def sem_task(f):
                async with semaphore:
                    await process_one(f, session, pbar)
                    await mark_done(f)          # ✅ 完成后立即记录

            # 让 Ctrl+C 也能把当前进度刷盘
            # def _flush(*_):
            #    asyncio.create_task(mark_done("dummy"))  # 触发一次写盘即可
            # signal.signal(signal.SIGINT, _flush)
            # atexit.register(lambda: asyncio.run(mark_done("dummy")))

            await asyncio.gather(*(sem_task(f) for f in todo))

if __name__ == "__main__":
    asyncio.run(main())

