import pickle
import requests
import json
from .utils import read_problem, problem_filenames, dump_json_safe, dump_json_safe_utf8, dump_pickle_safe
from openai import AsyncOpenAI
from together import Together, AsyncTogether
import anthropic
import hashlib
import asyncio
from tqdm.auto import tqdm
import time, os
import random
import numpy as np

with open("settings.json") as f:
    settings = json.load(f)

# 配置llama.cpp embedding API地址
LLAMA_CPP_EMBEDDING_URL = "http://localhost:11436/embedding"

def processed_promptmd5(statement, template):
    ORIGINAL = statement
    prompt = template.replace("[[ORIGINAL]]", ORIGINAL).strip()
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8]

def problem_embeds(problem_file_cur):
    problem = read_problem(problem_file_cur)      # 原本就有
    # try:
    with open(problem_file_cur.replace('.json', '.vopkl'), 'rb') as f:
        embeds = pickle.load(f)
    # except:
    #     embeds = []
    return problem, embeds

def get_llama_cpp_embeddings(texts, model="/root/Qwen3/Qwen3-Embedding-0.6B-Q8_0.gguf"):
    """调用llama.cpp embedding API获取文本嵌入（支持批量）"""
    if not texts:
        return []
    
    try:
        # 构建符合llama.cpp embedding API的请求数据
        # 注意：根据您的示例，content应该是一个字符串数组
        content_list = [text for text in texts]  # 为每个文本添加结束标记
        
        data = {
            "model": model,
            "content": content_list
        }

        # print(data)
        
        # 发送请求到llama.cpp embedding API
        response = requests.post(
            LLAMA_CPP_EMBEDDING_URL,
            headers={"Content-Type": "application/json"},
            json=data,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            embeddings = []
            
            # 处理批量响应 - 根据示例，返回的是包含多个embedding对象的数组
            for item in result:
                if "embedding" in item:
                    embedding = item["embedding"]
                    if embedding:
                        embeddings.append(np.array(embedding, dtype=np.float32))
                    else:
                        # 如果获取失败，使用零向量作为fallback
                        print(f"Warning: Empty embedding received for index {item.get('index', 'unknown')}")
                        embeddings.append(np.zeros(1024, dtype=np.float32))
                else:
                    print(f"Warning: No embedding field in response item: {item}")
                    embeddings.append(np.zeros(1024, dtype=np.float32))
            
            if len(embeddings) != len(texts):
                print(f"Warning: Expected {len(texts)} embeddings, got {len(embeddings)}")
                # 补齐缺失的嵌入向量
                while len(embeddings) < len(texts):
                    embeddings.append(np.zeros(1024, dtype=np.float32))
            
            return embeddings
        else:
            print(f"Error: {response.status_code} - {response.text}")
            # 返回与输入文本数量相同的零向量
            return [np.zeros(1024, dtype=np.float32) for _ in texts]
            
    except Exception as e:
        print(f"Exception during llama.cpp embedding: {e}")
        # 返回与输入文本数量相同的零向量
        return [np.zeros(1024, dtype=np.float32) for _ in texts]

# quick and dirty vector database implementation
class VectorDB:
    def __init__(self):
        pass
    
    def load_all(self, shuffle=False, load_around=None, record_tasks=False, skipped_sources=[]):
        self.arr = []
        self.metadata = []
        self.todos = []
        self.sources = {}
        fns = list(problem_filenames())
        if shuffle:
            random.shuffle(fns)
        for problem_file_cur in tqdm(fns):
            if load_around is not None and len(self.arr) > load_around * 2:
                break
            if not record_tasks and not os.path.exists(problem_file_cur.replace(".json", ".vopkl")):
                continue
            problem, embeds = problem_embeds(problem_file_cur)
            if problem is None:
                continue
            statement = problem['statement']
            source = problem['source']
            if source in skipped_sources:
                continue
            self.sources[source] = self.sources.get(source, 0) + 1
            need_work = False
            for template in settings["TEMPLATES"]:
                md5 = processed_promptmd5(statement, template)
                found = False
                for m, u in embeds:
                    if m[:8] == md5:
                        found = True
                        # 处理嵌入向量形状
                        if hasattr(u, 'shape'):
                            # 如果是二维数组 (1, 1024)，转换为 (1024,)
                            if len(u.shape) == 2 and u.shape[0] == 1:
                                u = u[0]  # 取第一行
                            elif len(u.shape) != 1:
                                print(f"Warning: Unexpected embedding shape {u.shape} for {problem_file_cur}, skipping")
                                continue

                            # 检查维度是否正确
                            if u.shape[0] != 1024:
                                print(f"Warning: Invalid embedding dimension {u.shape[0]} for {problem_file_cur}, skipping")
                                continue

                        self.arr.append(np.array(u/np.linalg.norm(u), dtype=np.float16))
                        self.metadata.append((problem_file_cur, source, len(statement.strip())))
                        break
                if not found:
                    need_work = True
            if need_work and record_tasks:
                self.todos.append(problem_file_cur)
        print('found', len(self.arr), 'embeds')
        if len(self.arr) > 0:
            self.arr = np.array(self.arr, dtype=np.float16)
            print(f"Final array shape: {self.arr.shape}")
        else:
            print("Warning: No valid embeddings found")
        if record_tasks:
            print('found', len(self.todos), 'todos')

    def complete_todos(self, chunk_size=200, length_limit=1300, shuffle=False):
        todos = self.todos
        if shuffle:
            random.shuffle(todos)
        for i in tqdm(range(0, len(todos), chunk_size)):
            problems = todos[i:i+chunk_size]
            infos = {}
            for problem_file_cur in problems:
                try:
                    full_problem = read_problem(problem_file_cur)
                    statement = full_problem['statement']
                except Exception as e:
                    print('error', problem_file_cur, e)
                    continue
                try:
                    embeds = []
                    with open(problem_file_cur.replace(".json", ".vopkl"), "rb") as f:
                        embeds = pickle.load(f)
                except:
                    pass
                infos[problem_file_cur] = full_problem.get('processed', []), statement, embeds
            
            for template in settings["TEMPLATES"]:
                queues = []
                max_length = 0
                for problem_file_cur, (processed, statement, embeds) in infos.items():
                    md5 = processed_promptmd5(statement, template)
                    if any(m[:8] == md5 for m, u in embeds): 
                        continue
                    processed_text = None
                    for f in processed:
                        if True or f["prompt_md5"][:8] == md5:
                            if len(f['result']) > length_limit:
                                continue
                            processed_text = f["result"]
                            max_length = max(max_length, len(processed_text))
                    if processed_text is None:
                        continue
                    queues.append((processed_text, problem_file_cur, md5))
                # print('len = ', len(queues))
                if len(queues) == 0:
                    continue
                
                print('batch', len(queues), 'maxlen', max_length)
                try:
                    t0 = time.time()
                    # 使用llama.cpp获取嵌入（批量处理）
                    embeddings = get_llama_cpp_embeddings(
                        [x[0] for x in queues],
                        model="/root/Qwen3/Qwen3-Embedding-0.6B-Q8_0.gguf"
                    )
                    t1 = time.time()
                    
                    print(f'Embedding generation took {t1-t0:.2f} seconds for {len(queues)} texts')
                    
                    # 添加延迟以避免请求过快
                    # if t1 - t0 < 0.2:
                    #    time.sleep(0.2 - (t1 - t0))
                    
                    for q, e in zip(queues, embeddings):
                        infos[q[1]][2].append((q[2], np.array(e)))
                        
                except Exception as e:
                    print('error', e)
                    import traceback
                    traceback.print_exc()
            
            # 保存更新后的嵌入
            for problem_file_cur, (processed, statement, embeds) in infos.items():
                dump_pickle_safe(embeds, problem_file_cur.replace(".json", ".vopkl"))

    def query_nearest(self, emb, k=50, dedup=True):
        emb = np.array(emb)
        if len(emb.shape) == 1:
            emb = emb[None, :]  # Ensure it's 2D even for single embedding

        # 检查数据库是否为空
        if len(self.arr) == 0:
            print("Warning: Database is empty")
            return []

        # 检查嵌入向量维度是否匹配
        if emb.shape[1] != self.arr.shape[1]:
            print(f"Warning: Embedding dimension mismatch. Query: {emb.shape[1]}, DB: {self.arr.shape[1]}")
            # 如果维度不匹配，尝试调整维度
            if emb.shape[1] > self.arr.shape[1]:
                emb = emb[:, :self.arr.shape[1]]
            else:
                # 如果查询向量维度小于数据库向量维度，使用零填充
                emb = np.pad(emb, ((0, 0), (0, self.arr.shape[1] - emb.shape[1])), mode='constant')

        # Normalize embeddings
        emb_norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        emb_norm = np.array(emb_norm, dtype=np.float16)

        # Calculate similarities: shape (db_size, num_query_embs)
        all_sims = self.arr @ emb_norm.T

        # For each database entry, take the maximum similarity across all query embeddings
        sims = np.max(all_sims, axis=1)
        sims = np.clip((sims + 1) / 2, 0, 1)

        # Get the top k indices
        topk_indices = np.argsort(sims)[::-1]

        nearest = []
        keys = set()

        for idx in range(min(len(topk_indices), k * 2)):  # 获取更多结果用于去重
            i = topk_indices[idx]
            # Convert to Python int if it's a numpy type
            if hasattr(i, 'item'):
                i = i.item()
            i = int(i)  # Ensure it's a Python int
            if dedup:
                key = (self.metadata[i][0], self.metadata[i][1])  # 文件名和源
                if key in keys:
                    continue
                keys.add(key)
            nearest.append((sims[i], i))
            if len(nearest) >= k:
                break

        print(f"Returning {len(nearest)} results from query_nearest")
        return nearest

if __name__ == "__main__":
    db = VectorDB()
    db.load_all(record_tasks=True)
    db.complete_todos(chunk_size=16)
    print('总向量条数：', len(db.arr))
    print('总题目文件数：', len({m[0] for m in db.metadata}))