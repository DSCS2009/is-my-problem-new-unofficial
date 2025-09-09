import numpy as np
from .embedder import VectorDB, processed_promptmd5
from .utils import read_problem
from tqdm.auto import tqdm
import gradio as gr
import json
import asyncio
from openai import AsyncOpenAI
from together import AsyncTogether
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn
import urllib
import time
import voyageai
from concurrent.futures import ThreadPoolExecutor
import requests
import numpy as np

LLAMA_CPP_EMBEDDING_URL = "http://localhost:11436/embedding"   # 与 [2] 保持一致

def get_llama_cpp_embeddings(texts, model="/root/Qwen3/Qwen3-Embedding-0.6B-Q8_0.gguf"):
    """批量调用 llama.cpp embedding 接口，返回 List[np.ndarray]"""
    if not texts:
        return []
    try:
        resp = requests.post(
            LLAMA_CPP_EMBEDDING_URL,
            json={"model": model, "content": list(texts)},
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        resp.raise_for_status()
        return [np.array(item["embedding"], dtype=np.float32) for item in resp.json()]
    except Exception as e:
        print("llama.cpp embedding error:", e)
        # fallback：返回与输入等长的零向量
        return [np.zeros(1024, dtype=np.float32) for _ in texts]


executor = ThreadPoolExecutor(max_workers=8)

db = VectorDB()
db.load_all()
print("read", len(set(x[0] for x in db.metadata)), "problems")
print(db.metadata[:100])

with open("settings.json") as f:
    settings = json.load(f)

voyage_client = voyageai.Client(
    api_key=settings['VOYAGE_API_KEY'],
    max_retries=3,
    timeout=120,
)

# 创建本地llama.cpp客户端
llamacpp_client = AsyncOpenAI(
    base_url="http://localhost:11435/v1",
    api_key="sk-no-key-required"  # llama.cpp通常不需要API密钥
)

# openai_client = AsyncOpenAI(
#     api_key=settings["OPENAI_API_KEY"],
# )

together_client = AsyncTogether(
    api_key=settings['TOGETHER_API_KEY'],
)

async def querier_i18n(locale, statement, *template_choices):
    assert len(template_choices) % 3 == 0
    yields = []
    ORIGINAL = statement.strip()
    t1 = time.time()

    async def process_template(engine, prompt, prefix):
        if 'origin' in engine.lower() or '保' in engine.lower():
            return ORIGINAL
        if 'none' in engine.lower() or '跳' in engine.lower():
            return ''

        prompt = prompt.replace("[[ORIGINAL]]", ORIGINAL).strip()
        
        if False and "gemma" in engine.lower():
            response = await together_client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": prefix}
                ],
                model="google/gemma-2-27b-it",
            )
            return response.choices[0].message.content.strip()
        elif False and "gpt" in engine.lower():
            # response = await openai_client.chat.completions.create(
            #     messages=[
            #         {"role": "system", "content": "You are a helpful assistant."},
            #         {"role": "user", "content": prompt},
            #     ],
            #     model="gpt-4o-mini"
            # )
            return response.choices[0].message.content.strip().replace(prefix.strip(), '', 1).strip()
        elif True or "llama" in engine.lower() or "local" in engine.lower():
            # 使用本地llama.cpp模型
            response = await llamacpp_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that simplifies programming problem statements."},
                    {"role": "user", "content": prompt + ' /no_think'},
                ],
                model="/root/Qwen3/Qwen3-0.6B-Q8_0.gguf",
                temperature=0.1,
                top_p=0.9,
                max_tokens=4096
            )
            return response.choices[0].message.content.strip().replace(prefix.strip(), '', 1).strip()
        else:
            raise NotImplementedError(engine)

    tasks = [process_template(template_choices[i], template_choices[i+1], template_choices[i+2]) 
             for i in range(0, len(template_choices), 3)]
    yields = await asyncio.gather(*tasks)

    t2 = time.time()
    unique_texts = list(set(y.strip() for y in yields if len(y)))
    print(f"Generated {len(unique_texts)} unique texts: {unique_texts}")
    
    embs = get_llama_cpp_embeddings(unique_texts)
    print(f"Generated {len(embs)} embeddings")
    
    # 检查嵌入向量维度
    if embs and len(embs) > 0:
        print(f"Embedding dimension: {len(embs[0])}")
    
    print('Token spent (llama.cpp), texts:', len(unique_texts))
    t3 = time.time()
    print('query emb', t3-t2)

    loop = asyncio.get_running_loop()
    if embs:
        # 使用所有嵌入向量进行查询，取最大相似度
        all_nearest = []
        for emb in embs:
            nearest = await loop.run_in_executor(executor, db.query_nearest, emb, 100)  # 获取更多结果
            all_nearest.extend(nearest)
            print(f"Found {len(nearest)} results for one embedding")
        
        # 合并所有结果并取每个问题的最大相似度
        results_by_key = {}
        for sim, idx in all_nearest:
            key = (db.metadata[idx][0], db.metadata[idx][1])  # 使用文件名和源作为唯一标识
            if key not in results_by_key or sim > results_by_key[key][0]:
                results_by_key[key] = (sim, idx)
        
        # 按相似度排序并取前50个
        nearest = sorted(results_by_key.values(), key=lambda x: x[0], reverse=True)[:50]
        print(f"After deduplication, found {len(nearest)} unique results")
    else:
        nearest = []
        print("No embeddings generated")
    
    t4 = time.time()
    print('query nearest', t4-t3)

    sim = np.array([x[0] for x in nearest])
    ids = np.array([x[1] for x in nearest], dtype=np.int32)

    info = 'Fetched top ' + str(len(sim)) + ' matches! Go to the next tab to view results~' if locale == 'en' else \
           '已查找到前' + str(len(sim)) + '个匹配！进入下一页查看结果~'
    print('【1】raw nearest 50:', len(nearest))
    print('【2】sim 数组长度:', sim.shape if sim.size else 0)
    return [info, (sim, ids)] + yields


def format_problem_i18n(locale, uid, sim):
    def tr(en,zh):
        if locale == 'en': return en
        if locale == 'zh': return zh
        raise NotImplementedError(locale)
    
    # Convert sim to scalar if it's a numpy array
    if hasattr(sim, 'item'):
        sim_scalar = sim.item()
    else:
        sim_scalar = sim
    
    # be careful about arbitrary reads
    uid = db.metadata[int(uid)][0]
    problem = read_problem(uid)
    statement = problem["statement"].replace("\n", "\n\n")
    # summary = sorted(problem.get("processed",[]), key=lambda t: t["template_md5"])
    # if len(summary):
    #     summary = summary[0]["result"]
    # else:
    #     summary = None
    title = problem['title']
    lang = problem.get('locale',('un', 'Unknown'))
    def to_flag(t,u):
        if t == 'un':
            # get a ? with border, 14x20
            return f"""<div style="display: inline-block; border: 1px solid black; width: 20px; text-align: center;" alt="{u}" title="{u}">?</div>"""
        else:
            return f"""<img
    src="https://flagcdn.com/w20/{t}.png"
    srcset="https://flagcdn.com/w40/{t}.png 2x"
    style="display: inline-block"
    height="14"
    width="20"
    title="{u}"
    alt="{u}" />"""
    # flag = ''.join(to_flag(t) for t in lang_mapper.values()) # debug only
    flag = to_flag(*lang)
    url = problem["url"]
    problemlink = uid.replace('/',' ').replace('\\',' ').strip().replace('problems vjudge','',1).strip().replace('_','-')
    assert problemlink.endswith('.json')
    problemlink = problemlink[:-5].strip()
    # markdown = f"# [{title} ({problemlink})]({url})\n\n"
    html = f'<p><span style="font-size:22px; font-weight: 500;">{title}</span>&nbsp;&nbsp;<span style="font-size:15px">{problemlink} ({round(sim_scalar*100)}%)</span></p>\n'
    link0 = 'https://www.google.com/search?'+urllib.parse.urlencode({'q': problemlink})
    link1 = 'https://www.google.com/search?'+urllib.parse.urlencode({'q': problem['source']+' '+title})
    link0_bd = 'https://www.baidu.com/s?'+urllib.parse.urlencode({'wd': problemlink})
    link1_bd = 'https://www.baidu.com/s?'+urllib.parse.urlencode({'wd': problem['source']+' '+title})
    # <a href="{link0}" target="_blank">{tr("Google2","谷歌2")}</a> 
    #  <a href="{link0_bd}" target="_blank">Baidu2</a>
    html += f'{flag}&nbsp;&nbsp;&nbsp;<a href="{url}" target="_blank">VJudge</a>&nbsp;&nbsp;<a href="{link1}" target="_blank">{tr("Google","谷歌")}</a>&nbsp;&nbsp;<a href="{link1_bd}" target="_blank">{tr("Baidu","百度")}</a>'
    markdown = ''
    rsts = []
    for template in settings['TEMPLATES']:
        md5 = processed_promptmd5(problem['statement'], template)
        rst = None
        for t in problem.get("processed",[]):
            if t["prompt_md5"][:8] == md5:
                rst = t["result"]
        if rst is not None:
            rsts.append(rst)
    rsts.sort(key=len)
    for idx, rst in enumerate(rsts):
        markdown += f'### {tr("Summary", "简要题意")} {idx+1}\n\n{rst}\n\n'
    if markdown != '':
        markdown += '<br/>\n\n'
    markdown += f'### {tr("Raw Statement", "原始题面")}\n\n{statement}'
    return html, markdown

def get_block(locale):
    def tr(en,zh):
        if locale == 'en': return en
        if locale == 'zh': return zh
        raise NotImplementedError(locale)

    with gr.Blocks(
        title=tr("Is my problem new?","原题机"), css="""
        .mymarkdown {font-size: 15px !important}
        footer{display:none !important}
        .centermarkdown{text-align:center !important}
        .pagedisp{text-align:center !important; font-size: 20px !important}
        .realfooter{color: #888 !important; font-size: 14px !important; text-align: center !important;}
        .realfooter a{color: #888 !important;}
        .smallbutton {min-width: 30px !important;}
        """,
        head=settings.get('CUSTOM_HEADER','')
    ) as demo:
        gr.Markdown(
            tr("""
        # Is my problem new?
        A semantic search engine for competitive programming problems.
        ""","""
# 原题机
原题在哪里啊，原题在这里~"""
        ))
        with gr.Tabs() as tabs:
            with gr.TabItem(tr("Search",'搜索'),id=0):
                input_text = gr.TextArea(
                    label=tr("Statement",'题目描述'),
                    info=tr("Paste your statement here!",'在这里粘贴你要搜索的题目！'),
                    value=tr("Calculate the longest increasing subsequence of the input sequence.",
                             '计算最长上升子序列长度。'),
                )
                bundles = []
                with gr.Accordion(tr("Rewriting Setup (Advanced)","高级设置"), open=False):
                    gr.Markdown(tr("Several rewritten version of the original statement will be calculated and the maximum embedding similarity is used for sorting.",
                                   "输入的问题描述将被重写为多个版本并计算与每个原问题的最大相似度。"))
                    for template_id in range(5):
                        with gr.Accordion(tr("Template ",'版本 ')+str(template_id+1)):
                            with gr.Row():
                                with gr.Group():
                                    template = settings['TEMPLATES'][(template_id-1)%2] if template_id in [1] else None
                                    # 添加本地LLM选项
                                    engines = [tr("Keep Original",'保留原描述'), "GPT4o Mini", "Local LLM (Qwen3)", tr('None', '跳过该版本')]
                                    engine = gr.Radio(
                                        engines,
                                        label=tr("Engine",'使用的语言模型'),
                                        value=engines[-1] if template is None else engines[2],  # 默认使用本地LLM
                                        interactive=True,
                                    )
                                    prompt = gr.TextArea(
                                        label=tr("Prompt ([[ORIGINAL]] will be replaced)",'提示词 ([[ORIGINAL]] 将被替换为问题描述)'),
                                        value=template if template is not None else settings['TEMPLATES'][0],
                                        interactive=True,
                                        visible=template is not None,
                                    )
                                    prefix = gr.Textbox(
                                        label=tr("Prefix", '回复前缀'),
                                        value="Simplified statement:",
                                        interactive=True,
                                        visible=template is not None,
                                    )
                                    # 修改可见性条件，包含本地LLM
                                    engine.change(lambda engine: (gr.update(visible=any(s in engine.lower() for s in ['gpt','local','llama'])),)*2, engine, prompt, prefix)
                                output_text = gr.TextArea(
                                    label=tr('Output','重写结果'),
                                    value="",
                                    interactive=False,
                                )
                        bundles.append((engine, prompt, prefix, output_text))
                search_result = gr.State(([],[]))
                submit_button = gr.Button(tr("Search!",'搜索！'))
                status_text = gr.Markdown("", elem_classes="centermarkdown")
            with gr.TabItem(tr("View Results",'查看结果'),id=1):
                cur_idx = gr.State(0)
                num_columns = gr.State(50)
                ojs = [f'{t} ({c})' for t,c in sorted(db.sources.items())]
                oj_dropdown = gr.Dropdown(
                    ojs, value=ojs, multiselect=True, label=tr("Displayed OJs",'展示的OJ'),
                    info=tr('Problems from OJ not in this list will be ignored.',
                            '不在这个列表里的OJ的题目将被忽略。可以在这里删掉你不认识的OJ。'),
                )
                # on change, change cur_idx to 1
                oj_dropdown.change(lambda: 0, None, cur_idx)
                statement_min_len = gr.Slider(
                    minimum=1,
                    maximum=1000,
                    label=tr("Minimum Statement Length",'最小题面长度'),
                    value=20,
                    info=tr('The statements shorter than this after removing digits + blanks will be ignored. Useful for filtering out meaningless statements.',
                            '去除数字和空白字符后题面长度小于该值的题目将被忽略。可以用来筛掉一些奇怪的题面。'),
                )

                with gr.Row():
                    # home_page = gr.Button("H")
                    add_column = gr.Button("+", elem_classes='smallbutton')
                    prev_page = gr.Button("←", elem_classes='smallbutton')
                    home_page = gr.Button("H", elem_classes='smallbutton')
                    next_page = gr.Button("→", elem_classes='smallbutton')
                    remove_column = gr.Button("-", elem_classes='smallbutton')
                    # bind to cur_page and num_columns
                    # home_page.click(lambda: 1, None, cur_page)
                    prev_page.click(lambda cur_idx, num_columns: max(cur_idx - num_columns, 0), [cur_idx, num_columns], cur_idx, concurrency_limit=None)
                    next_page.click(lambda cur_idx, num_columns: cur_idx + num_columns, [cur_idx, num_columns], cur_idx, concurrency_limit=None)
                    home_page.click(lambda: 0, None, cur_idx, concurrency_limit=None)
                    def adj_idx(idx, col):
                        return int(round(idx / col)) * col
                    add_column.click(lambda cur_idx, num_columns: (adj_idx(cur_idx, num_columns + 1), num_columns + 1), [cur_idx, num_columns], [cur_idx, num_columns], concurrency_limit=None)
                    remove_column.click(lambda cur_idx, num_columns: (adj_idx(cur_idx, num_columns - 1), num_columns - 1) if num_columns >1 else (cur_idx, num_columns), [cur_idx, num_columns], [cur_idx, num_columns], concurrency_limit=None)


                @gr.render(inputs=[search_result, oj_dropdown, cur_idx, num_columns, statement_min_len], concurrency_limit=None)
                def show_OJs(search_result, oj_dropdown, cur_idx, num_columns, statement_min_len):
                    allowed_OJs = set([oj[:oj.find(' (')] for oj in oj_dropdown])
                    filtered_results = []
                    
                    # 过滤结果
                    for sim, idx in zip(search_result[0], search_result[1]):
                        if db.metadata[idx][1] not in allowed_OJs or db.metadata[idx][2] < statement_min_len:
                            continue
                        filtered_results.append((sim, idx))
                    
                    tot = len(filtered_results)
                    total_pages = (tot + num_columns - 1) // num_columns if tot > 0 else 1
                    current_page = (cur_idx // num_columns) + 1 if tot > 0 else 1
                    
                    gr.Markdown(tr(f"Page {current_page} of {total_pages} ({num_columns} per page, total {tot} results)",
                                   f'第 {current_page} 页 / 共 {total_pages} 页 (每页显示 {num_columns} 个，共 {tot} 个结果)'),
                                   elem_classes="pagedisp")
                    
                    # 显示当前页的结果
                    start_idx = cur_idx
                    end_idx = min(cur_idx + num_columns, tot)
                    
                    with gr.Row():
                        for i in range(start_idx, end_idx):
                            sim, idx = filtered_results[i]
                            with gr.Column(variant='compact'):
                                html, md = format_problem_i18n(locale, idx, sim)
                                gr.HTML(html)
                                gr.Markdown(
                                    latex_delimiters=[
                                        {"left": "$$", "right": "$$", "display": True},
                                        {"left": "$", "right": "$", "display": False},
                                        {"left": "\\(", "right": "\\)", "display": False},
                                        {"left": "\\[", "right": "\\]", "display": True},
                                    ],
                                    value=md,
                                    elem_classes="mymarkdown",
                                )
            if 'CUSTOM_ABOUT_PY' in settings and settings['CUSTOM_ABOUT_PY'].endswith('.py'):
                with gr.TabItem(tr("About",'关于'),id=2):
                    with open(settings['CUSTOM_ABOUT_PY'], 'r', encoding='utf-8') as f: eval(f.read())

        # add a footer
        gr.HTML(
            """<div class="realfooter">Built with ❤️ by <a href="https://github.com/fjzzq2002">@TLE</a></div>"""
        )
        async def async_querier_wrapper(*args):
            result = await querier_i18n(locale, *args)
            return (gr.Tabs(selected=1),) + tuple(result)
        submit_button.click(
            fn=async_querier_wrapper,
            inputs=sum([list(t[:-1]) for t in bundles], [input_text]),
            outputs=[tabs, status_text, search_result] + [t[-1] for t in bundles],
            concurrency_limit=7,
        )
        # output_labels.select(fn=show_problem, inputs=None, outputs=[my_markdown])
    return demo



app = FastAPI()
favicon_path = 'favicon.ico'

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)
@app.get("/", response_class=HTMLResponse)
async def read_main():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Is my problem new?</title>
        <script type="text/javascript">
            window.onload = function() {
                var userLang = navigator.language || navigator.userLanguage;
                if (userLang.startsWith('zh')) {
                    window.location.href = "/zh";
                } else {
                    window.location.href = "/en";
                }
            }
        </script>
    </head>
    <body>
        <p>Redirecting based on your browser's locale...</p>
        <p><a href="/en">English</a> | <a href="/zh">中文</a></p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

app = gr.mount_gradio_app(app, get_block('zh'), path="/zh")
app = gr.mount_gradio_app(app, get_block('en'), path="/en")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8090)