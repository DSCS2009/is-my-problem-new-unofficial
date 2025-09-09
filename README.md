# Is my problem new?
A simple semantic search engine on competitive programming problems using llama.cpp.
<a href="http://yuantiji.ac" target="_blank" style="color: blue">http://yuantiji.ac</a> | <a href="https://www.buymeacoffee.com/fjzzq2002" target="_blank" style="color: brown">Buy me a boba</a>

<img src="logo.gif" style="zoom:50%;" />

**Update (2024/7/16):** It has been a long time :) Reorganized problems path. Switched LLM / embedder to [Gemma 2 9B](https://huggingface.co/google/gemma-2-9b-it) hosted by [together.ai](https://docs.together.ai) and [voyage-large-2-instruct](https://docs.voyageai.com/docs/pricing). Tweaked the prompt a little bit. Bought a new domain (see the link above) and switched to [vjudge](https://vjudge.net) as data source. See branch `old_ver` or history commits for the previous version.

**Update (2024/5/19):** Added AtCoder. Thanks [@fstqwq](https://github.com/fstqwq) for the contribution!

#### How does this work?

This idea is simple:

1. Simplify the statement & remove background by prompting LLM.

2. Embed the simplified documents and queries to perform vector searches.

It only happens recently that both models are good and cheap enough.

This pipeline is also not limited, of course, to competitive programming problems. You can use it to search for any kind of documents by modifying the prompt.

#### Deploy

You need a computer with more than 6GB RAM or vRAM.

First, install [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).

You will need 2 llama.cpp servers, Qwen3-0.6B and Qwen3-Embedding-0.6B. You can download them online.

You need to run these commands to run the servers.
```bash
./build/bin/llama-server -m ./Qwen3-0.6B-Q8_0.gguf --ctx-size 8192 --port 11435
./build/bin/llama-server -m ./Qwen3-Embedding-0.6B-Q8_0.gguf --port 11436 --embedding --pooling last -ub 8192 --verbose-prompt
```

Then, you should replace the server URLs and model names in `src/build_summary.py`, `src/embedding.py` and `src/ui.py`.

Put problems in `problems/` folder following the given example (`problems/1000.json`). Naming could be arbitrary and you could also have nested folders. Run `python -m src.build_summary` to get paraphrased statements, run `python -m src.embedder` to build embeddings and run `python -m src.build_locale` to detect language of problems. Finally, run `python -m src.ui` to start serving.

For large-scale running decent CPUs are needed as doing vector searching is CPU-dense. You might also want to modify `max_workers` in `src/ui.py`.

For reference, adding all ~34k problems from [luogu](https://www.luogu.com.cn/) cost ~$0.1 and as of writing the deployed site is running on a 8vCPU server.

---

### Note: 

You can get the problems from [LuoguCrawler](https://github.com/tommyjink/LuoguCrawler).

You can format the problem with `src/collect.py`.