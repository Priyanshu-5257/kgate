# KGate

Use a Kaggle GPU from this laptop. KGate pushes a private notebook on an account you own, gives you a shell on that machine, and forwards an OpenAI-compatible API to `http://127.0.0.1:8000/v1`.

A local dashboard shows quota for every configured account, the sessions started from this machine, and the live notebook log.

GPU time stays on each Kaggle account. A session spends quota for the whole time the notebook is up, including while the model is idle. Stop it when you are done.

## How it works

```
laptop                         Kaggle notebook (private, GPU, internet)
------                         -----------------------------------------
kgate CLI                      session script
dashboard :8787                gateway :8788  (auth, shell, /v1 proxy)
OpenAI client :8000  ------->  Cloudflare quick tunnel
kgate attach                   bash on a PTY, optional vLLM / Ollama / SGLang
```

1. `kgate up` writes a private kernel and pushes it with the Kaggle CLI.
2. The notebook downloads `cloudflared` and opens a quick tunnel.
3. The laptop reads that URL from Kaggle's **live** log stream. The saved log stays empty until the notebook finishes, so `kgate logs` and the dashboard follow the stream.
4. A proxy on `127.0.0.1` adds the session bearer token and streams responses, including chat completions.
5. `kgate attach` opens a WebSocket to the same tunnel and connects to one bash on the notebook.

The tunnel address changes every session. The local port does not. Prompt text crosses Cloudflare's network. The bearer token is written into the private kernel source, so leave the notebook private.

## Requirements

- Python 3.11+
- The [Kaggle CLI](https://github.com/Kaggle/kaggle-cli) on `PATH` (`pip install kaggle`)
- A phone-verified Kaggle account and an API token from <https://www.kaggle.com/settings/api>

KGate itself has no third-party Python dependencies. The dashboard is one HTML page served by the standard library. vLLM, Ollama, SGLang, and cloudflared are installed on the notebook only when a session needs them.

## Install

```bash
pip install -e .
kaggle --version
kgate doctor
```

## Accounts

Tokens are copied into `~/.config/kgate/tokens/` with mode `600`. They are not stored in this repository.

```bash
kgate account add main --token-file ~/.kaggle/access_token
kgate account add alt --token-file ~/secrets/kaggle-alt.token
kgate account list
kgate account use main
kgate quota
```

`kgate quota` prints GPU and TPU hours for every account. On the free tier that is often about 30 GPU hours and 20 TPU hours per account each week. `kgate quota` is the source of truth. The machine you get (often one or two T4s) is whatever Kaggle assigns to the requested shape.

## Quick start

Shell only. Anything you start on port `8000` inside the notebook is already forwarded.

```bash
kgate up
kgate attach          # Ctrl-] disconnects. The notebook keeps running.
kgate down
```

vLLM. This is the default when you pass `--model`. T4 and P100 need `half`, not bfloat16. A 1.5B model fits easily. A 7B model in fp16 is tight on 16GB. An AWQ build is the safer choice, and `--tensor-parallel 2` uses both GPUs when the session has two.

```bash
kgate up --model Qwen/Qwen2.5-1.5B-Instruct

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","messages":[{"role":"user","content":"Say hello in one sentence."}]}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="kgate")
print(client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
).choices[0].message.content)
```

The local API key can be any non-empty string. The proxy replaces it with the session token.

Ollama:

```bash
kgate up --engine ollama --model qwen2.5:7b
```

Larger vLLM model:

```bash
kgate up --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --quantization awq --tensor-parallel 2 --max-model-len 4096
```

Gated Hugging Face models need a token file. It is embedded in the private kernel so the notebook can download the weights.

```bash
kgate up --model meta-llama/Llama-3.1-8B-Instruct --hf-token-file ~/.cache/huggingface/token
```

Extra engine flags need an equals sign so the CLI does not treat them as its own options.

```bash
kgate up --model Qwen/Qwen2.5-1.5B-Instruct --engine-arg=--enforce-eager
```

`kgate up` prints the dashboard URL and waits until the model answers, unless you pass `--no-wait`. Ctrl-C during that wait stops waiting only. `kgate down` stops the notebook. `--hours` defaults to 6 and is shortened when the account has less GPU time left.

One notebook slug is reused per account (`username/kgate-session`). Kaggle refuses a second push while that notebook is queued or running.

## Commands

| Command | What it does |
|---|---|
| `kgate account add\|list\|use\|remove` | Store tokens and pick the default account |
| `kgate quota` | GPU and TPU hours left on each account |
| `kgate up` | Start a private GPU notebook and forward it locally |
| `kgate down` | Stop the notebook and the local proxy |
| `kgate ps` | Sessions started from this laptop |
| `kgate logs` | Live notebook log. `--follow` keeps streaming |
| `kgate attach` | Interactive shell. `Ctrl-]` detaches |
| `kgate exec nvidia-smi` | Run one command on the notebook |
| `kgate dash` | Dashboard at `http://127.0.0.1:8787` |
| `kgate doctor` | Check the Kaggle CLI, accounts, and quota |

## Attach

`kgate attach` uses the terminal you launched it from. It does not open a new window.

The first attach starts one interactive bash on the notebook. Later attaches reuse that bash, so the working directory and a foreground process are still there. A new bash is started only after that one exits.

This is not tmux. While you are detached, nothing reads the PTY, so a noisy process can block when the PTY buffer fills, and output from the gap is not replayed. Two attaches at the same time share that one shell. The dashboard command box is separate: each command is a one-off `bash -c`.

## Dashboard

`kgate dash` binds to `127.0.0.1:8787` only.

- Quota bars for every account, with the reset time
- Sessions, engine, model, status, and the local API URL
- Live notebook log
- A box that runs one remote command
- Stop, which asks the notebook to exit and then stops the local proxy

`kgate up` starts the dashboard if it is not already running.

## Layout

```
src/kgate/cli.py          commands
src/kgate/launch.py       kernel metadata, push, wait for the tunnel
src/kgate/kaggle_cli.py   Kaggle CLI subprocess, live log stream
src/kgate/remote_agent.py script that runs inside the notebook
src/kgate/proxy.py        localhost OpenAI proxy
src/kgate/term.py         attach
src/kgate/dashboard.py    usage and log UI
src/kgate/store.py        ~/.config/kgate
```

State on the laptop:

```
~/.config/kgate/accounts.json
~/.config/kgate/tokens/<name>     mode 600
~/.config/kgate/sessions.json
~/.config/kgate/kernels/<id>/     private kernel that was pushed
```

## Development

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The tests do not start a Kaggle notebook and do not spend GPU quota.

## Limits

- Quota is wall-clock time on the accelerator, not tokens generated.
- A new `kgate up` gets a new tunnel URL. The local port stays `8000` unless that port is taken.
- If the tunnel dies before `kgate down` can reach it, stop the notebook on its Kaggle page. Otherwise it keeps spending quota until Kaggle's own limit.
- The first vLLM start downloads a large wheel and the model. The shell is usable as soon as the local URL is printed. `kgate logs --follow` shows the install.
- KGate does not raise the weekly cap and does not keep a session past the hours you set.
