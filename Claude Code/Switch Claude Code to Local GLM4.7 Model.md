---
Date: 2026-06-08
tags:
  - ClaudeCode
---
## Step1: 测试GLM4.7 API的可用性和工具调用支持

### 1. 查一下可用的模型列表
```sh
curl -s http://172.18.1.13:38080/v1/models | python3 -m json.toolool
```

![](https://pic1.imgdb.cn/item/6a262bd067f23bae9b0069e8.png)
模型名称为 `glm-4.7`，运行在 **vLLM** 上，最大上下文长度 **202,752 tokens**

### 2. 测试基础对话能力
```sh
curl -s http://172.18.1.13:38080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4.7",
    "messages": [
      {"role": "system", "content": "你是一个有帮助的助手。"},
      {"role": "user", "content": "你好，请用一句话回复我"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }' | python3 -m json.tool
```
![](https://pic1.imgdb.cn/item/6a262be567f23bae9b0069fb.png)
### 3. 测试工具调用/函数调用
```sh
curl -s http://172.18.1.13:38080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4.7",
    "messages": [
      {"role": "system", "content": "你是一个有帮助的助手。"},
      {"role": "user", "content": "北京的天气怎么样？帮我查一下。"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "获取指定城市的天气信息",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {
                "type": "string",
                "description": "城市名称"
              }
            },
            "required": ["city"]
          }
        }
      }
    ],
    "temperature": 0.7,
    "max_tokens": 200
  }' | python3 -m json.tool
```

### 4. 测试一下**流式输出**和**图片理解**支持
```sh
curl -s http://172.18.1.13:38080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4.7",
    "messages": [
      {"role": "system", "content": "你是一个有帮助的助手。"},
      {"role": "user", "content": "说一句话，测试流式输出"}
    ],
    "stream": true,
    "max_tokens": 50
  }' | head -c 2000
```
流式输出正常（SSE格式）
```sh
# Test with a small base64 image (1x1 pixel PNG) to check if GLM-4.7 accepts image inputs
# Use the smallest valid PNG
SMALL_PNG=$(curl -s http://172.18.1.13:38080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4.7",
    "messages": [
      {"role": "user", "content": [
        {"type": "text", "text": "描述一下这张图片"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="}}
      ]}
    ],
    "max_tokens": 100
  }' 2>&1)
echo "$SMALL_PNG" | python3 -m json.tool 2>/dev/null || echo "$SMALL_PNG"
```
![|584x315](https://pic1.imgdb.cn/item/6a262c2967f23bae9b006a33.png)
**LM-4.7 不支持多模态（图片）**，它只是一个纯文本模型。

## Step2：编写Anthropic→OpenAI代理脚本

FastAPI + uvicorn + httpx 都可用

```python
python -c "
import importlib
for mod in ['fastapi', 'uvicorn', 'httpx', 'flask', 'aiohttp', 'starlette', 'pydantic', 'uvloop']:
    try:
        importlib.import_module(mod)
        print(f'{mod}: available')
    except ImportError:
        print(f'{mod}: not available')
"
```

![](https://pic1.imgdb.cn/item/6a262c5367f23bae9b006a4e.png)
用已有的 FastAPI + httpx 写一个轻量级的 Anthropic→OpenAI 代理

```python
#!/usr/bin/env python3
"""
Anthropic Messages API → OpenAI Chat Completions API Proxy
=========================================================
Makes a local OpenAI-compatible model (GLM-4.7 via vLLM) look like the
Anthropic Messages API so Claude Code can use it as a drop-in backend.

Usage:
    python anthropic_proxy.py --port 38081 --backend http://172.18.1.13:38080/v1

Supported:
    - /v1/messages (non-streaming + SSE streaming)
    - /v1/models
    - Tool use (function calling) round-trip
    - System prompt mapping

Caveats:
    - Streaming: Anthropic SSE event types differ from OpenAI chunk streams.
      We emit the minimal set that Claude Code's client expects.
    - Image/vision content blocks are passed through but the backend model
      (GLM-4.7) is text-only — multimodal requests will error at the backend.
"""

from __future__ import annotations

import json
import time
import uuid
import argparse
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

app = FastAPI(title="Anthropic→OpenAI Proxy")

BACKEND_BASE: str = ""
BACKEND_MODEL: str = ""
HTTP_CLIENT: httpx.AsyncClient | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def anthropic_to_openai_messages(
    messages: list[dict],
    system: str | list | None = None,
) -> list[dict]:
    """Convert Anthropic message list → OpenAI message list."""
    openai_msgs: list[dict] = []

    # System prompt
    if system:
        if isinstance(system, str):
            openai_msgs.append({"role": "system", "content": system})
        elif isinstance(system, list):
            # Anthropic system can be an array of text blocks
            parts = []
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block["text"])
                elif isinstance(block, str):
                    parts.append(block)
            if parts:
                openai_msgs.append({"role": "system", "content": "\n".join(parts)})

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Map Anthropic role → OpenAI role
        if role == "user":
            oai_role = "user"
        elif role == "assistant":
            oai_role = "assistant"
        else:
            oai_role = "user"  # fallback

        if isinstance(content, str):
            openai_msgs.append({"role": oai_role, "content": content})
        elif isinstance(content, list):
            # Anthropic content blocks → OpenAI content array
            oai_content: list[dict] = []
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "text":
                        oai_content.append({"type": "text", "text": block.get("text", "")})
                    elif btype == "image":
                        src = block.get("source", {})
                        media_type = src.get("media_type", "image/png")
                        data = src.get("data", "")
                        oai_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{data}"},
                        })
                    elif btype == "tool_use":
                        # Anthropic tool_use → OpenAI tool_calls (on assistant msg)
                        if oai_role == "assistant":
                            tc = {
                                "id": block.get("id", f"call_{uuid.uuid4().hex[:12]}"),
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "arguments": json.dumps(block.get("input", {})),
                                },
                            }
                            if "tool_calls" not in openai_msgs[-1] if openai_msgs else False:
                                pass  # handled below
                            # We embed tool_calls in the assistant message
                            openai_msgs.append({
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [tc],
                            })
                            continue  # skip regular append
                    elif btype == "tool_result":
                        openai_msgs.append({
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": block.get("content", ""),
                        })
                        continue  # skip regular append
                oai_content.append({"type": "text", "text": str(block)})
            if oai_content:
                openai_msgs.append({"role": oai_role, "content": oai_content})
        else:
            openai_msgs.append({"role": oai_role, "content": str(content)})

    return openai_msgs


def anthropic_tools_to_openai(tools: list[dict] | None) -> list[dict] | None:
    """Convert Anthropic tool definitions → OpenAI tool definitions."""
    if not tools:
        return None
    oai_tools = []
    for tool in tools:
        oai_tools.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        })
    return oai_tools


def openai_response_to_anthropic(oai_resp: dict, model: str) -> dict:
    """Convert OpenAI chat completion → Anthropic Messages response."""
    choice = oai_resp.get("choices", [{}])[0]
    oai_msg = choice.get("message", {})
    finish_reason = choice.get("finish_reason", "stop")

    anthropic_content: list[dict] = []

    # Text content
    text = oai_msg.get("content")
    if text:
        anthropic_content.append({"type": "text", "text": text})

    # Tool calls
    tool_calls = oai_msg.get("tool_calls") or []
    for tc in tool_calls:
        try:
            arguments = json.loads(tc["function"]["arguments"])
        except (json.JSONDecodeError, TypeError, KeyError):
            arguments = {}
        anthropic_content.append({
            "type": "tool_use",
            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
            "name": tc["function"]["name"],
            "input": arguments,
        })

    stop_reason = "end_turn"
    if finish_reason == "tool_calls":
        stop_reason = "tool_use"
    elif finish_reason == "length":
        stop_reason = "max_tokens"
    elif finish_reason == "stop":
        stop_reason = "end_turn"

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": anthropic_content,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": oai_resp.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": oai_resp.get("usage", {}).get("completion_tokens", 0),
        },
    }


async def openai_stream_to_anthropic_sse(
    stream: AsyncIterator[bytes],
    model: str,
) -> AsyncIterator[str]:
    """Convert OpenAI SSE chunk stream → Anthropic SSE event stream."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    content_block_idx = 0
    tool_use_idx: dict[int, str] = {}
    started = False
    pending_tool_calls: dict[int, dict] = {}

    async for chunk_bytes in stream:
        text = chunk_bytes.decode("utf-8", errors="replace")
        for line in text.split("\n"):
            line = line.strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break

            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue

            choices = data.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})

            finish_reason = choices[0].get("finish_reason")

            # --- message_start ---
            if not started:
                started = True
                yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model}})}\n\n"
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

            # Skip reasoning-only deltas
            if delta.get("reasoning_content") and not delta.get("content") and not delta.get("tool_calls"):
                continue

            # Text delta
            text_delta = delta.get("content")
            if text_delta:
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text_delta}})}\n\n"

            # Tool call deltas
            tc_deltas = delta.get("tool_calls") or []
            for tc_delta in tc_deltas:
                idx = tc_delta.get("index", 0)
                tc_id = tc_delta.get("id")
                tc_func = tc_delta.get("function", {})

                # New tool call
                if tc_id and idx not in tool_use_idx:
                    tool_use_idx[idx] = tc_id
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': idx + 1, 'content_block': {'type': 'tool_use', 'id': tc_id, 'name': tc_func.get('name', '')}})}\n\n"

                # Tool call argument delta
                if tc_func.get("arguments"):
                    pending_tool_calls[idx] = True
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': idx + 1, 'delta': {'type': 'input_json_delta', 'partial_json': tc_func['arguments']}})}\n\n"

            # Tool call content blocks end
            if finish_reason:
                for idx in tool_use_idx:
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': idx + 1})}\n\n"

            # --- message_delta (stop_reason) ---
            if finish_reason:
                stop_reason = "tool_use" if finish_reason == "tool_calls" else "end_turn"
                if finish_reason == "length":
                    stop_reason = "max_tokens"
                yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
                yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/messages")
async def messages(request: Request):
    """Handle Anthropic /v1/messages and proxy to OpenAI /v1/chat/completions."""
    body = await request.json()

    messages = body.get("messages", [])
    system = body.get("system")
    tools = body.get("tools")
    model = body.get("model", BACKEND_MODEL)
    max_tokens = body.get("max_tokens", 4096)
    temperature = body.get("temperature", 0.7)
    top_p = body.get("top_p", 1.0)
    stream = body.get("stream", False)
    # tool_choice is handled below

    openai_body: dict[str, Any] = {
        "model": BACKEND_MODEL,
        "messages": anthropic_to_openai_messages(messages, system),
        "max_tokens": min(max_tokens, 32000),
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
    }

    oai_tools = anthropic_tools_to_openai(tools)
    if oai_tools:
        openai_body["tools"] = oai_tools
        # tool_choice mapping
        tool_choice = body.get("tool_choice")
        if tool_choice:
            if isinstance(tool_choice, dict) and tool_choice.get("type") == "any":
                openai_body["tool_choice"] = "auto"
            elif isinstance(tool_choice, dict) and tool_choice.get("type") == "tool":
                openai_body["tool_choice"] = {"type": "function", "function": {"name": tool_choice.get("name")}}
            elif tool_choice == "auto":
                openai_body["tool_choice"] = "auto"
            elif tool_choice == "any":
                openai_body["tool_choice"] = "required"

    url = f"{BACKEND_BASE}/chat/completions"

    if stream:
        async def _stream():
            async with httpx.AsyncClient(timeout=600.0) as client:
                async with client.stream("POST", url, json=openai_body) as resp:
                    if resp.status_code != 200:
                        error_body = await resp.aread()
                        raise HTTPException(status_code=resp.status_code, detail=error_body.decode())
                    async for chunk in openai_stream_to_anthropic_sse(resp.aiter_bytes(), model):
                        yield chunk

        return StreamingResponse(_stream(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=600.0) as client:
        resp = await client.post(url, json=openai_body)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        data = resp.json()

    return JSONResponse(openai_response_to_anthropic(data, model))


@app.get("/v1/models")
@app.get("/v1/models/{model_id}")
async def list_models(model_id: str | None = None):
    """Return a minimal model list that Claude Code expects."""
    models_data = {
        "object": "list",
        "data": [
            {
                "id": BACKEND_MODEL,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local-proxy",
            }
        ],
    }
    return JSONResponse(models_data)


@app.get("/health")
async def health():
    return {"status": "ok", "backend": BACKEND_BASE, "model": BACKEND_MODEL}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global BACKEND_BASE, BACKEND_MODEL
    parser = argparse.ArgumentParser(description="Anthropic→OpenAI Proxy for Claude Code")
    parser.add_argument("--port", type=int, default=38081)
    parser.add_argument("--backend", type=str, default="http://172.18.1.13:38080/v1")
    parser.add_argument("--model", type=str, default="glm-4.7")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    BACKEND_BASE = args.backend.rstrip("/")
    BACKEND_MODEL = args.model

    print(f"🚀 Anthropic→OpenAI Proxy")
    print(f"   Backend: {BACKEND_BASE}")
    print(f"   Model:   {BACKEND_MODEL}")
    print(f"   Listen:  http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
```

启动代理并测试
```sh
python anthropic_proxy.py
```

![](https://pic1.imgdb.cn/item/6a262c6b67f23bae9b006a6c.png)
```sh
sleep 2 && curl -s http://localhost:38081/health | python3 -m json.tool
```
![](https://pic1.imgdb.cn/item/6a262c7c67f23bae9b006a80.png)
用 Anthropic 格式请求测试翻译是否正常工作
```sh
curl -s http://localhost:38081/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-dummy" \
  -d '{
    "model": "glm-4.7",
    "max_tokens": 1024,
    "system": "你是一个有帮助的助手。",
    "messages": [
      {"role": "user", "content": "你好，请用一句话回复我"}
    ]
  }' 2>&1 | python3 -m json.tool
```
![](https://pic1.imgdb.cn/item/6a262c8a67f23bae9b006a8c.png)
用 Anthropic 格式请求测试工具调用是否正常工作
```
curl -s http://localhost:38081/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-dummy" \
  -d '{
    "model": "glm-4.7",
    "max_tokens": 1024,
    "system": "你是一个有帮助的助手。",
    "messages": [
      {"role": "user", "content": "帮我查一下北京的天气"}
    ],
    "tools": [
      {
        "name": "get_weather",
        "description": "查询城市天气",
        "input_schema": {
          "type": "object",
          "properties": {
            "city": {"type": "string", "description": "城市名称"}
          },
          "required": ["city"]
        }
      }
    ]
  }' | python3 -m json.tool
```

测试流式输出（Claude Code 依赖 SSE 流式）
```sh
curl -s http://localhost:38081/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-dummy" \
  -d '{
    "model": "glm-4.7",
    "max_tokens": 256,
    "stream": true,
    "system": "你是一个有帮助的助手。",
    "messages": [
      {"role": "user", "content": "你好，请用一句话回复我"}
    ]
  }' 2>&1 | head -30
```

## Step3：更新~/.claude.json配置指向本地模型

```json
"env": {
    "ANTHROPIC_BASE_URL": "http://localhost:38081",
    "ANTHROPIC_AUTH_TOKEN": "sk-dummy",
    "ANTHROPIC_MODEL": "glm-4.7",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "glm-4.7",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "glm-4.7",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "glm-4.7",
    "CLAUDE_CODE_SUBAGENT_MODEL": "glm-4.7",
    "CLAUDE_CODE_EFFORT_LEVEL": "max"
  }
```

## Step4：测试claude code
![](https://pic1.imgdb.cn/item/6a262c9b67f23bae9b006a9d.png)
