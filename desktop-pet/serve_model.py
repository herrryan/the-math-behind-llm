#!/usr/bin/env python3
"""Standalone Python Model Server for AI Desktop Pet.

Provides an OpenAI-compatible HTTP API (/v1/chat/completions) on port 8765.
Supports 3 inference backends:
1. 'mock' (Default, zero dependencies):
   Lightweight heuristic neural mock for instant testing without downloading weights.
2. 'transformers' (PyTorch + MPS for Apple Silicon GPU):
   Loads any HuggingFace or fine-tuned local weights (e.g., Qwen/Qwen2.5-0.5B-Instruct).
3. 'mlx' (Apple MLX framework):
   Ultra-fast 4-bit native Apple Silicon inference.

Usage:
    # 1. Run mock server (zero pip dependencies):
    python3 serve_model.py --engine mock

    # 2. Run real PyTorch MPS model (loads Qwen2.5-0.5B on Mac GPU):
    uv run --with "transformers>=4.40.0" --with "torch" python3 serve_model.py --engine transformers --model Qwen/Qwen2.5-0.5B-Instruct

    # 3. Direct in-code Python import:
    from serve_model import LocalNeuralModel
    bot = LocalNeuralModel(engine="mock")
    print(bot.generate("用户在终端里遇到了段错误"))
"""

import sys
import os
import json
import argparse
import random
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, List, Optional


class LocalNeuralModel:
    """In-memory neural model engine for direct Python execution."""

    def __init__(self, engine: str = "mock", model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.engine = engine
        self.model_name = model_name_or_path
        self.model = None
        self.tokenizer = None
        self._init_engine()

    def _init_engine(self):
        if self.engine == "transformers":
            print(f"Loading HuggingFace model '{self.model_name}' with PyTorch...")
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Detect Apple Silicon MPS / CUDA / CPU
            if torch.backends.mps.is_available():
                device = "mps"
                dtype = torch.float16
                print(">> Accelerated by Apple Silicon GPU (MPS)!")
            elif torch.cuda.is_available():
                device = "cuda"
                dtype = torch.float16
                print(">> Accelerated by NVIDIA CUDA GPU!")
            else:
                device = "cpu"
                dtype = torch.float32
                print(">> Running on CPU.")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map=device,
            )
            self.device = device
            print(f">> Model successfully loaded into memory on {device}!")

        elif self.engine == "mlx":
            print(f"Loading MLX 4-bit model '{self.model_name}' on Apple Silicon...")
            from mlx_lm import load
            self.model, self.tokenizer = load(self.model_name)
            print(">> MLX model loaded successfully!")

        else:
            print(">> Initialized zero-dependency Mock Neural Engine (Ready).")

    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 50, temperature: float = 0.7) -> str:
        """Generate response given chat messages."""
        if self.engine == "transformers" and self.model and self.tokenizer:
            import torch
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                )
            generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)]
            return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        elif self.engine == "mlx" and self.model and self.tokenizer:
            from mlx_lm import generate
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False).strip()

        else:
            # Smart neural mock generation
            user_msg = ""
            for m in messages:
                if m.get("role") == "user":
                    user_msg = m.get("content", "")

            mock_pool = [
                "我在后台 Python 进程中实时运转呢！有什么代码难题尽管交给我！",
                "Python Server 响应耗时 2ms！你写的那行代码看起来相当优雅！",
                "检测到输入，Python 端侧推理完毕。建议多喝温水，保持节奏！",
                "代码如诗，Bug 如瑕。我们一起把这段逻辑打磨完美！",
            ]
            if "报错" in user_msg or "error" in user_msg.lower():
                return f"Python 神经模型排查建议：检查刚才复制的报错，先确认变量类型与作用域！"
            return random.choice(mock_pool)


class ModelHTTPHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible HTTP handler."""

    model_engine: Optional[LocalNeuralModel] = None

    def log_message(self, format, *args):
        # Clean logging
        sys.stderr.write(f"[Python-Model-Server] {self.command} {self.path} - {args[0]}\n")

    def do_GET(self):
        if self.path in ["/", "/health"]:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = {
                "status": "online",
                "engine": self.model_engine.engine if self.model_engine else "unknown",
                "model": self.model_engine.model_name if self.model_engine else "none",
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)
            try:
                data = json.loads(body_bytes.decode("utf-8"))
                messages = data.get("messages", [])
                max_tokens = data.get("max_tokens", 60)
                temp = data.get("temperature", 0.7)

                start_t = time.time()
                reply_text = self.model_engine.generate(messages, max_tokens=max_tokens, temperature=temp)
                cost_ms = int((time.time() - start_t) * 1000)

                resp_data = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": self.model_engine.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": reply_text,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": len(reply_text),
                        "total_tokens": 10 + len(reply_text),
                    },
                    "_cost_ms": cost_ms,
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(resp_data, ensure_ascii=False).encode("utf-8"))

            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def run_server(port: int = 8765, engine: str = "mock", model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """Launch the Python Model Server."""
    print("=" * 60)
    print(f" Starting Local Python Model Server on port {port}...")
    print(f" - Engine: {engine}")
    print(f" - Model: {model_name}")
    print(f" - API Endpoint: http://localhost:{port}/v1/chat/completions")
    print("=" * 60)

    model_instance = LocalNeuralModel(engine=engine, model_name_or_path=model_name)
    ModelHTTPHandler.model_engine = model_instance

    server_address = ("", port)
    httpd = HTTPServer(server_address, ModelHTTPHandler)
    print(f">> Server listening on http://localhost:{port}")
    print(">> Press Ctrl+C to stop.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n>> Stopping server cleanly.")
        httpd.server_close()


def main():
    parser = argparse.ArgumentParser(description="Standalone Python Model Server for Desktop Pet")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    parser.add_argument("--engine", choices=["mock", "transformers", "mlx"], default="mock", help="Inference backend engine")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="HuggingFace model ID or local weights path")
    args = parser.parse_args()

    run_server(port=args.port, engine=args.engine, model_name=args.model)


if __name__ == "__main__":
    main()
