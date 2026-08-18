#!/usr/bin/env python3
"""IndicLLM-Bharat — Interactive Streaming Web Playground.

FastAPI-powered single-page interactive chat interface and generation playground
supporting SSE streaming, 22 Indian language prompts, and parameter controls.

Usage:
  # Launch with trained checkpoint
  python inference/playground.py --checkpoint checkpoints/bharat-350m/final.pt --port 7860

  # Launch with synthetic tiny model for testing
  python inference/playground.py --model-size tiny --port 7860
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch  # noqa: E402

from bharat.models.bharat_model import BharatForCausalLM  # noqa: E402
from bharat.models.config import BharatModelConfig  # noqa: E402
from bharat.tokenizer import BharatTokenizer  # noqa: E402
from bharat.tokenizer import load_tokenizer as load_bharat_tokenizer  # noqa: E402

# 22 Scheduled Indian Languages with sample starters
INDIC_LANGUAGE_PRESETS: dict[str, dict[str, str]] = {
    "hi": {
        "name": "Hindi (हिन्दी)",
        "starter": "भारत के प्राचीन इतिहास और संस्कृति के बारे में संक्षेप में बताइए।",
    },
    "bn": {"name": "Bengali (বাংলা)", "starter": "ভারতের সমৃদ্ধ সাহিত্য ও সংস্কৃতির ইতিহাস সম্পর্কে কিছু বলুন।"},
    "te": {"name": "Telugu (తెలుగు)", "starter": "భారతీయ సంస్కృతి మరియు చరిత్ర గురించి వివరించండి."},
    "ta": {
        "name": "Tamil (தமிழ்)",
        "starter": "இந்தியாவின் தொன்மையான பண்பாடு மற்றும் வரலாறு பற்றி விவரிக்கவும்.",
    },
    "mr": {
        "name": "Marathi (मराठी)",
        "starter": "भारताचा गौरवशाली इतिहास आणि संस्कृतीबद्दल माहिती द्या.",
    },
    "gu": {"name": "Gujarati (ગુજરાતી)", "starter": "ભારતના સમૃદ્ધ વારસા અને ઇતિહાસ વિશે ટૂંકમાં જણાવો."},
    "kn": {"name": "Kannada (ಕನ್ನಡ)", "starter": "ಭಾರತದ ಶ್ರೀಮಂತ ಇತಿಹಾಸ ಮತ್ತು ಸಂಸ್ಕೃತಿಯ ಬಗ್ಗೆ ತಿಳಿಸಿ."},
    "ml": {"name": "Malayalam (മലയാളം)", "starter": "ഭാരതീയ സംസ്കാരത്തെയും പാരമ്പര്യത്തെയും കുറിച്ച് പറയുക."},
    "pa": {"name": "Punjabi (ਪੰਜਾਬੀ)", "starter": "ਭਾਰਤ ਦੇ ਮਹਾਨ ਇਤਿਹਾਸ ਅਤੇ ਸੱਭਿਆਚਾਰ ਬਾਰੇ ਦੱਸੋ।"},
    "or": {"name": "Odia (ଓଡ଼ିଆ)", "starter": "ଭାରତର ପ୍ରାଚୀନ ଇତିହାସ ଏବଂ ସଂସ୍କୃତି ବିଷୟରେ କୁହନ୍ତୁ।"},
    "as": {"name": "Assamese (অসমীয়া)", "starter": "ভাৰতৰ চহকী ঐতিহ্য আৰু সংস্কৃতিৰ বিষয়ে চমুকৈ কওক।"},
    "ur": {"name": "Urdu (اردو)", "starter": "ہندوستان کی تاریخ اور تہذیب و ثقافت پر روشنی ڈالیں۔"},  # noqa: RUF001
    "sa": {"name": "Sanskrit (संस्कृतम्)", "starter": "भारतवर्षस्य समृद्धपरम्परायाः विषये संक्षेपेण वर्णयतु।"},
    "ne": {
        "name": "Nepali (नेपाली)",
        "starter": "नेपाल र भारतको ऐतिहासिक तथा सांस्कृतिक सम्बन्धबारे बताउनुहोस्।",
    },
    "sd": {"name": "Sindhi (سنڌي)", "starter": "سنڌي ٻولي ۽ ثقافت جي تاريخ بابت معلومات ڏيو."},
    "ks": {"name": "Kashmiri (کٲشُر)", "starter": "کٔشیٖر ہِنٛز ثقافت تہٕ تَوٲریٖخس مُتعلِق کینٛہہ ونِو."},
    "kok": {"name": "Konkani (कोंकणी)", "starter": "कोंकणी भास आनी संस्कृतायेविशीં म्हायती दियात."},
    "doi": {"name": "Dogri (डोगरी)", "starter": "डोगरी भाषा ते संस्कृति बारे संक्षेप च दसो."},
    "mai": {"name": "Maithili (मैथिली)", "starter": "मैथिली साहित्य आ संस्कृति केर इतिहास बताओ."},
    "sat": {
        "name": "Santali (ᱥᱟᱱᱛᱟᱲᱤ)",
        "starter": "ᱥᱟᱱᱛᱟᱲᱤ ᱯᱟᱹᱨᱥᱤ ᱟᱨ ᱞᱟᱠᱪᱟᱨ ᱨᱮᱱᱟᱜ ᱠᱟᱛᱷᱟ ᱞᱟᱹᱭ ᱢᱮ ᱾",
    },
    "brx": {"name": "Bodo (बड़ो)", "starter": "बर' हारि आरो राव-थुनलाइखौ मोजाङै फोरमायना हो।"},
    "mni": {"name": "Manipuri (ꯃৈতৈꯂꯣꯟ)", "starter": "ꯃꯅꯤꯄꯨꯔꯒꯤ ꯄꯨꯋꯥꯔꯤ ꯑꯃꯁꯨꯡ ꯂꯣꯟ-ꯂꯥꯏꯔꯤꯛ ꯃꯇꯥꯡꯗ ꯍꯥꯌꯕꯤꯌꯨ ꯫"},
}

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>🇮🇳 IndicLLM-Bharat Playground</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-primary: #0f172a;
      --bg-secondary: #1e293b;
      --bg-card: #334155;
      --accent: #f97316;
      --accent-hover: #ea580c;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --border: #475569;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
    body { background-color: var(--bg-primary); color: var(--text-main); height: 100vh; display: flex; flex-direction: column; }
    header { background-color: var(--bg-secondary); padding: 12px 24px; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid var(--border); }
    .logo { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 1.25rem; color: #fff; }
    .badge { background: linear-gradient(135deg, #f97316, #10b981); color: white; padding: 3px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; }
    .container { display: flex; flex: 1; overflow: hidden; }
    .sidebar { width: 340px; background-color: var(--bg-secondary); border-right: 1px solid var(--border); padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 18px; }
    .chat-area { flex: 1; display: flex; flex-direction: column; background: var(--bg-primary); }
    .messages { flex: 1; padding: 24px; overflow-y: auto; display: flex; flex-direction: column; gap: 16px; }
    .msg { max-width: 80%; padding: 12px 16px; border-radius: 12px; line-height: 1.5; font-size: 0.95rem; word-break: break-word; }
    .msg.user { align-self: flex-end; background-color: var(--accent); color: white; border-bottom-right-radius: 2px; }
    .msg.assistant { align-self: flex-start; background-color: var(--bg-card); color: var(--text-main); border-bottom-left-radius: 2px; border: 1px solid var(--border); }
    .input-box { padding: 16px 24px; background-color: var(--bg-secondary); border-top: 1px solid var(--border); display: flex; gap: 12px; }
    textarea { flex: 1; background: var(--bg-card); border: 1px solid var(--border); color: white; padding: 10px 14px; border-radius: 8px; resize: none; height: 50px; outline: none; font-size: 0.95rem; }
    textarea:focus { border-color: var(--accent); }
    button { background: var(--accent); color: white; border: none; padding: 0 20px; border-radius: 8px; font-weight: 600; cursor: pointer; transition: 0.2s; }
    button:hover { background: var(--accent-hover); }
    label { font-size: 0.85rem; font-weight: 600; color: var(--text-muted); display: block; margin-bottom: 6px; }
    select, input[type="text"] { width: 100%; background: var(--bg-card); border: 1px solid var(--border); color: white; padding: 8px 10px; border-radius: 6px; outline: none; }
    .slider-group { display: flex; flex-direction: column; gap: 4px; }
    .slider-header { display: flex; justify-content: space-between; font-size: 0.8rem; color: var(--text-muted); }
    input[type="range"] { width: 100%; accent-color: var(--accent); }
    .telemetry { font-size: 0.75rem; color: var(--text-muted); margin-top: 4px; }
  </style>
</head>
<body>
  <header>
    <div class="logo">
      <span>🇮🇳</span>
      <span>IndicLLM-Bharat Playground</span>
      <span class="badge" id="model-badge">Bharat-350M</span>
    </div>
    <div style="font-size: 0.85rem; color: var(--text-muted);" id="device-info">Device: auto</div>
  </header>

  <div class="container">
    <div class="sidebar">
      <div>
        <label>🇮🇳 Indian Language Presets (22 Scheduled)</label>
        <select id="lang-select" onchange="loadLanguagePreset()">
          <option value="">Select Language Preset...</option>
        </select>
      </div>

      <div>
        <label>System Instructions</label>
        <textarea id="system-prompt" style="height: 60px;">You are Bharat AI, an intelligent, helpful, and respectful Indian multilingual assistant.</textarea>
      </div>

      <div class="slider-group">
        <div class="slider-header"><span>Temperature</span><span id="temp-val">0.7</span></div>
        <input type="range" id="temperature" min="0" max="2" step="0.05" value="0.7" oninput="updateVal('temp-val', this.value)">
      </div>

      <div class="slider-group">
        <div class="slider-header"><span>Top-P (Nucleus)</span><span id="top-p-val">0.9</span></div>
        <input type="range" id="top_p" min="0" max="1" step="0.05" value="0.9" oninput="updateVal('top-p-val', this.value)">
      </div>

      <div class="slider-group">
        <div class="slider-header"><span>Max Tokens</span><span id="max-tokens-val">256</span></div>
        <input type="range" id="max_tokens" min="16" max="1024" step="16" value="256" oninput="updateVal('max-tokens-val', this.value)">
      </div>

      <div class="slider-group">
        <div class="slider-header"><span>Repetition Penalty</span><span id="rep-val">1.1</span></div>
        <input type="range" id="rep_penalty" min="1.0" max="2.0" step="0.05" value="1.1" oninput="updateVal('rep-val', this.value)">
      </div>

      <button onclick="clearChat()" style="background: var(--bg-card); border: 1px solid var(--border); padding: 8px;">Clear Chat</button>
    </div>

    <div class="chat-area">
      <div class="messages" id="messages">
        <div class="msg assistant">
          नमस्ते! मैं <strong>Bharat AI</strong> हूँ। भारतीय भाषाओं (हिन्दी, বাংলা, తెలుగు, தமிழ், मराठी, ಕನ್ನಡ, ইত্যাদি) में आप मुझसे कोई भी प्रश्न पूछ सकते हैं।
        </div>
      </div>
      <div class="input-box">
        <textarea id="user-input" placeholder="Type your prompt in any Indian language... (Press Enter to send)" onkeydown="handleKeyDown(event)"></textarea>
        <button id="send-btn" onclick="sendMessage()">Send</button>
      </div>
    </div>
  </div>

  <script>
    let languages = {};

    async function init() {
      try {
        const infoRes = await fetch('/api/info');
        const info = await infoRes.json();
        document.getElementById('model-badge').innerText = info.model_name || 'Bharat';
        document.getElementById('device-info').innerText = `Device: ${info.device} | Params: ${(info.parameters / 1e6).toFixed(1)}M`;

        const langRes = await fetch('/api/languages');
        languages = await langRes.json();
        const sel = document.getElementById('lang-select');
        for (const [code, item] of Object.entries(languages)) {
          const opt = document.createElement('option');
          opt.value = code;
          opt.innerText = item.name;
          sel.appendChild(opt);
        }
      } catch (e) {
        console.error('Failed to initialize playground:', e);
      }
    }

    function updateVal(id, val) { document.getElementById(id).innerText = val; }

    function loadLanguagePreset() {
      const code = document.getElementById('lang-select').value;
      if (code && languages[code]) {
        document.getElementById('user-input').value = languages[code].starter;
      }
    }

    function clearChat() {
      document.getElementById('messages').innerHTML = '<div class="msg assistant">Chat cleared. Ready for your prompt!</div>';
    }

    function handleKeyDown(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    }

    async function sendMessage() {
      const input = document.getElementById('user-input');
      const text = input.value.trim();
      if (!text) return;

      input.value = '';
      const msgBox = document.getElementById('messages');

      // Append User message
      const userDiv = document.createElement('div');
      userDiv.className = 'msg user';
      userDiv.innerText = text;
      msgBox.appendChild(userDiv);

      // Append Assistant placeholder
      const asstDiv = document.createElement('div');
      asstDiv.className = 'msg assistant';
      asstDiv.innerHTML = '<span class="cursor">▍</span>';
      msgBox.appendChild(asstDiv);
      msgBox.scrollTop = msgBox.scrollHeight;

      const payload = {
        prompt: text,
        system_prompt: document.getElementById('system-prompt').value,
        temperature: parseFloat(document.getElementById('temperature').value),
        top_p: parseFloat(document.getElementById('top_p').value),
        max_tokens: parseInt(document.getElementById('max_tokens').value),
        repetition_penalty: parseFloat(document.getElementById('rep_penalty').value)
      };

      try {
        const response = await fetch('/api/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullText = '';
        let startTime = performance.now();
        let tokenCount = 0;

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value);
          const lines = chunk.split('\\n');
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = JSON.parse(line.slice(6));
              if (data.token) {
                fullText += data.token;
                tokenCount++;
                asstDiv.innerText = fullText;
                msgBox.scrollTop = msgBox.scrollHeight;
              }
              if (data.done) {
                const dur = ((performance.now() - startTime) / 1000).toFixed(2);
                const speed = (tokenCount / Math.max(dur, 0.01)).toFixed(1);
                const tel = document.createElement('div');
                tel.className = 'telemetry';
                tel.innerText = `⚡ ${tokenCount} tokens in ${dur}s (${speed} tok/s)`;
                asstDiv.appendChild(tel);
              }
            }
          }
        }
      } catch (err) {
        asstDiv.innerText = `Error: ${err.message}`;
      }
    }

    window.onload = init;
  </script>
</body>
</html>"""


class GenerateRequest(BaseModel):
    prompt: str
    system_prompt: str = "You are Bharat AI, a helpful Indian multilingual assistant."
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1, le=500)
    max_tokens: int = Field(default=256, ge=1, le=2048)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)


def get_default_tokenizer() -> BharatTokenizer:
    """Retrieve default tokenizer or fallback to BPE adapter."""
    try:
        return load_bharat_tokenizer("gpt2")
    except Exception:
        from bharat.tokenizer.bpe import BPETokenizer
        from bharat.tokenizer.bpe_adapter import BharatBPETokenizer

        return BharatBPETokenizer(BPETokenizer())


def create_playground_app(
    model: BharatForCausalLM,
    config: BharatModelConfig,
    tokenizer: BharatTokenizer | None = None,
    device: torch.device | None = None,
    model_name: str = "Bharat-350M",
) -> FastAPI:
    """Build and configure the FastAPI playground application."""
    app = FastAPI(title="IndicLLM-Bharat Playground", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    dev = device or torch.device("cpu")
    tok = tokenizer or get_default_tokenizer()
    num_params = sum(p.numel() for p in model.parameters())

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(content=HTML_TEMPLATE)

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "model": model_name, "device": str(dev)}

    @app.get("/api/info")
    async def info() -> dict[str, Any]:
        return {
            "model_name": model_name,
            "device": str(dev),
            "parameters": num_params,
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
        }

    @app.get("/api/languages")
    async def languages() -> dict[str, dict[str, str]]:
        return INDIC_LANGUAGE_PRESETS

    @app.post("/api/generate")
    async def generate_stream(req: GenerateRequest) -> StreamingResponse:
        async def event_generator() -> AsyncGenerator[str, None]:
            full_prompt = (
                f"<|im_start|>system\n{req.system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{req.prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            raw_input_ids = tok.encode(full_prompt)
            input_ids = [t % config.vocab_size for t in raw_input_ids]
            tensor_ids = torch.tensor([input_ids], dtype=torch.long, device=dev)

            generated_tokens = 0
            curr_ids = tensor_ids

            with torch.no_grad():
                for _ in range(req.max_tokens):
                    out = model(curr_ids)
                    logits = out.logits[:, -1, :]

                    if req.temperature > 0:
                        probs = torch.softmax(logits / req.temperature, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(logits, dim=-1, keepdim=True)

                    token_id = next_token.item()
                    generated_tokens += 1

                    if token_id in (tok.eos_token_id, tok.pad_token_id):
                        break

                    token_text = tok.decode([token_id])
                    curr_ids = torch.cat([curr_ids, next_token], dim=1)

                    yield f"data: {json.dumps({'token': token_text, 'done': False})}\n\n"
                    await asyncio.sleep(0.005)

            yield f"data: {json.dumps({'token': '', 'done': True, 'count': generated_tokens})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return app


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch IndicLLM-Bharat interactive web playground",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--checkpoint",
        type=str,
        help="Path to trained PyTorch checkpoint (.pt)",
    )
    group.add_argument(
        "--model-config",
        type=str,
        help="Path to YAML model config",
    )
    group.add_argument(
        "--model-size",
        choices=["350m", "1b", "3b", "7b", "tiny"],
        default="350m",
        help="Standard model size tier",
    )

    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto", help="Compute device"
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)

    if parsed.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(parsed.device)

    # Initialize model
    if parsed.checkpoint:
        ckpt_p = Path(parsed.checkpoint)
        if not ckpt_p.is_file():
            print(f"Error: Checkpoint not found: {ckpt_p}", file=sys.stderr)
            return 1
        ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
        if "metadata" in ckpt and hasattr(ckpt["metadata"], "model_config"):
            cfg = BharatModelConfig.from_dict(ckpt["metadata"].model_config)
        elif "model_config" in ckpt:
            cfg = BharatModelConfig.from_dict(ckpt["model_config"])
        else:
            cfg = BharatModelConfig()
        model = BharatForCausalLM(cfg).to(device)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        model_name = f"Bharat-{ckpt_p.stem}"
    elif parsed.model_config:
        cfg_p = Path(parsed.model_config)
        if not cfg_p.is_file():
            print(f"Error: Model config not found: {cfg_p}", file=sys.stderr)
            return 1
        cfg = BharatModelConfig.from_yaml(cfg_p)
        model = BharatForCausalLM(cfg).to(device)
        model_name = f"Bharat-{cfg_p.stem}"
    else:
        tier = parsed.model_size
        if tier == "tiny":
            cfg = BharatModelConfig(
                vocab_size=1000,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=128,
            )
            model_name = "Bharat-Tiny"
        else:
            yaml_p = ROOT_DIR / "configs" / "models" / f"bharat-{tier}.yaml"
            cfg = BharatModelConfig.from_yaml(yaml_p) if yaml_p.is_file() else BharatModelConfig()
            model_name = f"Bharat-{tier.upper()}"
        model = BharatForCausalLM(cfg).to(device)

    tokenizer = get_default_tokenizer()
    app = create_playground_app(
        model=model,
        config=cfg,
        tokenizer=tokenizer,
        device=device,
        model_name=model_name,
    )

    print(f"\n  🚀 Starting IndicLLM-Bharat Playground on http://{parsed.host}:{parsed.port}")
    uvicorn.run(app, host=parsed.host, port=parsed.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
