#!/usr/bin/env python3
"""Tek seferlik vLLM isi: OCR veya JSON analiz. Surec bitince GPU tamamen bosalir."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def log(msg: str) -> None:
    print(msg, flush=True)


def load_cfg() -> dict:
    path = os.environ.get("PIPELINE_CONFIG", "/content/pipeline_config.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_llm(model: str, cfg: dict, multimodal: bool, enable_lora: bool):
    from vllm import LLM

    kwargs = dict(
        model=model,
        dtype="bfloat16",
        trust_remote_code=True,
        max_num_seqs=1,
        enforce_eager=True,
        disable_log_stats=True,
        gpu_memory_utilization=float(cfg["gpu_memory_utilization"]),
        max_model_len=int(cfg["max_model_len_vl"] if multimodal else cfg["max_model_len_text"]),
    )
    if multimodal:
        kwargs["limit_mm_per_prompt"] = {"image": 1}
        kwargs["mm_processor_kwargs"] = {
            "min_pixels": int(cfg["min_pixels"]),
            "max_pixels": int(cfg["max_pixels"]),
        }
        kwargs["mm_processor_cache_gb"] = 0
        if enable_lora:
            kwargs["enable_lora"] = True
            kwargs["max_lora_rank"] = int(cfg["max_lora_rank"])
            kwargs["max_loras"] = 1

    for attempt in range(3):
        try:
            return LLM(**kwargs)
        except TypeError as e:
            log(f"LLM TypeError, kwargs sadeleştiriliyor: {e}")
            msg = str(e)
            if "mm_processor_cache_gb" in msg or "mm_processor_cache_gb" in kwargs:
                kwargs.pop("mm_processor_cache_gb", None)
            if "mm_processor_kwargs" in msg or "max_pixels" in msg:
                kwargs.pop("mm_processor_kwargs", None)
            if "enforce_eager" in msg:
                kwargs.pop("enforce_eager", None)
            if "disable_log_stats" in msg:
                kwargs.pop("disable_log_stats", None)
            if "max_loras" in msg:
                kwargs.pop("max_loras", None)
        except Exception:
            if multimodal and "mm_processor_kwargs" in kwargs:
                log("mm_processor_kwargs ile yükleme başarısız, kaldırılıp tekrar denenecek.")
                traceback.print_exc()
                kwargs.pop("mm_processor_kwargs", None)
                kwargs.pop("mm_processor_cache_gb", None)
                continue
            raise
    return LLM(**kwargs)


def generate_vl(llm, prompt_text, image, sampling, lora_request, ocr_prompt: str):
    payloads = [
        {"prompt": prompt_text, "multi_modal_data": {"image": image}},
        {"prompt": prompt_text, "multi_modal_data": {"image": [image]}},
    ]
    last_err = None
    for payload in payloads:
        try:
            gen_kwargs = {"sampling_params": sampling}
            if lora_request is not None:
                gen_kwargs["lora_request"] = lora_request
            outs = llm.generate([payload], **gen_kwargs)
            return outs[0].outputs[0].text
        except Exception as e:
            last_err = e
            log(f"generate payload denemesi başarısız: {type(e).__name__}: {e}")
    try:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_pil", "image_pil": image},
                {"type": "text", "text": ocr_prompt},
            ],
        }]
        chat_kwargs = {"sampling_params": sampling}
        if lora_request is not None:
            chat_kwargs["lora_request"] = lora_request
        outs = llm.chat(messages, **chat_kwargs)
        return outs[0].outputs[0].text
    except Exception as e:
        last_err = e
        log(f"chat() fallback başarısız: {type(e).__name__}: {e}")
    raise last_err


def cmd_ocr(args, cfg):
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import SamplingParams

    image = Image.open(args.image).convert("RGB")
    processor = AutoProcessor.from_pretrained(cfg["vl_base_dir"], trust_remote_code=True)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": cfg["ocr_prompt"]},
        ],
    }]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    lora_dir = cfg["vl_lora_dir"]
    use_lora = bool(lora_dir) and os.path.isfile(os.path.join(lora_dir, "adapter_config.json"))
    log(f"VL yükleniyor (lora={use_lora}) -> {cfg['vl_base_dir']}")
    llm = None
    lora_request = None
    try:
        llm = build_llm(cfg["vl_base_dir"], cfg, multimodal=True, enable_lora=use_lora)
        if use_lora:
            from vllm.lora.request import LoRARequest
            lora_request = LoRARequest("qwen_vl_lora_finetuned", 1, lora_dir)
        sampling = SamplingParams(temperature=0.0, max_tokens=int(cfg["max_new_tokens_ocr"]))
        log("OCR üretimi başlıyor...")
        try:
            text = generate_vl(llm, prompt_text, image, sampling, lora_request, cfg["ocr_prompt"])
        except Exception:
            if lora_request is not None:
                log("LoRA ile OCR başarısız, base VL ile tekrar deneniyor.")
                traceback.print_exc()
                text = generate_vl(llm, prompt_text, image, sampling, None, cfg["ocr_prompt"])
            else:
                raise
        text = (text or "").strip()
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"ok": True, "text": text}, f, ensure_ascii=False)
        log(f"OCR bitti, {len(text)} karakter yazıldı.")
    finally:
        del llm
        try:
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass


def cmd_analyze(args, cfg):
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    with open(args.text_file, encoding="utf-8") as f:
        document_text = f.read()
    with open(cfg["system_prompt_path"], encoding="utf-8") as f:
        system_prompt = f.read()

    tokenizer = AutoTokenizer.from_pretrained(cfg["analyzer_dir"], trust_remote_code=True)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": document_text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    log(f"Analiz modeli yükleniyor -> {cfg['analyzer_dir']}")
    llm = None
    try:
        llm = build_llm(cfg["analyzer_dir"], cfg, multimodal=False, enable_lora=False)
        sampling = SamplingParams(temperature=0.0, max_tokens=int(cfg["max_new_tokens_analysis"]))
        log("JSON analiz üretimi başlıyor...")
        out = llm.generate([prompt], sampling)[0].outputs[0].text
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"ok": True, "text": out}, f, ensure_ascii=False)
        log("Analiz bitti.")
    finally:
        del llm
        try:
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["ocr", "analyze"])
    parser.add_argument("--image")
    parser.add_argument("--text_file")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = load_cfg()
    try:
        if args.task == "ocr":
            if not args.image:
                raise SystemExit("--image gerekli")
            cmd_ocr(args, cfg)
        else:
            if not args.text_file:
                raise SystemExit("--text_file gerekli")
            cmd_analyze(args, cfg)
    except Exception:
        traceback.print_exc()
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"ok": False, "error": traceback.format_exc()}, f, ensure_ascii=False)
        sys.exit(1)


if __name__ == "__main__":
    main()
