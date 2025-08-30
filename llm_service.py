# llm_service.py
from __future__ import annotations
from typing import Optional
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
)
from langchain_huggingface import HuggingFacePipeline

# DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

class LLMService:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        # device: int = 0,                 # -1 for CPU; 0 for first CUDA device
        max_new_tokens: int = 1024,
        do_sample: bool = False,         # deterministic
        top_p: float = 1.0,              # ignored when do_sample=False
        use_4bit: bool = True,           # quantize to fit on 8GB VRAM
    ):
        self.model_name = model_name

        # --- Tokenizer ---
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

        # --- Quantization config (4-bit NF4) ---
        quant_cfg = None
        try:
            if use_4bit:
                quant_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
        except Exception as e:
            print(f"No quantization config found for {model_name}")

        # --- Load model ---
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_cfg,
            device_map="auto",   # auto offload if needed
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16 if not use_4bit else None,
        )

        # --- Text-generation pipeline (deterministic) ---
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            batch_size=1,
            top_p=top_p,
            return_full_text=False,
        )
        # DO NOT pass temperature when do_sample=False (it gets ignored)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tok,
            device_map="auto",
            **gen_kwargs
        )

        self.pipe = pipe
        self.llm = HuggingFacePipeline(pipeline=self.pipe)
