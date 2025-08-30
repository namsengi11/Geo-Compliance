# llm_service.py
from __future__ import annotations
from typing import Optional
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
)
from langchain_huggingface import HuggingFacePipeline

# DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3-3B-Instruct"

class LLMService:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_new_tokens: int = 512,
        do_sample: bool = False,     # deterministic
        top_p: float = 1.0,
        use_4bit: bool = True,
    ):
        self.model_name = model_name

        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

        quant_cfg = None
        try:
            if use_4bit:
                quant_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
        except Exception:
            pass

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_cfg,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16 if not use_4bit else None,
        )

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            batch_size=1,
            top_p=top_p,
            return_full_text=False,
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tok,
            device_map="auto",
            **gen_kwargs
        )

        self.pipe = pipe
        self.llm = HuggingFacePipeline(pipeline=self.pipe)

    # Small helper so main can classify regions uniformly (works for local)
    def generate_text(self, prompt: str) -> str:
        out = self.pipe(prompt)[0]["generated_text"]
        return out
