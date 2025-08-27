from __future__ import annotations
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline

class LLMService:
    def __init__(self,
        model_name: str = "google/gemma-3-1b-it", 
        task: Optional[str] = None,   # autodetect if None
        device: int = 0,             # -1=CPU, 0=first GPU
        max_new_tokens: int = 192,
        temperature: float = 0.2,
        top_p: float = 0.9):

        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        # decoder only models like GPT, Llama, etc.
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token # avoid warnings  
        mdl = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
        self.pipe = pipeline(
            "text-generation",
            model=mdl,
            tokenizer=tok,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            return_full_text=False
        )

        self.llm = HuggingFacePipeline(pipeline=self.pipe)