#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, os, warnings, socket, re
from pathlib import Path
from typing import List, Iterable
from collections import Counter

import torch
from datasets import Dataset
from lxml import etree
import zss
import deepspeed
import sacrebleu

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
    EarlyStoppingCallback,
)
from trl import GRPOConfig, GRPOTrainer
from transformers.trainer_utils import IntervalStrategy
STRUCAUC_AVAILABLE = False

class MemoryUsageLogger(TrainerCallback):
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3 
            reserved = torch.cuda.memory_reserved() / 1024**3   
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            print(f"\n[Memory Usage] Step: {state.global_step} | "
                  f"Allocated: {allocated:.2f} GB | "
                  f"Reserved: {reserved:.2f} GB | "
                  f"Peak Allocated: {max_allocated:.2f} GB")

# Data loading utils 
def load_jsonl_for_rl(path: str | Path) -> Dataset:
    """Convert JSONL doc-pairs into GRPO-ready prompts."""
    items: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                if isinstance(obj, list) and len(obj) == 2:
                    src, tgt = obj
                elif isinstance(obj, list) and len(obj) == 1:
                    src, tgt = "", obj[0]
                elif isinstance(obj, dict):
                    src, tgt = obj.get("source", ""), obj.get("target", "")
                else:
                    continue
                
                src = src.strip().replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"') if src else ""
                tgt = tgt.strip().replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"') if tgt else ""
                
                if src and tgt:
                    items.append(
                        {
                            "prompt":    [{"role": "user", "content": src}],
                            "reference": tgt,
                        }
                    )
            except Exception as e:  # noqa
                warnings.warn(f"Skipping malformed line: {e}")
    return Dataset.from_list(items)

# XML/Structure helpers 
# Regex for matching XML tags (excluding <!...> sections)
TAG_RE = re.compile(r"<(?!\!)[^>]+>")

def strip_tags(s: str) -> str:
    """Remove XML/HTML tags, leaving only text."""
    return TAG_RE.sub("", s)

def _parse_fragment(fragment: str):
    """Parse XML fragment wrapped in dummy root with error recovery."""
    try:
        return etree.fromstring(f"<root>{fragment}</root>".encode(), parser=etree.XMLParser(recover=True))
    except Exception:  # noqa
        return None

def _convert_xml_wrapped(text: str):
    """Try parsing the string as an XML fragment with robust error handling."""
    try:
        # Escape any <!some text> sections to avoid parsing errors
        text = re.sub(r"<!.*?>", lambda m: m.group(0).replace(">", "&gt;").replace("<", "&lt;"), text)
        # Parse with recovery mode
        return etree.fromstring(text.encode(), parser=etree.XMLParser(recover=True))
    except Exception:
        return None

def _xml_shapes_equal(a: etree._Element | None, b: etree._Element | None) -> bool:
    """Safely compare two XML trees (None tolerated)."""
    if a is None or b is None:
        return False
    if a.tag != b.tag or len(a) != len(b):
        return False
    return all(_xml_shapes_equal(x, y) for x, y in zip(a, b))

def _collect_tags(elem: etree._Element) -> List[str]:
    """All tag names in document order (excluding root)."""
    return [e.tag for e in elem.iter() if e is not elem]

def tag_f1(pred_elem: etree._Element, ref_elem: etree._Element) -> float:
    """Calculate tag-level F1 score with proper handling of empty cases"""
    pred_tags = _collect_tags(pred_elem)
    ref_tags = _collect_tags(ref_elem)
    
    # If both have no tags, perfect match
    if not pred_tags and not ref_tags:
        return 1.0
    
    # If one has no tags, no match possible
    if not pred_tags or not ref_tags:
        return 0.0
    
    pc, rc = Counter(pred_tags), Counter(ref_tags)
    common = sum((pc & rc).values())
    
    # If no common tags, F1 is 0
    if not common:
        return 0.0
    
    p = common / sum(pc.values())
    r = common / sum(rc.values())
    return 2 * p * r / (p + r)

# TED-similarity utils 
class _Node:
    def __init__(self, elem: etree._Element):
        self.label = elem.tag
        self.children = [_Node(c) for c in elem] 

def _ted_similarity(a: etree._Element, b: etree._Element) -> float:
    """Tree-edit-distance similarity in [0,1] - fixed version that excludes dummy root"""
    try:
        # Calculate content size excluding the dummy root node
        a_size = len(list(a.iter())) - 1  # subtract root
        b_size = len(list(b.iter())) - 1    # subtract root
        
        # If both trees are empty (only root), return 1.0
        if a_size == 0 and b_size == 0:
            return 1.0
        
        # If one tree is empty, similarity should be 0.0
        if a_size == 0 or b_size == 0:
            return 0.0
        
        # Calculate tree edit distance
        dist = zss.simple_distance(_Node(a), _Node(b))
        
        # If no edit distance, trees are identical
        if dist == 0:
            return 1.0
        
        # Use the larger tree size for normalization
        max_size = max(a_size, b_size)
        
        # Calculate similarity ensuring it stays in [0,1] range
        similarity = 1.0 - (dist / max_size)
        return max(0.0, min(1.0, similarity))
        
    except Exception:
        return 0.0

# BLEU helpers
TOKENIZER_MAP = {
    "ja": "ja-mecab",
    "ko": "ko-mecab",
    "zh": "zh",
    "zf": "zh",
}

def bleu_tokenizer_for(lang: str) -> str:
    return TOKENIZER_MAP.get(lang, sacrebleu.BLEU.TOKENIZER_DEFAULT)

def salesforce_xml_bleu(preds, refs, tokenizer):
    """Calculate XML-BLEU score using Salesforce's approach."""
    pred_corpus = []
    ref_corpus = []
    for pred, ref in zip(preds, refs):
        pred_xml = _convert_xml_wrapped(f"<root>{pred}</root>")
        ref_xml  = _convert_xml_wrapped(f"<root>{ref}</root>")
        valid = pred_xml is not None
        match = valid and ref_xml is not None and _xml_shapes_equal(pred_xml, ref_xml)        

        # use TAG_RE to split pred and ref on xml elements into text chunks        
        pred_texts = [strip_tags(t) for t in TAG_RE.split(pred)]
        ref_texts  = [strip_tags(t) for t in TAG_RE.split(ref)]
        if match and len(pred_texts) == len(ref_texts):
            #assert len(pred_texts) == len(ref_texts), f"Mismatch in text chunks: {len(pred_texts)} vs {len(ref_texts)}"
            ref_corpus.extend(ref_texts)
            pred_corpus.extend(pred_texts)
        else:
            ref_corpus.extend(ref_texts)
            pred_corpus.extend([""] * len(ref_texts))  # empty pred for unmatched refs
    # Calculate Corpus BLEU
    bleu = sacrebleu.corpus_bleu(pred_corpus, [ref_corpus], tokenize=tokenizer)
    return bleu

# Reward Functions
def ted_reward_func(prompts, completions, reference, **kwargs):
    """TED similarity reward (0-10)"""
    preds = [c[0]["content"] for c in completions]
    scores = []
    for pred, ref in zip(preds, reference):
        p_xml, r_xml = _parse_fragment(pred), _parse_fragment(ref)
        if p_xml is None or r_xml is None:
            scores.append(-1.0)
        else:
            scores.append(10.0 * _ted_similarity(p_xml, r_xml))
    return scores

def xml_valid_reward_func(prompts, completions, reference, **kwargs):
    """Reward based on XML validity (0 or 10) - using robust parsing"""
    preds = [c[0]["content"] for c in completions]
    scores = []
    for pred in preds:
        pred_xml = _parse_fragment(pred)
        if pred_xml is not None:
            scores.append(10.0)
        else:
            scores.append(0.0)
    return scores

def xml_match_reward_func(prompts, completions, reference, **kwargs):
    """Reward based on XML structure matching (0 or 10) - using robust parsing"""
    preds = [c[0]["content"] for c in completions]
    scores = []
    for pred, ref in zip(preds, reference):
        pred_xml = _parse_fragment(pred)
        ref_xml = _parse_fragment(ref)
        if pred_xml is not None and ref_xml is not None and _xml_shapes_equal(pred_xml, ref_xml):
            scores.append(10.0)
        else:
            scores.append(0.0)
    return scores

def tag_f1_reward_func(prompts, completions, reference, **kwargs):
    """Reward based on tag-level F1 score (0-10)"""
    preds = [c[0]["content"] for c in completions]
    scores = []
    for pred, ref in zip(preds, reference):
        try:
            pred_elem = _parse_fragment(pred)
            ref_elem = _parse_fragment(ref)
            if pred_elem is not None and ref_elem is not None:
                f1 = tag_f1(pred_elem, ref_elem)
                scores.append(10.0 * f1)
            else:
                scores.append(0.0)
        except Exception:
            scores.append(0.0)
    return scores

def bleu_reward_func(prompts, completions, reference, **kwargs):
    """Reward based on sentence-level BLEU score (0-10)"""
    preds = [c[0]["content"] for c in completions]
    scores = []
    tokenizer = kwargs.get('tokenizer', 'zh')  # default tokenizer
    
    for pred, ref in zip(preds, reference):
        # Remove XML tags for plain BLEU
        pred_plain = strip_tags(pred)
        ref_plain = strip_tags(ref)
        
        # Sentence-level BLEU
        bleu = sacrebleu.sentence_bleu(pred_plain, [ref_plain], tokenize=tokenizer)
        scores.append(bleu.score / 10.0)  # BLEU is 0-100, normalize to 0-10
    return scores

def xml_bleu_reward_func(prompts, completions, reference, **kwargs):
    """Reward based on sentence-level XML-BLEU score (0-10)"""
    preds = [c[0]["content"] for c in completions]
    scores = []
    tokenizer = kwargs.get('tokenizer', 'zh')
    
    for pred, ref in zip(preds, reference):
        pred_xml = _convert_xml_wrapped(f"<root>{pred}</root>")
        ref_xml = _convert_xml_wrapped(f"<root>{ref}</root>")
        
        valid = pred_xml is not None
        match = valid and ref_xml is not None and _xml_shapes_equal(pred_xml, ref_xml)
        
        if match:
            # Split by tags and calculate BLEU on matching segments
            pred_texts = [strip_tags(t) for t in TAG_RE.split(pred)]
            ref_texts = [strip_tags(t) for t in TAG_RE.split(ref)]
            
            # Check if segments match in count
            if len(pred_texts) == len(ref_texts):
                # Sentence-level BLEU on concatenated segments
                pred_concat = " ".join(pred_texts)
                ref_concat = " ".join(ref_texts)
                bleu = sacrebleu.sentence_bleu(pred_concat, [ref_concat], tokenize=tokenizer)
                scores.append(bleu.score / 10.0)  # Normalize to 0-10
            else:
                scores.append(0.0)
        else:
            scores.append(0.0)
    return scores


# mapping of reward names to functions
reward_functions = {
    "ted": ted_reward_func,
    "xml_valid": xml_valid_reward_func,
    "xml_match": xml_match_reward_func,
    "tag_f1": tag_f1_reward_func,
    "bleu": bleu_reward_func,
    "xml_bleu": xml_bleu_reward_func,
}

# helpers
def free_port() -> int:
    """pick an available tcp port for torchrun master."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("GRPO DeepSpeed fine-tune with dev loss early stopping")
    # paths
    p.add_argument("--train_file", required=True)
    p.add_argument("--dev_file",   required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_dir", default="checkpoints/finetune_grpo_1")
    p.add_argument("--ds_config",  default="config/ds_grpo.json")
    # core hyper-params
    p.add_argument("--epochs", type=float, default=10)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--per_device_batch", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--num_generations", type=int, default=2)
    # seq lengths
    p.add_argument("--max_prompt_len",     type=int, default=1200)
    p.add_argument("--max_completion_len", type=int, default=1200)
    # generation parameters
    p.add_argument("--temperature", type=float, default=1.0, 
                   help="temperature for generation sampling (default: 1.0)")
    p.add_argument("--top_p", type=float, default=1.0,
                   help="top-p (nucleus sampling) probability threshold (default: 1.0)")
    p.add_argument("--top_k", type=int, default=None,
                   help="top-k sampling parameter (default: none)")
    p.add_argument("--repetition_penalty", type=float, default=1.0,
                   help="repetition penalty for generation (default: 1.0)")
    # logging / saving / eval
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps",    type=int, default=10)
    p.add_argument("--eval_steps",    type=int, default=10)
    p.add_argument("--early_stop",    type=int, default=10)
    p.add_argument("--bf16", action="store_true")
    # reward selection
    p.add_argument("--rewards", type=str, nargs='+', default=["ted"],
                   choices=list(reward_functions.keys()),
                   help="List of reward functions to use")
    # language for BLEU tokenizer
    p.add_argument("--lang", type=str, default="zh",
                   help="Language code for BLEU tokenization")
    return p

#@profile
def main():
    args = build_arg_parser().parse_args()

    eff_batch = (
        args.per_device_batch
        * args.grad_accum
        * (torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1)
    )
    if eff_batch % args.num_generations:
        raise ValueError(
            f"Effective batch {eff_batch} not divisible by num_generations={args.num_generations}"
        )

    train_ds = load_jsonl_for_rl(args.train_file)
    dev_ds   = load_jsonl_for_rl(args.dev_file)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )
    model.generation_config.temperature = args.temperature
    model.generation_config.top_p = args.top_p
    model.generation_config.top_k = args.top_k
    model.generation_config.repetition_penalty = args.repetition_penalty
    model.generation_config.do_sample = True

    cfg = GRPOConfig(
        output_dir=args.output_dir,
        #use_vllm=True,
        deepspeed=args.ds_config,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=args.per_device_batch,
        bf16=args.bf16,
        remove_unused_columns=False,

        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_len,
        max_completion_length=args.max_completion_len,

        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,

        logging_steps=args.logging_steps,
        eval_strategy="steps",  # Enable evaluation
        eval_steps=args.eval_steps,  # Set eval_steps in config
        save_steps=args.save_steps,
        save_total_limit=5,
    )

    # Print generation parameters and selected rewards
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(f"\n[Generation Parameters]")
            print(f"Temperature: {args.temperature}")
            print(f"Top-p: {args.top_p}")
            print(f"Top-k: {args.top_k}")
            print(f"Repetition penalty: {args.repetition_penalty}")
            print(f"\n[Selected Rewards]")
            print(f"Rewards: {args.rewards}\n")
    else:
        print(f"\n[Generation Parameters]")
        print(f"Temperature: {args.temperature}")
        print(f"Top-p: {args.top_p}")
        print(f"Top-k: {args.top_k}")
        print(f"Repetition penalty: {args.repetition_penalty}")
        print(f"\n[Selected Rewards]")
        print(f"Rewards: {args.rewards}\n")

    # Prepare reward functions
    selected_reward_funcs = []
    bleu_tokenizer = bleu_tokenizer_for(args.lang)
    
    for reward_name in args.rewards:
        reward_func = reward_functions[reward_name]
        # Create a wrapper that passes the tokenizer for BLEU-based rewards
        if reward_name in ["bleu", "xml_bleu"]:
            def make_reward_with_tokenizer(func, tok):
                def wrapped(prompts, completions, reference, **kwargs):
                    kwargs['tokenizer'] = tok
                    return func(prompts, completions, reference, **kwargs)
                return wrapped
            reward_func = make_reward_with_tokenizer(reward_func, bleu_tokenizer)
        selected_reward_funcs.append(reward_func)

    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        reward_funcs=selected_reward_funcs,
    )
    trainer.args.eval_strategy = IntervalStrategy.STEPS
    trainer.args.eval_steps = args.eval_steps
    trainer.args.metric_for_best_model = "eval_loss"
    trainer.args.greater_is_better = False          
    trainer.args.load_best_model_at_end = True 

    trainer.add_callback(
        EarlyStoppingCallback(
            early_stopping_patience=args.early_stop, 
            early_stopping_threshold=0.0
        )
    )
    trainer.add_callback(MemoryUsageLogger())

    trainer.train()

    if trainer.is_world_process_zero():
        trainer.save_model(Path(args.output_dir) / "final")

if __name__ == "__main__":
    main()
