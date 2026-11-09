"""QLoRA supervised fine-tuning on the PyVRP-labelled training set.

    python -m vrp_service.train.sft --data data/train/n100 \
      --model artifacts/llm/qwen2.5-7b-instruct --out artifacts/llm/adapters/sft-v1

Each row is the eval prompt (``render``) and the PyVRP routes as the compact JSON answer;
the loss covers the answer only. The base is loaded in 4-bit NF4 and a rank-64 LoRA adapter
is trained on every linear projection. The last ``--holdout`` feasible instances of each
variant are kept out of training and scored as eval loss.
"""

import argparse

from vrp_service.llm.examples import load_examples
from vrp_service.llm.prompt import answer, render

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def rows(data: str, holdout: int, limit: int | None = None) -> tuple[list[dict], list[dict]]:
    """Prompt/completion rows for training and for eval loss, every variant in each."""
    train, held_out = [], []
    for examples in load_examples(data).values():
        pairs = [
            {
                "prompt": render(request),
                "completion": [{"role": "assistant", "content": answer(routes)}],
            }
            for request, routes in examples
        ]
        train += pairs[: len(pairs) - holdout][:limit]
        held_out += pairs[len(pairs) - holdout :]
    return train, held_out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    # Prompt v2 is at most 4,377 tokens on the eval set; plus a ~350-token answer.
    parser.add_argument("--max-length", type=int, default=5120)
    parser.add_argument("--holdout", type=int, default=10)
    parser.add_argument("--limit", type=int, help="first N training rows per variant (smoke)")
    parser.add_argument("--no-quantize", action="store_true", help="CPU smoke runs only")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    train, held_out = rows(args.data, args.holdout, args.limit)
    quantization = None
    if not args.no_quantize:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quantization, dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    gpu = torch.cuda.is_available()
    config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        bf16=gpu,
        gradient_checkpointing=gpu,
        eval_strategy="steps",
        eval_steps=200,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=Dataset.from_list(train),
        eval_dataset=Dataset.from_list(held_out),
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=64,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=LORA_TARGETS,
            task_type="CAUSAL_LM",
        ),
    )
    trainer.train(resume_from_checkpoint=_last_checkpoint(args.out))
    trainer.save_model(args.out)
    print(f"adapter -> {args.out}; train rows {len(train)}, eval rows {len(held_out)}")


def _last_checkpoint(out: str) -> str | None:
    """Resume after a time limit: a resubmitted job continues from the newest checkpoint."""
    from pathlib import Path

    from transformers.trainer_utils import get_last_checkpoint

    return get_last_checkpoint(out) if Path(out).is_dir() else None


if __name__ == "__main__":
    main()
