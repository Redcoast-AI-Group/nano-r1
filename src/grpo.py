from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from unsloth import is_bfloat16_supported
import torch
from trl import GRPOConfig, GRPOTrainer
from .prompt_utils import (
    SYSTEM_PROMPT,
    XML_COT_FORMAT
)
from .rewards import (
    accuracy_reward_func,
    format_reward_func,
    reasoning_steps_reward_func,
    get_cosine_scaled_reward_func
)
from .data_utils import get_gsm8k_questions, get_numinamath_tir_questions


def run_grpo(args):

    if "gsm8k" in args.dataset_name_or_path:
        dataset = get_gsm8k_questions(args.dataset_name_or_path)
    elif "NuminaMath-TIR" in args.dataset_name_or_path:
        dataset = get_numinamath_tir_questions(args.dataset_name_or_path)
    else:
        raise NotImplementedError("NotImplemented yet.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name_or_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        fast_inference=args.fast_inference,
        max_lora_rank=args.lora_rank,
        gpu_memory_utilization=args.gpu_memory_utilization
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=args.target_modules,
        lora_alpha=args.lora_rank,
        use_gradient_checkpointing = args.use_gradient_checkpointing,
        random_state=3407
    )

    training_args = GRPOConfig(
        use_vllm = args.use_vllm,
        learning_rate = args.learning_rate,
        adam_beta1 = args.adam_beta1,
        adam_beta2 = args.adam_beta2,
        weight_decay = args.weight_decay,
        warmup_ratio = args.warmup_ratio,
        lr_scheduler_type = args.lr_scheduler_type,
        optim = args.optim,
        log_level = args.log_level,
        logging_steps = args.logging_steps,
        bf16 = args.bf16,
        fp16 = not args.bf16,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        num_generations = args.num_generations,
        max_prompt_length = args.max_prompt_length,
        max_completion_length = args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        max_steps = args.max_steps,
        save_strategy = args.save_strategy,
        max_grad_norm = args.max_grad_norm,
        report_to = args.report_to,
        output_dir = args.output_dir,
        overwrite_output_dir = args.overwrite_output_dir
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class = tokenizer,
        reward_funcs = [
            accuracy_reward_func,
            format_reward_func,
            reasoning_steps_reward_func,
            get_cosine_scaled_reward_func
        ],
        args = training_args,
        train_dataset = dataset,
    )
    trainer.train()

    if args.lora_save_path:
        model.save_lora(args.lora_save_path)

