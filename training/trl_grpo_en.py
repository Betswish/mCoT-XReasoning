import argparse
import logging
import re
from typing import Any, List

import torch
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer

from math_verify import LatexExtractionConfig, parse, verify


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT: str = (
    "Always think in English.\n\n"
    "Solve the following math problem step by step.\n"
    "For each step, write your reasoning in the format <think>your reasoning process here</think>.\n"
    "Finally, provide the final answer enclosed in \\boxed{}.\n\n"
    "Problem: {text}\n\n"
    "Your answer should be in the format: 'The answer is: \\boxed{final_answer}', with your reasoning inside <think></think> tags and the answer inside \\boxed{}."
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the script.

    Returns:
        argparse.Namespace: Parsed arguments with vllm_server_host.
    """
    parser = argparse.ArgumentParser(
        description="GRPO Training Script with vLLM server host option."
    )
    parser.add_argument(
        "--vllm_server_host",
        type=str,
        default="0.0.0.0",
        help="Hostname or IP address of the vLLM server (default: 0.0.0.0)."
    )
    # Add more arguments as needed.
    return parser.parse_args()


def make_conversation(example: dict) -> dict:
    """Format a training example into a conversation prompt.

    Args:
        example (dict): Example with a 'question' field.

    Returns:
        dict: Dictionary with a 'prompt' key for the conversation.
    """
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ],
        "solution": example["answer"],
    }


def make_conversation_test(example: dict) -> dict:
    """Format a test example into a conversation prompt.

    Args:
        example (dict): Example with a 'problem' field.

    Returns:
        dict: Dictionary with a 'prompt' key for the conversation.
    """
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }


def format_reward(completions: List[Any], **kwargs) -> List[float]:
    """Reward function that checks if the completion matches the required Japanese math prompt format.

    The expected format is:
        - At least one <think>...</think> block (for reasoning)
        - At least one '\\boxed{...}' (for the answer)

    Args:
        completions (List[Any]): List of completions, each a list of dicts with a "content" key.

    Returns:
        List[float]: 1.0 if the completion matches the required format, else 0.0.

    Example:
        >>> completions = [[{"content": "<think>推論</think>\n答えは：\\boxed{42}"}]]
        >>> format_reward(completions)
        [1.0]
    """
    think_pattern = re.compile(r"<think>.*?</think>", re.DOTALL)
    # Regex for '答えは：\boxed{...}'
    answer_pattern = re.compile(r"\\boxed\{([^}]+?)\}",)

    rewards: List[float] = []
    for completion in completions:
        if (
            not completion
            or not isinstance(completion[0], dict)
            or "content" not in completion[0]
        ):
            rewards.append(0.0)
            continue
        content = completion[0]["content"]
        # has_think = bool(think_pattern.search(content))
        has_think = "</think>" in content 
        has_boxed = "\\boxed{" in content
        rewards.append(0.5 if has_think and has_boxed else 0.0)
    return rewards


def accuracy_reward(completions: List[Any], **kwargs) -> List[float]:
    """Reward function that checks if the completion matches the ground truth solution.

    Args:
        completions (List[Any]): List of completions, each a list of dicts with a "content" key.
        **kwargs: Must contain 'solution' (List[str]).

    Returns:
        List[float]: 1.0 if the answer matches, else 0.0.
    """
    solutions = kwargs.get("solution", [])
    completion_contents = [
        completion[0]["content"] if completion and isinstance(completion[0], dict) and "content" in completion[0] else ""
        for completion in completions
    ]
    rewards: List[float] = []
    for content, solution in zip(completion_contents, solutions):
        gold_parsed = [solution]
        try:
            answer_parsed = parse(
                content,
                extraction_mode="any_match",
                extraction_config=[LatexExtractionConfig()],
            )
            rewards.append(float(verify(answer_parsed, gold_parsed)))
        except Exception as e:
            logger.warning(f"Verification failed: {e}")
            rewards.append(0.0)
    return rewards


def main() -> None:
    """Main function to run GRPO training."""
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model


    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )
    ###
        #     quantization_config=BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     llm_int8_threshold=6.0,
        #     llm_int8_has_fp16_weight=False,
        #     bnb_4bit_compute_dtype=torch.float16,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        # ),
    ###
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # Load datasets
    train_dataset: Dataset = load_dataset("GAIR/LIMO", split="train")
    test_dataset: Dataset = load_dataset(
        "shanchen/aime_2025_multilingual", "default", split="en"
    )

    # Format datasets
    train_dataset = train_dataset.map(make_conversation)
    test_dataset = test_dataset.map(make_conversation_test)

    # Retain only the 'prompt' and 'solution' columns
    train_dataset = train_dataset.select_columns(["prompt", "solution"])
    test_dataset = test_dataset.select_columns(["prompt", "solution"])

    logger.info(f"First training prompt: {train_dataset[0]['prompt']}")
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")

    # Training configuration
    training_args = GRPOConfig(
        output_dir="/temp_work/ch225816/trl_grpo/cache_qwen15b_dapo_0.28_ENGLISH_0.5_format_64lora",
        learning_rate=1e-5,
        remove_unused_columns=False,  # to access the solution column in accuracy_reward
        gradient_accumulation_steps=64,
        per_device_train_batch_size=1,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        num_train_epochs=10,
        bf16=True,
        epsilon_high=0.28,
        loss_type="grpo",
        max_completion_length=5000,
        num_generations=8,
        max_prompt_length=512,
        use_vllm=True,
        vllm_mode="server",
        report_to=["wandb"],
        logging_steps=5,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",
        eval_steps=10,
        vllm_server_timeout=360,
    )

    # Set vLLM server host from command-line argument
    setattr(training_args, "vllm_server_host", args.vllm_server_host)
    logger.info(f"Using vLLM server host: {training_args.vllm_server_host}")

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward, accuracy_reward],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Start training
    logger.info("Starting GRPO training...")
    trainer.train()

    # Save the model
    logger.info("Saving trained model...")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
