from unsloth import FastModel
import torch

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3-4b-it",
    max_seq_length = 2048, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # SHould leave on always!

    r = 8,           # Larger = higher accuracy, but might overfit
    lora_alpha = 8,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)

from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)

from datasets import load_dataset
# Load the full dataset
full_dataset = load_dataset("Thermostatic/Axolotl-Spanish-Nahuatl-ShareGPT-Filtered", split="train")

# First, split off the test set (10% of the total data)
train_val_split = full_dataset.train_test_split(test_size=0.1, seed=42)
test_dataset = train_val_split['test']

# Next, split the remaining 90% into training (80%) and validation (10%)
# To get 10% of the original dataset for validation from the remaining 90%,
# we need to take 1/9th of it (0.1 / 0.9 = 1/9)
final_split = train_val_split['train'].train_test_split(test_size=(1/9), seed=42)
train_dataset = final_split['train']
validation_dataset = final_split['test']

print(f"Dataset splits:")
print(f"  Train: {len(train_dataset)} examples")
print(f"  Validation: {len(validation_dataset)} examples")
print(f"  Test: {len(test_dataset)} examples")

### MODIFICATION ###
# 2. Standardize and format all three dataset splits
from unsloth.chat_templates import standardize_data_formats

train_dataset = standardize_data_formats(train_dataset)
validation_dataset = standardize_data_formats(validation_dataset)
test_dataset = standardize_data_formats(test_dataset)

def formatting_prompts_func(examples):
   convos = examples["conversations"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
   return { "text" : texts, }

# Apply formatting to all splits
train_dataset = train_dataset.map(formatting_prompts_func, batched = True)
validation_dataset = validation_dataset.map(formatting_prompts_func, batched = True)
test_dataset = test_dataset.map(formatting_prompts_func, batched = True)


### MODIFICATION ###
# 3. Configure the trainer to use the validation set and report metrics
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = validation_dataset, # Pass the validation set
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 256,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_ratio = 0.1,
        num_train_epochs = 2, # Set this for 1 full training run.
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "wandb", # Use this for WandB etc
        dataset_num_proc=4,  # Use more processes for mapping

        # New arguments for validation and saving
        eval_strategy="steps",
        eval_steps = 5,       # Evaluate at the end of each epoch
        save_strategy = "epoch",             # Save a checkpoint at the end of each epoch
    ),
)

from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)

# Start training. Validation loss will be calculated and reported to W&B automatically.
print("Starting training...")
trainer_stats = trainer.train()

### MODIFICATION ###
# 4. Evaluate the final model on the unseen test set
print("\nEvaluating the final model on the test set...")
test_results = trainer.evaluate(test_dataset)
print("Test set evaluation results:")
print(test_results)

# The test results are also automatically logged to WandB

# --- The rest of the script for inference and saving remains the same ---

print("\n--- Example Inference ---")
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)

messages = [{
    "role": "user",
    "content": [{"type" : "text", "text" : "Traduce al español: Auh in ye yuhqui in on tlenamacac niman ye ic teixpan on motlalia ce tlacatl itech mocaua.",}]
}]
text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
)

from transformers import TextStreamer
_ = model.generate(
    **tokenizer([text], return_tensors = "pt").to("cuda"),
    max_new_tokens = 64, # Increase for longer outputs!
    # Recommended Gemma-3 settings!
    temperature = 1.0, top_p = 0.95, top_k = 64,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)

# Save the final, best-performing model
print("\nSaving and uploading model to Hugging Face Hub...")
model.push_to_hub_merged(
    "Thermostatic/neuraltranslate-nahuatl-v0.0.1", tokenizer,
    # Add your Hugging Face username if needed, e.g., "my_username/my_model_name"
)

model.save_pretrained_gguf(
    "Thermostatic/neuraltranslate-nahuatl-v0.0.1-GGUF",
    quantization_type = "Q8_0",
)

print("\nScript finished successfully!")