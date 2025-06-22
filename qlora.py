from unsloth import FastModel
import torch

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    max_seq_length = 256, # Choose any for long context!
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

    r = 64,           # Larger = higher accuracy, but might overfit
    lora_alpha = 64,  # Recommended alpha == r at least
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
train_dataset = load_dataset("Thermostatic/Axolotl-Spanish-Nahuatl-ShareGPT-Filtered-Splits", split="train")
validation_dataset = load_dataset("Thermostatic/Axolotl-Spanish-Nahuatl-ShareGPT-Filtered-Splits", split="validation")

print(f"Dataset splits:")
print(f"  Train: {len(train_dataset)} examples")
print(f"  Validation: {len(validation_dataset)} examples")

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
train_dataset = train_dataset.map(
    formatting_prompts_func, 
    batched = True
    )
validation_dataset = validation_dataset.map(
    formatting_prompts_func, 
    batched = True
    )

from evaluate import load
import numpy as np # Make sure to import numpy

# Load the metric. chrF is a great choice here.
chrf_metric = load("chrf")

def preprocess_logits_for_metrics(logits, labels):
    """
    This function is called by the Trainer before computing metrics.
    It takes the raw logits and converts them to the predicted token IDs.
    This is memory-efficient as it avoids storing all logits.
    """
    # The logits are often a tuple, so we take the first element.
    if isinstance(logits, tuple):
        logits = logits[0]
        
    # Get the predicted token IDs by taking the argmax along the last dimension.
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids

def compute_metrics(eval_pred):
    pred_ids, label_ids = eval_pred

    # **THE FIX IS HERE**
    # Mask the predictions where the label is -100.
    # This prevents the tokenizer from trying to decode irrelevant tokens
    # predicted for the prompt or padding.
    pred_ids = np.where(label_ids != -100, pred_ids, tokenizer.pad_token_id)
    
    # Also clean up the labels as before.
    label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)

    # Decode predictions and labels.
    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # The `chrf` metric expects a list of predictions and a list of lists of references.
    decoded_labels_nested = [[label] for label in decoded_labels]

    # Compute the metric
    result = chrf_metric.compute(predictions=decoded_preds, references=decoded_labels_nested)

    return {"chrf": result["score"]}

# 3. Configure the trainer to use the validation set and report metrics
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = validation_dataset,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics = preprocess_logits_for_metrics,
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_ratio = 0.1,
        num_train_epochs = 10, # Set this for 1 full training run.
        learning_rate = 2e-5, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "wandb", # Use this for WandB etc
        dataset_num_proc=4,  # Use more processes for mapping
        prediction_loss_only=False,
        # New arguments for validation and saving
        eval_strategy="epoch",
        save_strategy = "epoch",             # Save a checkpoint at the end of each epoch
        load_best_model_at_end=True,     # 4. Enable loading the best model
        metric_for_best_model="chrf",    # 5. Tell it WHICH metric defines "best"
        greater_is_better=True,  
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
