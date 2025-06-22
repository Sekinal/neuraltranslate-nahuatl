from unsloth import FastModel
model, tokenizer = FastModel.from_pretrained(
    model_name = "trainer_output/checkpoint-1932",
    max_seq_length = 256,
    load_in_4bit = False,
)

# Save the final, best-performing model
print("\nSaving and uploading model to Hugging Face Hub...")
model.push_to_hub(
    "Thermostatic/neuraltranslate-27b-mt-es-nah-v1", tokenizer,
    # Add your Hugging Face username if needed, e.g., "my_username/my_model_name"
)

print("\nScript finished successfully!")
