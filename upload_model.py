from unsloth import FastModel
model, tokenizer = FastModel.from_pretrained(
    model_name = "trainer_output/checkpoint-2415",
    max_seq_length = 256,
    load_in_4bit = True,
)

# Save the final, best-performing model
print("\nSaving and uploading model to Hugging Face Hub...")
model.push_to_hub_merged(
    "Thermostatic/neuraltranslate-nahuatl-v0.0.1-12b", tokenizer,
    # Add your Hugging Face username if needed, e.g., "my_username/my_model_name"
)

print("\nScript finished successfully!")
