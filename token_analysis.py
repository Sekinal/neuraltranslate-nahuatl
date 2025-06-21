import polars as pl
from transformers import AutoTokenizer
from datasets import load_dataset

# --- User's existing code ---
tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-4b-it")
full_dataset = load_dataset("Thermostatic/Axolotl-Spanish-Nahuatl-ShareGPT-Filtered", split="train")
df = full_dataset.to_polars()

transformed_df = df.select(
    # Create the 'nahuatl' column
    pl.col("conversations")
    .list.filter(pl.element().struct.field("from") == "human") # 1. Filter the list for the 'human' entry
    .list.first()                                              # 2. Get the first (and only) struct from the filtered list
    .struct.field("value")                                     # 3. Extract the 'value' field from that struct
    .alias("nahuatl"),                                         # 4. Rename the resulting column to 'nahuatl'

    # Create the 'spanish' column
    pl.col("conversations")
    .list.filter(pl.element().struct.field("from") == "gpt") # 1. Filter for the 'gpt' entry
    .list.first()                                            # 2. Get the struct
    .struct.field("value")                                   # 3. Extract its 'value'
    .alias("spanish"),                                       # 4. Rename to 'spanish'
)

print("Transformed DataFrame (with text):")
print(transformed_df.head())

# This DataFrame contains lists of token IDs
df2 = transformed_df.select(pl.all().map_elements(tokenizer.encode, return_dtype=pl.List(pl.Int64)))
print("\nTokenized DataFrame (df2):")
print(df2.head())
# --- End of user's existing code ---


# --- SOLUTION ---

# 1. Count the amount of tokens for each row
# We use .list.len() to get the length of each list in the columns of df2.
token_counts_df = df2.select(
    pl.col("nahuatl").list.len().alias("nahuatl_token_count"),
    pl.col("spanish").list.len().alias("spanish_token_count")
)

# 2. Print the count dataframe
print("\nToken Counts DataFrame:")
print(token_counts_df)

# 3. Get the summary statistics for these counts
# The .describe() method is perfect for this, calculating all common statistics.
summary_stats = token_counts_df.describe()

# 4. Print the summary statistics
print("\nSummary Statistics for Token Counts:")
print(summary_stats)