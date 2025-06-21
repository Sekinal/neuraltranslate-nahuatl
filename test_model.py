from unsloth import FastModel
model, tokenizer = FastModel.from_pretrained(
    model_name = "trainer_output/checkpoint-1932",
    max_seq_length = 256,
    load_in_4bit = True,
)

messages_1 = [{
    "role": "user",
    "content": [{"type" : "text", "text" : "Traduce al español: Auh in ye yuhqui in on tlenamacac niman ye ic teixpan on motlalia ce tlacatl itech mocaua.",}]
}]

messages_2 = [{
    "role": "user",
    "content": [{"type" : "text", "text" : "Traduce al español: ¿In chalchihuitl, teocuitlatl, mach ah ca on yaz?",}]
}]

messages_3 = [{
    "role": "user",
    "content": [{"type" : "text", "text" : "Traduce al español: Auh yn oyuh in yoca hualmotlalli tonatiuh ylhuicatitech, niman yc peuh yn huel ye tlacoçahuia, çan ihuiantzin ye tlayohuatiuh ynic ye poliuhtiuh tonatiuh, ynic huel ixpan || 147 ye yatiuh ynic huel ixpan ye onmomana metztli, huel cacitimoman ynic yahualtic tonatiuh ynic quixtzacuilli, y çan ihuiantzin huel onpolihuico tonatiuh.",}]
}]

messages_4 = [{
    "role": "user",
    "content": [{"type" : "text", "text" : "Auh in ye yuhqui in on tlenamacac niman ye ic teixpan on motlalia ce tlacatl itech mocaua.",}]
}]

messages_5 = [{
    "role": "user",
    "content": [{"type" : "text", "text" : "¿In chalchihuitl, teocuitlatl, mach ah ca on yaz?",}]
}]

messages_6 = [{
    "role": "user",
    "content": [{"type" : "text", "text" : "Auh yn oyuh in yoca hualmotlalli tonatiuh ylhuicatitech, niman yc peuh yn huel ye tlacoçahuia, çan ihuiantzin ye tlayohuatiuh ynic ye poliuhtiuh tonatiuh, ynic huel ixpan || 147 ye yatiuh ynic huel ixpan ye onmomana metztli, huel cacitimoman ynic yahualtic tonatiuh ynic quixtzacuilli, y çan ihuiantzin huel onpolihuico tonatiuh.",}]
}]

messages_list = [messages_1, messages_2, messages_3, messages_4, messages_5, messages_6]

from transformers import TextStreamer
for messages in messages_list:
    text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    )
    
    _ = model.generate(
        **tokenizer([text], return_tensors = "pt").to("cuda"),
        max_new_tokens = 64, # Increase for longer outputs!
        # Recommended Gemma-3 settings!
        temperature = 1.0, top_p = 0.95, top_k = 64,
        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )
