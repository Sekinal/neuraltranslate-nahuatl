from unsloth import FastModel
model, tokenizer = FastModel.from_pretrained(
    model_name = "Thermostatic/neuraltranslate-27b-mt-es-nah-v1.1",
    max_seq_length = 256,
    load_in_4bit = False,
)

messages_1 = [{
    "role": "user",
    "content": [{"type" : "text", "text" : "Pero on inomachtijcahuan xcajsicamatiyaj tlinon quimijliaya niman nomojtiayaj para quitlajtojlisquej ma quinmelajcaijli on tlen quimijlijticatca.",}]
}]

messages_2 = [{
    "role": "user",
    "content": [{"type" : "text", "text" : "On miyec tlajyohuilistli yejhuan huajlau ipan Cristo, no topan huajlau.No ijqui itechcopa Cristo ticseliaj se hueyi teyoltlalijli.",}]
}]

messages_3 = [{
    "role": "user",
    "content": [{"type" : "text", "text" : "AUH INIC TLENAMACOYA",}]
}]

messages_4 = [{
    "role": "user",
    "content": [{"type" : "text", "text" : "pampa ijcuac ajsis on hora ica nentlajtosquej, on Espíritu Santo yejhua mechijlis tlinon nenquijtosquej.",}]
}]

messages_5 = [{
    "role": "user",
    "content": [{"type" : "text", "text" : "Dios ica iteicnelilis onechcalaquij quen nitlayecanquetl tepanchijchiquetl.Onictlalij on cimiento campa notlalis se cajli, niman ocse tlacatl oquetzteu on tepantli ipan on cimiento.Pero cada se ma nota sa no yejhua quen ijqui cuajli quetztehuas on tepantli.",}]
}]

messages_6 = [{
    "role": "user",
    "content": [{"type" : "text", "text" : "Yn isquichti cuicoanoaya muchinti tlatlataque unca mique; yn techpeualti ye techmictia yey hora, yn ountemictique teutualco.",}]
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
