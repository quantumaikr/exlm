from ezlm import PlanktonForCausalLM
from transformers import AutoTokenizer, TextIteratorStreamer


base_model = 'quantumaikr/plankton-1B'
tokenizer = AutoTokenizer.from_pretrained('quantumaikr/plankton_tokenizer')
model = PlanktonForCausalLM.from_pretrained(base_model, cache_dir="hub", device_map="auto")


import gradio as gr
from threading import Thread


def random_response(message, history):
    message = f"<s> [INST] {message} [/INST]"
    input = tokenizer(message, return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        inputs=input['input_ids'],
        streamer=streamer,
        max_new_tokens=256,
        do_sample=True,
        top_p=0.9,
        temperature=float(0.9),
        top_k=30,
        repetition_penalty=1.2,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    model_output = ""
    for new_text in streamer:
        model_output += new_text
        yield model_output

    history.append(model_output)
    return model_output


demo = gr.ChatInterface(random_response)
demo.queue()
demo.launch(debug=True, share=True)