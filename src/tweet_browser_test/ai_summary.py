import re
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
# from transformers import TextStreamer
def construct_corpus(corpus):
    tweets = ''
    i=1
    for text in corpus:
        tweets += str(i) + "-[" + text + '] '
        i+=1
    # llama expects a prompt with the following format:
    prompt = f"""<s>[INST] <<SYS>>\nI would like you to help me by summarizing a group of tweets, delimited by triple backticks, and each tweet is labeled by a number in a given format: number-[tweet]. Give me a comprehensive summary in a concise paragraph and as you generate each sentence, provide the identifying number of tweets on which that sentence is based:\n<</SYS>>\n
                      ``` {tweets} ``` \n [/INST] \n"""
    return prompt


def llm_summarize(corpus: List(str)) -> str:
    """
    Summarize a corpus of text using the llama Large Language Model.
    """
    # Load the model info from the huggingface model hub
    peft_model_id = "TANUKImao/LLaMa2TS-0.1.0"
    config = PeftConfig.from_pretrained(peft_model_id)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, 
                                                 return_dict=True, 
                                                 load_in_8bit=True, 
                                                 device_map='auto',
                                                 #For Erik: you might need a user_auth_token to load the model
                                                 # use_auth_token=use_auth_token,
                                                 )
    # Load the Lora model
    model = PeftModel.from_pretrained(model, peft_model_id)
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, use_fast=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id # set the pad token to be the same as the eos token for open-ended generation

    # Construct the corpus
    prompt = construct_corpus(corpus)
    # Depends on whether we would like to show the process of generating the summary or not.
    # streamer = TextStreamer(tokenizer,  skip_prompt=True, skip_special_tokens=True)
    inputs = tokenizer(messages, return_tensors="pt").to('cuda')
    sequences = model.generate(
        **inputs,
        # streamer=streamer,
        max_new_tokens = 800,
        do_sample=True,
        top_k = 50,
        pad_token_id=tokenizer.eos_token_id
    )
    output = re.search(r'\n \[/INST\] \n(.*?)\n </s>', tokenizer.decode(sequences[0]), re.DOTALL)

    return output.group(1)

            
