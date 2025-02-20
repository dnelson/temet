"""
Development/training/fine-tuning and deployment related to LLMs.
"""
import requests
import json
import os

# ----------------------------- inference ----------------------------- 

# sgpu
# source activate /u/dnelson/.local/envs/vllm
# export PYTHONSTARTUP=
# vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-8B 
#   --tensor-parallel-size 1 --max-model-len 32768 --enforce-eager
#   --enable-lora
#   --lora-modules '{"name": "astrolora", "path": "/vera/ptmp/gc/dnelson/cache_hf/test/", "base_model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"}'
# docs: https://docs.vllm.ai/en/latest/features/lora.html

# astro LLMs:
#   cosmosage (https://arxiv.org/abs/2407.04420)
#   cosmogemma (https://github.com/sultan-hassan/CosmoGemma)
#   astrollama

# frontends:
#  open webui (complex)
#  chatbot ui
#  fastchat
#  -- needs to be ultra simple (no backend), so that I can integrate it into django
#  -- needs to stream from vllm (streamingresponse i.e. server side events) (or websockets?)
#  -- do not stream through django.
#  -- apache reverse proxy a subdirectory to the vllm server
#  -- let the javascript (htmx framework?!) in the client browser handle the streaming
#  -- vllm serve uses FastAPI, so we just need "fastapi stream javascript example"
#  -- https://blog.philip-huang.tech/?page=server-sent-events
#  -- https://nlpcloud.com/how-to-develop-a-token-streaming-ui-for-your-llm-with-go-fastapi-and-js.html
#  -- almost like https://github.com/codecaine-zz/llm_bootstrap5_template/blob/main/index.html
#  -- but need: markdown support, latex/mathjax support, code syntax highlighting
#  -- "bootstrap llm chat ui"
#  -- "lightweight llm chat ui markdown latex"
# https://github.com/victordonoso/chatgpt_clone
# https://github.com/abhishekkrthakur/aiaio
# and get nice html/css from one of https://github.com/billmei/every-chatgpt-gui?tab=readme-ov-file
# tlooto - nice UI and pre-login functionality ideas

host = "verag001.bc.rzg.mpg.de"
port = 8000

base_model = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
lora_model = "astrolora"

def get_llm_response(question: str, model: str):
    """
    Sends a request to the model API server and fetches a response.
    """
    url = f"http://{host}:{port}/v1/chat/completions"  # Adjust the URL if different
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()

def stream_llm_response(question:str, model:str):
    """
    Sends a request to the model API server for a streaming response.
    """
    url = f"http://{host}:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "max_tokens": 2000, # testing
        "temperature": 0.6, # testing
        "stream": True
    }

    with requests.post(url, headers=headers, json=data, stream=True) as response:
        for line in response.iter_lines():
            if line:
                # OpenAI-style streaming responses are prefixed with "data: "
                decoded_line = line.decode("utf-8").replace("data: ", "")
                if decoded_line == '[DONE]':
                    break
                response = json.loads(decoded_line)
                content = response['choices'][0]['delta']['content']
                yield content

def models():
    """ Get list of available models from API. """
    url = f"http://{host}:{port}/v1/models"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def ask(stream=True, finetuned=False):
    """ Query a LLM via an OpenAI compatible API. """
    prompt = "Can JWST galaxies quench due to internal processes?"
    #prompt = "There is a gopher in my garden, what should I do?"

    model = base_model if not finetuned else lora_model

    # https://unsloth.ai/blog/deepseek-r1
    # TESTING: (doesn't seem to substantially alter response)
    #prompt = "<｜User｜>" + prompt + "<｜Assistant｜>"

    if stream:
        for line in stream_llm_response(prompt, model):
            print(line, end='')
    else:
        result = get_llm_response(prompt, model)
        content = result['choices'][0]['message']['content']
        #print(json.dumps(result, indent=2))
        print(content)


# ----------------------------- fine-tuning ----------------------------- 

# sgpu
# conda activate /u/dnelson/.local/envs/peft
# export PYTHONSTARTUP=

# https://www.llama.com/docs/how-to-guides/fine-tuning/
# https://medium.com/@rafaelcostadealmeida159/how-to-fine-tune-deepseek-r1-using-lora-7033edf05ee0
# https://github.com/huggingface/open-r1?tab=readme-ov-file
# aratus: https://en.wikipedia.org/wiki/Aratus (or any astronomer of the ancient world)

# 1. arxiv latex -> markdown (pandoc, LatexML as used for arxiv HTML versions, or custom)
#    pandoc -s -t markdown -f latex main.tex
#    issues: figures/tables (skip), references (remove all?), bibliography (remove)
#    future: bib (https://github.com/sciunto-org/python-bibtexparser)
# 2. use raw markdown as next token predicion task?
#  -- 2b. mix in auto-generated Q&A structured input, generated by passing the same documents to e.g. Deepseek-R1, to retain instruct capabilities?
#  -- 2c. use chain-of-thought structured input, generated by passing same documents to e.g. Deepseek-R1, to retain CoT capabilities?
# 4. fine-tuning library for LORA: PEFT
# 5. goal: inference with vLLM without merging, i.e. keep the LORA matrices separate

def finetune():
    #import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from peft import LoraConfig, PeftModel, get_peft_model
    from trl import SFTTrainer, SFTConfig

    # login
    hf_token = os.getenv('HF_TOKEN') # set in .bashrc

    # config
    basepath = "/u/dnelson/data/cache_hf/hub/"
    #model_name = basepath + "models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/"
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    #model_name = "meta-llama/Llama-3.2-1B"

    lora_r = 2 # very low, 8 or 16 more reasonable
    lora_alpha = 16 # default: 32?
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    max_seq_length = 512 # depends on model and tokenizer

    seed = 42424242

    # load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                device_map="auto",
                                                local_files_only=True)
                                                #token=hf_token)
                                                #is_trainable=True (figure out how to load partially trained adapter and resume)

    tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                            padding=True, 
                                            truncation=True, 
                                            max_length=max_seq_length,
                                            local_files_only=True)
                                            #token=hf_token)

    # check
    #tokenizer.eos_token = '<|end_of_text|>'
    #tokenizer.encode('hello there 123!') = {'input_ids': [128000, 15339, 1070, 220, 4513, 0], 'attention_mask': [1, 1, 1, 1, 1, 1]}
    #tokenizer.all_special_tokens = ['<|begin_of_text|>', '<|end_of_text|>']
    #print(tokenizer.model_max_length) # i.e. max_seq_length

    #if tokenizer.pad_token is None: # not clear if it's a good idea
    #    tokenizer.pad_token = tokenizer.eos_token

    # could add quantization of the base model with bitsandbytes

    # set up the LoRA config
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=target_modules,
        init_lora_weights="gaussian",
        task_type="CAUSAL_LM",
        inference_mode=False
    )

    # wrap the model with LoRA and check the amount of trainable parameters
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    # prepare training data
    # https://huggingface.co/docs/transformers/preprocessing
    # https://github.com/huggingface/transformers/issues/11455

    training_data = ["Hello, how are you?",
                    "I am fine, thank you.",
                        "What are you doing?",
                        "I am writing some code.",
                        "What is your favorite programming language?",
                        "I like Python."]

    test_data = ["What is your favorite book?",
                "I like 'The Hitchhiker's Guide to the Galaxy'."]

    # padding warning: https://discuss.huggingface.co/t/padding-side-in-instruction-fine-tuning-using-sftt/113549
    training_data_tokenized = tokenizer(training_data, padding=True, padding_side="right", truncation=True, return_tensors="pt", max_length=max_seq_length)
    test_data_tokenized = tokenizer(test_data, padding=True, padding_side="right", truncation=True, return_tensors="pt", max_length=max_seq_length)

    # different try for dataset
    # https://huggingface.co/docs/datasets/use_with_pytorch
    from datasets import Dataset

    training_ds = Dataset.from_dict({"messages": training_data})
    training_ds = training_ds.with_format("torch")

    test_ds = Dataset.from_dict({"messages": test_data})
    test_ds = test_ds.with_format("torch")

    # can use SFTTrainer (supervised fine-tuning) (need to be in prompt style?)
    # or can just use torch .train(), is this always 'unspervised fine-tuning' i.e. next-token
    # I think these are the same, as they are both next-token prediction
    # just that "SFT" is usually phrased as "pre-training" with big data, and 
    # "USFT" is usually phrased as "fine-tuning" with small data
    num_train_epochs = 3
    max_steps = 100
    bfloat16 = False

    run_name = model_name.split('/')[-1] + f'-fine-tune-{num_train_epochs}-{max_steps}'

    output_dir = basepath + run_name + '/'

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # note: should pass a max_seq_length, defualt is min(tokenizer.model_max_length, 1024)
    training_args = SFTConfig(
        dataset_text_field="messages",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True, # Saves memory at the cost of additional training time.
        bf16=bfloat16,
        tf32=False, # use tf32 for faster training on Ampere GPUs or newer.
        dataloader_pin_memory=False, # pin data to memory.
        torch_compile=False, # compile to create graphs from existing PyTorch programs.
        warmup_steps=50,
        max_steps=max_steps,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_strategy="steps",
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        eval_strategy="steps",
        logging_steps=10,
        output_dir=output_dir,
        optim="paged_adamw_8bit",
        remove_unused_columns=True,
        seed=seed,
        run_name=run_name,
        report_to="none", #"wandb",
        push_to_hub=False,
        eval_steps=25,
    )

    trainer = SFTTrainer(
        model=model, # peft_model !?
        args=training_args,
        train_dataset=training_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,
        peft_config=lora_config
    )

    print("Training...")
    trainer.train()

    print("Saving...")
    save_dir = basepath + "../test/"

    if 1:
        peft_model.save_pretrained(save_dir) # save only adapter parameters

    if 0:
        # merge into base model, and save merged model
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(save_dir)

    #model = AutoModelForCausalLM.from_pretrained(save_dir) # reload for inference
