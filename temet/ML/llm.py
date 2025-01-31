"""
Development/training and deployment related to LLMs.
"""
import requests
import json
import requests

# sgpu
# vllm
# vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-8B --tensor-parallel-size 1 --max-model-len 32768 --enforce-eager

host = "verag001.bc.rzg.mpg.de"
port = 8000
model = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

def get_llm_response(question: str, model: str):
    """
    Sends a request to the model server and fetches a response.
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
    url = f"http://{host}:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
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

def ask(stream=True):
    """ Query a LLM via an OpenAI compatible API. """
    prompt = "Can JWST galaxies quench due to internal processes?"

    if stream:
        for line in stream_llm_response(prompt, model):
            print(line, end='')
    else:
        result = get_llm_response(prompt, model)
        content = result['choices'][0]['message']['content']
        #print(json.dumps(result, indent=2))
        print(content)
