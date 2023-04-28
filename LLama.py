import transformers
from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer
import torch
from datasets import load_dataset

dataset = load_dataset("cnn_dailymail", '3.0.0')


# print(dataset['train'])#features ['article', 'highlights' 'id]
# print(dataset['train'][0])#데이터 접근하는 법: dataset['train'][index]
# print(dataset['train']['article'][0])

# prompt = "Summarize sentence: 'Meta just suffered a major Facebook ad glitch that has advertisers asking about refunds'"
# context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."
#
#
# print(tokenizer.decode(outputs[0]))

def generateOutputs(dataset):
    model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
    tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
    outputs = []
    for i in range(len(dataset)):
        context = dataset[i]
        inputs = tokenizer.encode(context, return_tensors="pt")
        output = model.generate(inputs, max_length=60)
        response = tokenizer.decode(output[0])
        outputs.append(response)
    return outputs
a = ["Summarize sentence: 'Meta just suffered a major Facebook ad glitch that has advertisers asking about refunds'"]
print(generateOutputs(a)[0])
#