import transformers
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
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
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-1b1")
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-1b1")
    outputs = []
    for i in range(len(dataset)):
        context = dataset[i]
        inputs = tokenizer.encode(context, return_tensors="pt")
        output = model.generate(inputs, max_length=40)
        response = tokenizer.decode(output[0])
        outputs.append(response)
    return outputs
#

