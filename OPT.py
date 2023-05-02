import transformers
from transformers import OPTForCausalLM
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
    model = OPTForCausalLM.from_pretrained("facebook/opt-6.7b")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b")
    outputs = []
    for i in range(len(dataset)):
        context = dataset[i]
        inputs = tokenizer(context, return_tensors="pt")
        output = model.generate(inputs.input_ids, max_length=50, num_return_sequences=1)
        response = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        outputs.append(response)
    return outputs
a = ["Summarize it <> <Meta just suffered a major Facebook ad glitch that has advertisers asking about refunds.>"]
print(generateOutputs(a)[0])
#

