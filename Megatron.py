import json
import requests

port_num = 5555
headers = {"Content-Type": "application/json"}

def request_data(data):
    resp = requests.put('http://localhost:{}/generate'.format(port_num),
                        data=json.dumps(data),
                        headers=headers)
    sentences = resp.json()['sentences']
    return sentences



def generateOutputs(dataset):
    outputs = []

    for i in range(len(dataset)):
        sentences = dataset[i]
        data = {
            "sentences": [sentences] * 1,
            "tokens_to_generate": 50,
            "temperature": 1.0,
            "add_BOS": True,
            "top_k": 0,
            "top_p": 0.9,
            "greedy": False,
            "all_probs": False,
            "repetition_penalty": 1.2,
            "min_tokens_to_generate": 2,
        }
        output = request_data(data)
        outputs.append(output)
    return outputs
a = ["Summarize sentence: 'Meta just suffered a major Facebook ad glitch that has advertisers asking about refunds'"]
print(generateOutputs(a)[0])