import torch
import json
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

# This section gives output of the model given an image and an optional prompt.
def test_model_with_image(image):
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",
                                                            torch_dtype=torch.float16,
                                                            low_cpu_mem_usage=True,
                                                            load_in_4bit=True)
    raw_image = Image.open(image).convert('RGB')
    prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
    # prompt=""

    inputs = processor(prompt, raw_image, return_tensors="pt").to("cuda", torch.float16)

    output = model.generate(**inputs, max_new_tokens=100)

    print(processor.decode(output[0], skip_special_tokens=True))



# This section outputs the data from filtered-dataset.json file

def process_answers():
    with open("filtered_dataset.json","r") as f:
        data = json.load(f)


    answers = []
    for i in range(len(data)):
        a = []
        for j in data[i]['conversations']:
            if j['from']=="gpt":
                a.append(j['value'])
        if len(a)>0:
            obj={}
            obj['id'] = i+1
            obj['image'] = data[i]['image']
            conv = [{
                "from":"human",
                "value":"[INST] <image>\nWhat is shown in this image? [/INST]"          
            },
            {
                "from":"gpt",
                "value":' '.join(a)
            }]
            obj['conversations']=conv
            answers.append(obj)
    with open("final_data.json","w+") as output:
        json.dump(answers, output, indent=4)


def load_preprocess_images():
    with open("filtered_dataset.json","r") as f:
        data = json.load(f)
    images=[]
    for i in range(len(data)):
        images.append(data[i]['image'])
        image = Image.open(image).convert('RGB')



def process_captions():
    with open("filtered_dataset.json","r") as f:
        data = json.load(f)


    answers = []
    for i in range(len(data)):
        a = []
        for j in data[i]['conversations']:
            if j['from']=="gpt":
                a.append(j['value'])
        if len(a)>0:
            obj={}
            obj['id'] = i+1
            obj['image'] = data[i]['image']
            obj['caption']=' '.join(a)
            answers.append(obj)
    with open("final_data.json","w+") as output:
        json.dump(answers, output, indent=4)

def get_length():
     with open("filtered_dataset.json","r") as f:
        data = json.load(f)
        print(len(data))

get_length()