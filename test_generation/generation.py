from transformers import AutoModelForSeq2SeqLM,AutoTokenizer, AutoModelForCausalLM, set_seed
import torch
import argparse
import json
import os
from tqdm import tqdm
from utils import *
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="The path of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--max_generation",
        type=int,
        default=-1,
        help="Max input to generate."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Max length for generation"
    )
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "t5" in args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model.to(device)
    model.eval()
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    instruction=None
    if (args.dataset_name=="alpaca_eval"):
        dataset_path="./data/alpaca_eval.json"
        prompt_key = 'instruction'
    elif (args.dataset_name=="iCliniq"):
        dataset_path = './data/iCliniq.json'
        prompt_key=""
        instruction="If you are a doctor, please answer the medical questions based on the patient's description."
    elif (args.dataset_name=="medsi"):
        dataset_path = './data/medsi.jsonl'
        prompt_key="instruction"
    else:
        if args.dataset_path is not None:
            dataset_path = args.dataset_path
            prompt_key = 'instruction'
        else:
            raise ValueError("No dataset found")

    dataset=load_or_convert_to_dataframe(dataset_path)
    if args.max_generation!=-1:
        dataset=dataset[:min(len(dataset),args.max_generation)
        ]
    results=[]
    for idx, point in tqdm(enumerate(dataset)):
        output_point={}
        if prompt_key in point:
            instruction = point[prompt_key]
            if instruction=="NA":
                break
        
        if  'input' in point and point ['input']!='NA' :
            prompt = prompt_input.format_map({"instruction":instruction, 'input':point['input']})
        else:
            prompt = prompt_no_input.format_map({"instruction":instruction})
    
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        generate_ids = model.generate(input_ids,max_length=args.max_length)
        outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        output_point['id']=idx
        output_point["instruction"]=instruction
        if 'input' in point:
            output_point["input"]=point['input']

        output_point['raw_output'] = outputs
        output_point['response'] = outputs.split("Response:")[1]
        results.append(output_point)
    tmp = args.model_name_or_path.split('/')[-1]
    output_dir =  os.path.join('./results/'+tmp, args.dataset_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    saved_name = "seed_" + str(args.seed) + ".json"
    with open(os.path.join(output_dir, saved_name), "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()