import os
import json
import random
import re
import tqdm
import argparse
import openai
import time
from typing import Any
import logging
from utils import *
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")


random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        default="data/text-davinci-003/",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--input_tasks_path",
        type=str,
        required=False,
        default=None,
        help="The path to the input task data.",
    )

    parser.add_argument(
        "--output_tasks_path",
        type=str,
        required=False,
        default=None,
        help="The path to the output task data.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="davinci",
        help="The engine to use."
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=5,
        help="The number of requests to send to GPT3 at a time."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=600,
        help="Max input tokens."
    )
    parser.add_argument(
        "--max_generation",
        type=int,
        default=-1,
        help="Max input to generate."
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="failed retry times."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=False,
        default=None,
        help="the task to generate"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    input_tasks=load_or_convert_to_dataframe(args.input_tasks_path)
    if args.max_generation!=-1:
         input_tasks=input_tasks[:args.max_generation]
    instruction=""
    if args.task=='iCliniq':
        instruction="If you are a doctor, please answer the medical questions based on the patient's description."
    input_instructions = [
        {
            "instruction": t["instruction"] if "instruction" in t else instruction,
            "input": t["input"] if "input" in t else "",
        } 
        for t in input_tasks
    ]
        
    print(f"Loaded {len(input_instructions)} seed instructions")

    
    os.makedirs(args.batch_dir, exist_ok=True)
    # load the LM-generated instructions
    machine_output = []
    if os.path.exists(os.path.join(args.batch_dir, args.output_tasks_path)):
        with open(os.path.join(args.batch_dir, args.output_tasks_path), "r") as fin:
            for line in fin:
                instruction_info = json.loads(line)
                machine_output.append(instruction_info)
        print(f"Loaded {len(machine_output)} machine-generated outputs")


    
    # now let's generate ouput!
    total=len(input_instructions)
    progress_bar = tqdm.tqdm(total=total)

    progress_bar.update(len(machine_output))

    wait_base = 10
    retry_cnt = 0
    batch_size=args.request_batch_size
    results=[]
    target_length=args.max_tokens
    with open(os.path.join(args.batch_dir,args.output_tasks_path), "a") as fout:
        idx=len(machine_output)
        import pdb;pdb.set_trace()
        while len(machine_output) < len(input_instructions):
            prompts=[]
            input_list=[]
            j=0
            while j < min(batch_size,total-idx):
                input_task=input_instructions[idx+j]
                input_list.append(input_task)
                task_prompt = gpt_output_generation_encode_prompt(input_task,args)
                if task_prompt=="":
                    break
                prompts.append(task_prompt)
                j+=1
            if len(prompts)==0:
                break
            while retry_cnt <= args.retries:
                batch_results=[""] * len(prompts)
                try:
                    # batched example, with 10 story completions per request
                    batch_predictions = openai.Completion.create(
                        model=args.engine,
                        prompt=prompts,
                        max_tokens=target_length,
                        temperature= 0
                    )
                    for choice in batch_predictions.choices:
                        batch_results[choice.index] = choice.text

                    # predictions += batch_results
                    wait_base = 10
                    
                    retry_cnt=0
                    break

                except openai.error.OpenAIError as e:
                    print(f"OpenAIError: {e}.")
                    if "Please reduce the length of the messages or completion" in str(e):
                        target_length = int(target_length * 0.8)
                        print(f"Reducing target length to {target_length}, retrying...")
                    else:
                        retry_cnt += 1
                        print("retry number: ", retry_cnt)
                        time.sleep(wait_base)
                        wait_base = wait_base*1.5

            instructions = []
            for i,result in enumerate(batch_results):
                try:
                    input_list[i]['output']=result
                    fout.write(json.dumps(input_list[i]) + "\n")
                    machine_output.append(input_list[i]['output'])
                    idx+=1
                    progress_bar.update(1)
                except:
                    continue

