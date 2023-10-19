import os
import json
import random
import tqdm
import argparse
import sys
sys.path.append('..')  # Add the parent directory to sys.path
import anthropic 
import time
import logging
from utils import *
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY="your API key"



random.seed(42)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        default="data/",
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
        default="claude-2",
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

    client =anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, max_retries=args.retries)
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
    target_length=args.max_tokens
    with open(os.path.join(args.batch_dir,args.output_tasks_path), "a") as fout:
        idx=len(machine_output)
        while len(machine_output) < len(input_instructions):
            output_i={}
            input_task=input_instructions[idx]
            task_prompt = anthropic_output_generation_encode_prompt(input_task,args)
            while True:
                try:
                    response=  client.completions.create(prompt=task_prompt, model=args.engine,max_tokens_to_sample=target_length,temperature=0.0)
                    if response.completion == "":
                        retry_cnt +=1
                        print("retry number: ", retry_cnt)
                        time.sleep(wait_base)
                        wait_base = wait_base*1.5
                    else:
                        wait_base = 10
                        retry_cnt =0
                        break
                except:
                        print("retry number: ", retry_cnt)
                        time.sleep(wait_base)
                        retry_cnt +=1
                        wait_base = wait_base*1.5

            
            output_i["response"]= response.completion
            merge_dict={**input_task,**output_i}
            fout.write(json.dumps(merge_dict) + "\n")
            machine_output.append(output_i)
            progress_bar.update(1)
            idx+=1