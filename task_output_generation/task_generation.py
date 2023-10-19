import os
import json
import random
import sys
sys.path.append('..')  # Add the parent directory to sys.path
import tqdm
import argparse
import asyncio
import openai
import time
from utils import *
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")



random.seed(42)





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        default="data/gpt4_generations/",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--seed_tasks_path",
        type=str,
        required=False,
        default=None,
        help="The path to the seed task data.",
    )
    parser.add_argument(
        "--num_instructions_to_generate_total",
        type=int,
        default=300,
        help="",
    )
    parser.add_argument(
        "--num_instructions_to_generate_per_batch",
        type=int,
        default=10,
        help="th",
    )

    parser.add_argument(
        "--engine",
        type=str,
        default="davinci",
        help="The engine to use."
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="output.jsonl",
        help="output path"
    )
    parser.add_argument(
        "--total_num_prompt_instructions",
        type=int,
        default=3,
        help="The total number of instructions to use in the prompt."
    )
    parser.add_argument(
        "--num_prompt_machine_instructions",
        type=int,
        default=0,
        help="The number of machine generated instructions to use in the prompt."
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
        default=500,
        help="Max input tokens."
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="failed retry times."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.seed_tasks_path is not None:
        if 'jsonl' in args.seed_tasks_path:
            seed_tasks = [json.loads(l) for l in open(args.seed_tasks_path, "r")]
        elif 'json' in args.seed_tasks_path:
            with open(args.seed_tasks_path, 'r') as file:
                seed_tasks = json.load(file)


        seed_instructions = [{"instruction": t["instruction"],\
                             "input": t["input"],
                             "type": t["type"],
                             "topic": t["topic"],
                             "view": t["view"], 
                             "difficulty": t["difficulty"]} for t in seed_tasks]
        print(f"Loaded {len(seed_instructions)} seed instructions")
    else:
        seed_instructions=None
        print(f"Generated seed instructions")

    
    os.makedirs(args.batch_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_instructions = []
    if os.path.exists(os.path.join(args.batch_dir, args.output_path)):
        with open(os.path.join(args.batch_dir, args.output_path), "r") as fin:
            for line in fin:
                instruction_info = json.loads(line)
                machine_instructions.append(instruction_info)

                request_idx = instruction_info["request_idx"] + 1
        print(f"Loaded {len(machine_instructions)} machine-generated instructions")

    
    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=args.num_instructions_to_generate_total)
    if machine_instructions:
        progress_bar.update(len(machine_instructions))
    wait_base = 10
    retry_cnt = 0
    batch_size=args.request_batch_size

    system_prompt= "Your objective is to generate diverse medical-related tasks."
    prompt = open("../prompt/instrcution_prompt.txt").read() + "\n"
    results=[]
    target_length=args.max_tokens
    with open(os.path.join(args.batch_dir,args.output_path), "a") as fout:
        while len(machine_instructions) < args.num_instructions_to_generate_total:
            message_list=[]
            for _ in range(batch_size):
                if seed_instructions:
                    # sample machine instructions from the pool
                    prompt_instructions = sample_machine_instructions(
                        machine_instructions, 
                        n=args.num_prompt_machine_instructions)
                    prompt_instructions += random.sample(seed_instructions, args.toal_num_prompt_instructions - len(prompt_instructions))
                    random.shuffle(prompt_instructions)
                else:
                    prompt_instructions=[]
 
                task_prompt = task_encode_prompt(prompt,prompt_instructions,args)

                message =[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": task_prompt,
                        },
                ]
                message_list.append(message)
            while retry_cnt <= args.retries:
                batch_results=[]
                try:
                    batch_predictions = asyncio.run(
                        generation_dispatch_openai_requests(
                            messages_list=message_list,
                            model=args.engine,
                            temperature=1.0,
                            max_tokens=target_length,
                            top_p=1.0,
                            frequency_penalty=0,
                            presence_penalty=1.5,
                            stop_sequences=[ "###\n"+str(args.num_instructions_to_generate_per_batch+1), \
                            str(args.num_instructions_to_generate_per_batch+1)+".", str(args.num_instructions_to_generate_per_batch+1)+" ."]
                        )
                    )
                    for j, message in enumerate(message_list):
                        data = {
                            "prompt": message[1]['content'],
                            "response": batch_predictions[j]['choices'][0]['message']["content"]
                        }
                        batch_results.append(data)

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
            for result in batch_results:
                
                new_instructions = post_process_gpt3_response(result["response"],len(prompt_instructions)+1)
                instructions += new_instructions
            for output in instructions:
                try:
                    fout.write(json.dumps({
                            "instruction": output["instruction"],
                            "input": output["input"],
                            "type": output["type"],
                            "topic": output["topic"],
                            "view": output["view"],
                            "difficulty": output["difficulty"],
                            "request_idx": request_idx
                        }) + "\n")

                    machine_instructions.append(output)
                    progress_bar.update(1)
                    request_idx += 1
                except:
                    continue

