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
import logging
from utils import *
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")

# async def dispatch_openai_requests(
#     messages_list,
#     model,
#     temperature,
#     max_tokens,
#     top_p,
#     frequency_penalty,
#     presence_penalty
# ) :
#     """
#     Dispatches requests to OpenAI API asynchronously.
    
#     Args:
#         messages_list: List of messages to be sent to OpenAI ChatCompletion API.
#         model: OpenAI model to use.
#         temperature: Temperature to use for the model.
#         max_tokens: Maximum number of tokens to generate.
#         top_p: Top p to use for the model.

#     Returns:
#         List of responses from OpenAI API.
#     """
#     async_responses = [
#         openai.ChatCompletion.acreate(
#             model=model,
#             messages=x,
#             temperature=temperature,
#             max_tokens=max_tokens,
#             top_p=top_p,
#             frequency_penalty=frequency_penalty,
#             presence_penalty=presence_penalty
#         )
#         for x in messages_list
#     ]
#     return await asyncio.gather(*async_responses)


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
    ),
    parser.add_argument(
        "--task",
        type=str,
        default="gpt_generation",
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
        default=300,
        help="Max input tokens."
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="failed retry times."
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="start generate id"
    )

    parser.add_argument(
        "--end_idx",
        type=int,
        default=1000,
        help="end generate id"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    input_tasks=load_or_convert_to_dataframe(args.input_tasks_path)
    input_tasks = input_tasks[args.start_idx:args.end_idx]
    input_instructions = [
        {
            "instruction": t["instruction"] if "instruction" in t else "",
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
    # import pdb;pdb.set_trace()
    wait_base = 10
    retry_cnt = 0
    batch_size=args.request_batch_size
    if args.task=='iCliniq':
        system_prompt="If you are a doctor, please answer the medical questions based on the patient's description."
    elif args.task=='medis':
        system_prompt= 'You are a medical expert tasked with answering various medical questions.'
    else:
        system_prompt= 'You are a medical expert tasked with answering various medical questions. You MUST generate your response based on the requirements.\n\n\
Here are the requirements:\
1. For multiple-choice, calculation, and classification problems, you can generate intermediate thinking steps if necessary; otherwise, provide the final answer directly.\n\
2. All the intermediate thinking steps must be generated before final answer.\n\
3. For multiple-choice questions, you MUST generate the answer choice in the following format: "The answer is (your choice)." For example:\n\
\"Choose the correct answer. Where in your body will you find the tibia bone?A) Arm B) Foot C) Skull D) Leg\n\n\
The tibia bone is one of the two bones in the lower leg, the other being the fibula. The answer is D) Leg.\"\n\
4. For other types of questions, except multiple-choice, do not use the format mentioned in point 3.'
    results=[]
    target_length=args.max_tokens
    with open(os.path.join(args.batch_dir,args.output_tasks_path), "a") as fout:
        idx=len(machine_output)
        while len(machine_output) < len(input_instructions):
            message_list=[]
            input_list=[]
            j=0
            while j < min(batch_size,total-idx):
                input_task=input_instructions[idx+j]
                input_list.append(input_task)
                task_prompt = gpt_output_generation_encode_prompt(input_task,args)
                message =[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": task_prompt,
                        },
                ]
                message_list.append(message)
                j+=1
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
                            presence_penalty=1.5
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
            for i,result in enumerate(batch_results):
                input_list[i]['output']=result["response"]
                fout.write(json.dumps(input_list[i]) + "\n")
                machine_output.append(input_list[i]['output'])
                progress_bar.update(1)
                idx+=1


