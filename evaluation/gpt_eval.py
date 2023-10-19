import os
import sys
sys.path.append('..')  # Add the parent directory to sys.path
import json
import random
from tqdm import tqdm
import argparse
import asyncio
import openai
import time
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
        default="data/comparsion/",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--model_output",
        type=str,
        required=True,
        default=None,
        help="The path to the model output data.",
    )
    parser.add_argument(
        "--reference_output",
        type=str,
        required=True,
        default="./dataset/alpaca_eval/alpaca_eval.json",
        help="The path to the reference output data.",
    )
    parser.add_argument(
        "--output_file_name",
        type=str,
        required=False,
        default=None,
        help="The file name for output.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        default=None,
        help="The model comparsion name.",
    )
    parser.add_argument(
        "--refer_model_name",
        type=str,
        required=False,
        default="text-davinci-003",
        help="The reference model name.",
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
        default=100,
        help="Max input tokens."
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="failed retry times."
    )
    parser.add_argument(
        "--reference_first",
        action="store_true",
        help="If pass reference model will be model_1, otherwise reference model will be model_2.",
    )

    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The task_name for different instruction usage."
    )

    parser.add_argument(
        "--max_test_number",
        type=int,
        default=-1,
        help="set if the test instances is less than generation.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # import pdb;pdb.set_trace()
    model_output=load_or_convert_to_dataframe(args.model_output)
    if args.max_test_number !=-1:
        model_output=model_output[:args.test_number]
    reference_output=load_or_convert_to_dataframe(args.reference_output) 

    if args.task_name=="alpaca_eval" or args.task_name=='medsi':
        instructions=[item['instruction'] for item in reference_output]
    elif args.task_name=="iCliniq":
        instructions=[item['instruction']+"\n\n"+item['input'] for item in model_output]
    else:
        raise ValueError("Unsupported task.")


    if 'response' in model_output[0]:
        model_output=[{"generator":args.model_name,"output":item['response']} for item in model_output]
    else:
        model_output=[{"generator":args.model_name,"output":item['output']} for item in model_output]
    if 'response' in reference_output[0]:
        reference_output=[{"generator":args.refer_model_name,"output":item['response']} for item in reference_output]
    else:
        reference_output=[{"generator": args.refer_model_name,"output":item['output']} for item in reference_output]
    # import pdb;pdb.set_trace()
    assert len(model_output)==len(reference_output)==len(instructions)
    total=len(reference_output)
    progress_bar = tqdm(total=total)

    wait_base = 10
    retry_cnt = 0
    batch_size=args.request_batch_size

    if args.engine=="gpt-3.5-turbo":
        system_prompt='You are a helpful instruction-following assistant that prints the best model by selecting the best outputs for a given instruction.'
        prompt = open("../prompt/alpaca_eval_chat_gpt.txt").read() + "\n"
    else:
        raise ValueError("Unsupported engine.")
    
    results=[]
    target_length=args.max_tokens

    if not os.path.exists(args.batch_dir):
        os.makedirs(args.batch_dir)

    if args.output_file_name is None:
        if args.reference_first:
            args.output_file_name=args.model_name+"_"+args.engine+"_reference_first.jsonl"
        else:
            args.output_file_name=args.model_name+"_"+args.engine+"_reference_last.jsonl"
    output_path=os.path.join(args.batch_dir, args.output_file_name)
    outputs=[]
    if os.path.isfile(output_path):

        with open(output_path, "r") as fin:
            for line in fin:
                prev_output= json.loads(line)
                outputs.append(prev_output)
        print(f"Loaded {len(outputs)} machine-generated instructions")

    print(output_path)
    progress_bar.update(len(outputs))
    

    idx=len(outputs)
    with open(output_path, "a") as fout:
        while idx<total:
            message_list=[]
            model2name_list=[]
            j=0
            while j < min(batch_size,total-idx):
                instr, m_o, r_o=instructions[idx+j],model_output[idx+j], reference_output[idx+j]
                task_prompt, model2name = eval_encode_prompt(prompt,instr,m_o,r_o,args)
                message =[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": task_prompt,
                        },
                ]
                message_list.append(message)
                model2name_list.append(model2name)
                j+=1
            batch_results=[]
            while(len(batch_results)==0):
                try:
                    batch_predictions = asyncio.run(
                        eval_dispatch_openai_requests(
                            messages_list=message_list,
                            model=args.engine,
                            temperature=0,
                            max_tokens=target_length,
                            top_p=1.0,
                            frequency_penalty=0,
                            presence_penalty=0
                        )
                    )
                    for j, message in enumerate(message_list):
                        data = {
                            "prompt": message[1]['content'],
                            "response": batch_predictions[j]['choices'][0]['message']["content"]
                        }
                        batch_results.append(data)

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

            for (model2name,result) in zip(model2name_list,batch_results):
                merged_dict = {**model2name, **result}
                fout.write(json.dumps(merged_dict) + "\n")
                outputs.append(merged_dict)
                progress_bar.update(1)
                idx+=1


                
                
     