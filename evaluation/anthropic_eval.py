import sys
sys.path.append('..')  # Add the parent directory to sys.path
import random
import time
import asyncio
import anthropic 
from tqdm import tqdm
import json
import argparse
import os
from utils import * 
ANTHROPIC_API_KEY='your_API_here'

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



async def main(args):
    # import pdb;pdb.set_trace()
   
    client =anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, max_retries=args.retries)

    # import pdb;pdb.set_trace()
    model_output=load_or_convert_to_dataframe(args.model_output)
    if args.max_test_number !=-1:
        model_output=model_output[:args.test_number]
    reference_output=load_or_convert_to_dataframe(args.reference_output) 

    if args.task_name=="alpaca_eval" or  args.task_name=="medsi" :
        instructions=[item['instruction'] for item in reference_output]
    elif args.task_name=="iCliniq":
        instructions=[item['instruction']+"\n\n"+item['input'] for item in model_output]
    else:
        import pdb;pdb.set_trace()

    if 'response' in model_output[0]:
        model_output=[{"generator":args.model_name,"output":item['response']} for item in model_output]
    else:
        model_output=[{"generator":args.model_name,"output":item['output']} for item in model_output]
    if 'response' in reference_output[0]:
        reference_output=[{"generator": args.refer_model_name,"output":item['response']} for item in reference_output]
    else:
        reference_output=[{"generator": args.refer_model_name,"output":item['output']} for item in reference_output]
    # import pdb;pdb.set_trace()
    assert len(model_output)==len(reference_output)==len(instructions)
    total=len(reference_output)
    progress_bar = tqdm(total=total)

    wait_base = 10
    retry_cnt = 0

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


    progress_bar.update(len(outputs))

    prompt = open("../prompt/alpaca_eval_anthropic.txt").read() + "\n"
    idx=len(outputs)
    with open(output_path, "a") as fout:
        while idx<total:
            output_i={}
            instr, m_o, r_o=instructions[idx],model_output[idx], reference_output[idx]
            task_prompt, model2name = eval_encode_prompt(prompt,instr,m_o,r_o,args)
            while True:
                try:
                    response=  client.completions.create(prompt=task_prompt, model=args.engine,max_tokens_to_sample=target_length,temperature=0.0)
                    if response.completion == "":
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
                        wait_base = wait_base*1.5

            
            output_i["response"]= response.completion
            output_i['prompt']= task_prompt
            merged_dict = {**model2name, **output_i}
            try:
                fout.write(json.dumps(merged_dict) + "\n")
                outputs.append(merged_dict)
                progress_bar.update(1)
                idx+=1
            except:
                import pdb;pdb.set_trace()
                continue


                


if __name__ == "__main__":
    args = parse_args()
   
    asyncio.run(main(args))


    