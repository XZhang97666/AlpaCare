from rouge_score import rouge_scorer
import argparse
import json
import os
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
import numpy as np
import tqdm
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
    parser = argparse.ArgumentParser(description="Filter GPT generated data")
    parser.add_argument(
        "--output_dataset_path",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--input_dataset_path",
        type=str,
        default=None,
        help="The path of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "--num_cpus",
        type=int,
        default=16,
        help="num_cpus for using.",
    )
    args = parser.parse_args()
    return args


def main():
    # import pdb;pdb.set_trace()
    args = parse_args()

    # similarities = {}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    

    input_dataset_path = args.input_dataset_path

    output_dataset_path = args.output_dataset_path
    filtered_output_data=[]
    all_instruction_data= []
    with open(input_dataset_path, "r") as fin:
        for line in fin:
            instruction_info = json.loads(line)
            all_instruction_data.append(instruction_info)

    all_instruction_tokens = [scorer._tokenizer.tokenize(inst["instruction"]) for inst in all_instruction_data]


    total = len( all_instruction_data)
    keep = 0
    progress_bar=tqdm.tqdm(total)
    with open(output_dataset_path, "a") as fout:
        for idx, instruction_data_entry in enumerate(all_instruction_data):
            instruction = instruction_data_entry["instruction"]
            new_instruction_tokens = scorer._tokenizer.tokenize(instruction_data_entry["instruction"])
            with Pool(args.num_cpus) as p:
                rouge_scores = p.map(
                    partial(rouge_scorer._score_lcs, new_instruction_tokens),
                    all_instruction_tokens,
                )
            rouge_scores = [score.fmeasure for score in rouge_scores]
            most_similar_instructions = {
            all_instruction_data[i]['instruction']: rouge_scores[i] for i in np.argsort(rouge_scores)[-11:][::-1]
            if i != idx
            }
            if max(most_similar_instructions.values()) > 0.7:
                continue
            else:
                keep += 1
            fout.write(json.dumps(instruction_data_entry) + "\n")

            filtered_output_data.append(instruction_data_entry)
            progress_bar.update(1)
        print(f"Generated {total} instructions, kept {keep} instructions")

if __name__ == "__main__":
    main()