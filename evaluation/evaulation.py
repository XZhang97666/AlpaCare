
import re
import os
import numpy as np
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")

    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="eval_task",
    )

    parser.add_argument(
        "--compared_model",
        type=str,
        default=None,
        help="model output to comapre",
    )

    parser.add_argument(
        "--refer_model",
        type=str,
        default='text-davinci-003',
        help="model to reference",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default='./data/comparsion/',
        help="The path of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "--engine",
        type=str,
        default="gpt-3.5-turbo",
        help="engine for evualtion",
    )


    args = parser.parse_args()
    return args

def remove_special_characters(input_string):
    # Replace anything that is not a letter or a number with an empty string
    cleaned_string = re.sub(r'[^a-zA-Z0-9]', '', input_string)
    return cleaned_string

def str_win_calculate(completion,compared_model,refer_model):
    if completion==refer_model:
        return [0]
    elif completion==compared_model:
        return [1]
    elif "Tie" in completion:
        return [0.5]
    else:
        return [np.nan]

def dict_win_calculate(result_dict, compared_model, refer_model):
    if result_dict[compared_model]== result_dict[refer_model]:
        return [0.5]
    elif result_dict[compared_model]> result_dict[refer_model]:
        return [0]
    elif result_dict[compared_model]< result_dict[refer_model]:
        return [1]
    else:
        return [np.nan]

def read_jsonl(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def win_score_list_calculate(data,engine,compared_model, refer_model):
    score_list=[]
    for d_i in data:
        output=d_i["response"]
        model_1=d_i['model_1']
        model_2=d_i['model_2']
        if engine=="gpt-3.5-turbo" or engine=="claude-2":
            output=output.strip()
            output = output.replace("Output (a)", model_1)
            output = output.replace("Output (b)", model_2)
            compared_model=remove_special_characters(compared_model)
            refer_model=remove_special_characters(refer_model)
            output=remove_special_characters(output)
                
            score=str_win_calculate(output, compared_model, refer_model)
            score_list.extend(score)
        else:
            Exception("Invalid engine.")
            break

    return np.array(score_list)
        
def pairwise_mean(arr1, arr2):
    assert len(arr1) == len(arr2), "Arrays must be of the same length"
    
    means = np.where(
        (~np.isnan(arr1) & ~np.isnan(arr2)),
        (arr1 + arr2) / 2,
        np.nan
    )
    return means


                

def chat_gpt_eval_model_win(data_path,compared_model,refer_model,engine):
    refer_first_data=read_jsonl(os.path.join(data_path, compared_model+"_"+engine+"_reference_first.jsonl"))
    refer_last_data=read_jsonl(os.path.join(data_path, compared_model+"_"+engine+"_reference_last.jsonl"))
    refer_first_score_arr=win_score_list_calculate(refer_first_data,engine,compared_model, refer_model)
    refer_last_score_arr=win_score_list_calculate(refer_last_data,engine,compared_model, refer_model)
    
    score_mean=pairwise_mean(refer_first_score_arr, refer_last_score_arr)


    return np.nanmean(score_mean)

if __name__ == "__main__":
    args = parse_args()
    eval_score=chat_gpt_eval_model_win(args.data_path,args.compared_model,args.refer_model,args.engine)

    print(args.compared_model, str(eval_score))