import openai
import json
import asyncio
import copy 
import re
import random

def load_or_convert_to_dataframe(dataset_path):

    if 'jsonl' in dataset_path:
         dataset = [json.loads(l) for l in open(dataset_path, "r")]
            # import pdb;pdb.set_trace()
    elif 'json' in dataset_path:
        with open(dataset_path, 'r') as file:
            dataset = json.load(file)
    else:
         raise ValueError("Unsupported file format. Please provide a .json or .jsonl file.")
    return dataset





def gpt_output_generation_encode_prompt(task_dict):
    """Encode multiple prompt instructions into a single string."""
    prompt=""

    (instruction, task_input) = task_dict["instruction"], task_dict["input"]
    if  instruction=="NA":
        return ""
    prompt += instruction+"\n\n"

    if task_input=="" or "<noinput>" in task_input  or "NA" in task_input:
        return prompt
    prompt += task_input+"\n"
    return prompt

def anthropic_output_generation_encode_prompt(task_dict):
    """Encode multiple prompt instructions into a single string."""
    prompt="Human: "

    (instruction, task_input) = task_dict["instruction"], task_dict["input"]
    if  instruction=="NA":
        return ""
    prompt += instruction+"\n\n"

    if (task_input=="") or ("<noinput>" in task_input)  or  (task_input=="NA") :
        return prompt+'Assistant:'
    prompt += task_input+"\n\n"
    return prompt+'Assistant:'



async def eval_dispatch_openai_requests(
    messages_list,
    model,
    temperature,
    max_tokens,
    top_p,
    frequency_penalty,
    presence_penalty,
    timeout_seconds=10,
    base_wait_time=5,  # Base wait time in seconds
    backoff_factor=1.5,  #  # Adding a new parameter for timeout
) :
    """
    Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
        frequency_penalty: Frequency penalty to use for the model.
        presence_penalty: Presence penalty to use for the model.
        timeout_seconds: Maximum number of seconds to wait for a response.

    Returns:
        List of responses from OpenAI API.
    """
    async def send_request(message):
        return await openai.ChatCompletion.acreate(
            model=model,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )

    async def request_until_success(message):
        while True:
            wait_time = base_wait_time
            try:
                return await asyncio.wait_for(send_request(message), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                print(f"Timeout! Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)  # Wait for the calculated time
                wait_time *= backoff_factor  # Increase the wait time
    
    
    async_responses = [
        request_until_success(x)
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)


async def generation_dispatch_openai_requests(
    messages_list,
    model,
    temperature,
    max_tokens,
    top_p,
    frequency_penalty,
    presence_penalty,
    stop_sequences
):
    """
    Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.

    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop_sequences
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)

def eval_make_prompt (template,val_dict):

    text_to_format = re.findall("{([^ \s]+?)}", template)
    prompt = copy.deepcopy(template)
    for to_format in text_to_format:
        prompt = prompt.replace("{" + to_format + "}", val_dict[to_format] , 1)

    return prompt

def eval_encode_prompt(prompt,instruction, model_output,reference_output,args):
    """Encode multiple prompt instructions into a single string."""

    if args.reference_first:
        output_list=[reference_output,model_output]
    else:
        output_list=[model_output,reference_output]

    mapping_dict_output={"instruction":instruction}
    mapping_dict_generator={}
    for idx in range(2):
        mapping_dict_output['output_'+str(idx+1)]=output_list[idx]['output']
        mapping_dict_generator['model_'+str(idx+1)]=output_list[idx]['generator']


    filled_prompt=eval_make_prompt(prompt,mapping_dict_output)
    

    return filled_prompt, mapping_dict_generator

def task_encode_prompt(prompt,prompt_instructions,args):
    """Encode multiple prompt instructions into a single string."""
    prompt+="\nList of "+str(args.num_instructions_to_generate_per_batch)+" tasks:\n"
    
    if len(prompt_instructions)>0:
        for idx, task_dict in enumerate(prompt_instructions):
            (instruction, task_type, topic, view, difficulty, task_input) = task_dict["instruction"], task_dict["type"], task_dict["topic"], task_dict["view"], task_dict["difficulty"], task_dict["input"]
            instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
            input = "<noinput>" if task_input.lower() == "" else task_input
            prompt += f"###\n"
            prompt += f"{idx + 1}. Type: {task_type}\n"
            prompt += f"Topic: {topic}\n"
            prompt += f"View: {view}\n"
            prompt += f"Difficulty: {difficulty}\n"
            prompt += f"Instruction: {instruction}\n"
            prompt += f"Input: {task_input}\n"
        prompt += f"###\n"
        prompt += f"{idx + 2}. Type:"
    else:
        prompt += f"###\n"
        prompt += f"1. Type:"

    return prompt

def sample_machine_instructions(machine_instructions, n):
    """Sample n machine instructions from a list of machine instructions."""
    return random.sample(machine_instructions, min(n, len(machine_instructions)))



def post_process_gpt3_response(response,seed_number):
    if response is None:
        return []
    raw_instructions = str(seed_number)+". Type: "+ response
    raw_instructions = re.split("###", raw_instructions)
    # raw_instructions = re.split(r"\n\d+\s?\. ", response)
    instructions = []
    for i, inst in enumerate(raw_instructions):
        # if idx==0:
        #     inst="Type: "+inst
        inst = re.sub(r"\s+", " ", inst).strip()
        inst = inst.strip(f"{i+seed_number+1}\.\s+")
        inst = inst.strip()
        inst = inst.strip('"')
        if inst == "":
            continue

        keys = ["Type", "Topic", "View", "Difficulty","Instruction", "Input"]
        values = []
        start = 0

        for key in keys:
            idx = inst.find(key, start)
            if idx != -1:
                values.append(inst[start:idx].strip())
                start = idx + len(key) + 1
        values.append(inst[start:].strip())

        # Remove the first empty value and create the dictionary
        result_dict = result_dict = {key.lower(): value for key, value in zip(keys, values[1:])}

        instructions.append(result_dict)
    return instructions



