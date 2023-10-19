
eng="text-davinci-003"
python output_generation_completion.py \
  --engine ${eng} \
  --request_batch_size 5 \
  --input_tasks_path /Path/to/input_file.jsonl/json \
  --output_tasks_path  /Output_path_for_task_generation.jsonl 

eng="gpt-3.5-turbo"

python output_generation_chat_completion.py \
  --engine ${eng} \
  --request_batch_size 5 \
  --input_tasks_path /Path/to/input_file.jsonl/json \
  --output_tasks_path  /Output_path_for_task_generation.jsonl 




python output_generation_anthropic.py \
  --input_tasks_path  /Path/to/input_file.jsonl/json \
  --output_tasks_path /Output_path_for_task_generation.jsonl 






