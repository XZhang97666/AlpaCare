
ref_model='text-davinci-003' # the reference model you want to compare
task_name='alpaca_eval' # the task you want to compare

for model_name in 'AlpaCare-7b' 'AlpaCare-13b'  # your model for comparsion
do 
python gpt_eval.py --model_output  your_model_generation_output_path.json/jsonl \
                         --reference_output reference_model_generation_output_path.json/jsonl \
                         --model_name $model_name \
                          --refer_model_name $ref_model \
                         --engine "gpt-3.5-turbo"\
                         --reference_first \
                         --task_name $task_name \
                         --batch_dir "./data/comparsion/refer_"${ref_model}"/"${task_name}"/" \
                         --max_test_number 1000 # max number of instance you want to evaluate 

python gpt_eval.py --model_output  your_model_generation_output_path.json/jsonl \
                         --reference_output  reference_model_generation_output_path.json/jsonl \
                         --model_name $model_name \
                          --refer_model_name $ref_model \
                         --engine "gpt-3.5-turbo" \
                         --task_name $task_name \
                         --batch_dir "./data/comparsion/refer_"${ref_model}"/"${task_name}"/" \
                         --max_test_number 1000  # max number of instance you want to evaluate 

done


for model_name in 'AlpaCare-7b' 'AlpaCare-13b'  # your model for comparsion
do 
python anthropic_eval.py --model_output  your_model_generation_output_path.json/jsonl \
                         --reference_output reference_model_generation_output_path.json/jsonl \
                         --model_name $model_name \
                        --refer_model_name $ref_model \
                         --reference_first \
                         --task_name $task_name \
                         --batch_dir "./data/comparsion/refer_"${ref_model}"/"${task_name}"/" \
                         --max_test_number 1000  # max number of instance you want to evaluate 

python anthropic_eval.py --model_output  your_model_generation_output_path.json/jsonl \
                         --reference_output  reference_model_generation_output_path.json/jsonl \
                         --model_name $model_name \
                        --refer_model_name $ref_model \
                         --task_name $task_name \
                         --batch_dir "./data/comparsion/refer_"${ref_model}"/"${task_name}"/" \
                         --max_test_number 1000  # max number of instance you want to evaluate 

done


