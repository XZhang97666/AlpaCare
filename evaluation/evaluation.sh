
ref="text-davinci-003" # your reference model
eng="gpt-3.5-turbo" # your judge 
task="alpaca_eval" #yout task name

data_path='./data/comparsion/'refer_${ref}'/'${task}'/' #  path to gpt evualtion output
for model_name in   'AlpaCare-7b' 'AlpaCare-13b'  # your model for comparsion
do
python evaulation.py --compared_model $model_name --data_path $data_path --engine ${eng} --refer_model ${ref}
done

