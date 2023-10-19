
export CUDA_VISIBLE_DEVICES=0
for dataset_name in  'alpaca_eval' "medsi"  'iCliniq' 
do
    python generation.py --dataset_name ${dataset_name} \
     --model_name_or_path /Path/to/your/model \
    --max_generation 1000 #  set the number of max generation in your dataset if needed
done

