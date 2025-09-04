act_labels=('depressed mood' 'low self-esteem' 'negativity bias' 'guilt' 'risk-aversion' 'self-destruction' 'manic mood' 'grandiosity' 'positivity bias' 'lack of remorse' 'risk-seeking' 'hostility')
strlist=('0.25') 
# model="Llama-3.1-8B-Instruct"   
# model="Qwen3-0.6B"   
model="Qwen3-32B"
gpus="0"

python query_dup.py --gpus "${gpus}" --engine "${model}"

python store.py --engine "str_none+${model}"

for act in "${act_labels[@]}"; do
    for strength in "${strlist[@]}"; do
        python store.py --engine "str_${act}_${strength}+${model}"
    done
done

act_labels=('random 0' 'random 1' 'random 2' 'random 3' 'random 4' 'random 5' 'random 6' 'random 7' 'random 8' 'random 9')
strlist=('0.25') 

for act in "${act_labels[@]}"; do
    for strength in "${strlist[@]}"; do
        python store.py --engine "str_${act}_${strength}+${model}"
    done
done