import os
import warnings
warnings.filterwarnings('ignore')  # Suppress all other warnings
# os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Suppress transformer warnings
import torch
from ..base_classes import LLM 

import pdb
import sys
import gc

class LLAMA(LLM):
    def __init__(self, llm_info):
        super().__init__(llm_info)
        engine, max_tokens, self.temperature, gpus = llm_info

        cur_dir = os.getcwd()
        os.chdir('../../../../')
        os.environ['HF_HOME'] = 'HF/hf_home'
        os.environ['HF_HUB_CACHE'] = 'HF/hub'
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] =  gpus

        from transformers import AutoTokenizer, AutoModelForCausalLM
        from transformer_lens import HookedTransformer

        sys.path.append('.')
        
        from Jaewan.utils.utils_jw import Config, Data_Manager

        self.cfg = Config()
        dm = Data_Manager(self.cfg)
        self.label_dict = dm.load_dict("label")[0]
        # df, self.label_dict, sev_dict, act_max_dict, abbv_dict  = dm.load_data()

        hook_device = 'cuda:0'
        self.token_device = 'cuda:0'

        model_id = "meta-llama/" + engine
        precision = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(model_id,
            device_map='auto',  # device_map options 'auto', 'balanced', 'balanced_low_0', 'sequential'
            torch_dtype=precision,
            attn_implementation="flash_attention_2",
            # quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        )
        # self.model = HookedTransformer.from_pretrained_no_processing(model_id, n_devices=len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")), device='cuda', dtype='bfloat16')
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left' # 'right'

        # self.sae = dm.load_pt('sae').eval().to(hook_device)
        
        if model_id == "meta-llama/Llama-3.1-8B-Instruct":
            self.hook_layers = [5,11,18,24]
        elif model_id == "meta-llama/Llama-3.2-3B-Instruct":
            self.hook_layers = [4,10,15,21]
        elif model_id == "meta-llama/Llama-3.3-70B-Instruct":
            self.hook_layers = [15,31,47,63]

        # self.model = remove_model_hooks(self.model, keep_hook_layers=self.hook_layers)

        # self.sae = dm.load_pt('sae').eval().to(hook_device)

        self.saes = []

        for layer in self.hook_layers:
            self.saes.append(dm.load_pt('sae', layer).eval().to(hook_device))
        
        os.chdir(cur_dir)
        
    def _generate_batched(self, input_prompt_batch, tmp, max_tokens, return_prob=False, add_generation_prompt=True):

        if self.act.startswith('random'):
            rand_num = self.act.split()[1]
            steer_vecs = []
            for layer in self.hook_layers:
                random_vec_file = f"../../../../data_rand/llama-3.1-8b-{layer}-it/random_vector_{rand_num}.pt"
                steer_vec = torch.load(random_vec_file, weights_only=True).to(self.token_device)
                steer_vecs.append(steer_vec)
        elif self.act == 'none':
            steer_vecs = []
            for layer in self.hook_layers:
                random_vec_file = f"../../../../data_rand/llama-3.1-8b-{layer}-it/random_vector_0.pt"
                steer_vec = torch.load(random_vec_file, weights_only=True).to(self.token_device)
                steer_vecs.append(steer_vec)
        else:
            steer_vecs = []
            for layer in self.hook_layers:
                # random_vec_file = f"../../../../{self.cfg.sae_dsir}/layer_{layer}/contrast{self.cfg.feat_type}_emo_{layer}_v1.pt"
                # random_vec_file = f"../../../../{self.cfg.sae_dir}/layer_{layer}/contrast{self.cfg.feat_type}_{layer}_v{self.cfg.current_v}.pt"
                # itv_W = torch.load(random_vec_file, weights_only=True).to(self.token_device)
                # steer_vecs.append(itv_W[self.label_dict[self.act]])
                index = self.hook_layers.index(layer)
                steer_vecs.append(self.saes[index].decoder.weight.T[self.label_dict[self.act]].to(self.token_device))

        handles = []
        for i in range(len(self.hook_layers)):
            layer = self.hook_layers[i]

            def modify_activations(module, input, activations, layer=layer, steer_vec = steer_vecs[i]):
                if activations.shape[1] == 1:
                    norm_orig = activations[:, -1, :].norm(dim=-1, keepdim=True)
                    norm_sae = steer_vec.norm(dim=-1, keepdim=True).to(activations.device)
                    scale = norm_orig / norm_sae

                    activations[:, -1, :] = activations[:, -1, :] + steer_vec.to(activations.device) * scale * self.str_str

                    norm_new = activations[:, -1, :].norm(dim=-1, keepdim=True)
                    activations[:, -1, :] = activations[:, -1, :] / norm_new * norm_orig

                    return activations
                else:
                    return activations

            target_layer = self.model.model.layers[layer]
            handle = target_layer.register_forward_hook(modify_activations)
            handles.append(handle)

        with torch.no_grad():
            rendered = [self.tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=add_generation_prompt) for c in input_prompt_batch]
            tokens = self.tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False).to(self.token_device)
            if return_prob:
                outputs = self.model.forward(tokens)
                outputs = outputs[:, -1, :]  # Get the logits for the last token
            else:
                output_text = self.model.generate(
                    **tokens, max_new_tokens=max_tokens, 
                    temperature=tmp, 
                    do_sample=True, 
                    pad_token_id=self.tokenizer.eos_token_id, 
                    use_cache=True, 
                    cache_implementation='dynamic', 
                    repetition_penalty=1.0
                )[:, tokens.input_ids.shape[1]:]
                outputs = self.tokenizer.batch_decode(output_text, skip_special_tokens=True)

        for handle in handles:
            handle.remove()

        torch.cuda.empty_cache()
        gc.collect()

        return outputs

    def _generate(self, text, tmp, max_tokens, return_prob=False, add_generation_prompt=True):
        
        if self.act.startswith('random'):
            rand_num = self.act.split()[1]
            steer_vecs = []
            for layer in self.hook_layers:
                random_vec_file = f"../../../../data_rand/llama-3.1-8b-{layer}-it/random_vector_{rand_num}.pt"
                steer_vec = torch.load(random_vec_file, weights_only=True).to(self.token_device)
                steer_vecs.append(steer_vec)
        elif self.act == 'none':
            steer_vecs = []
            for layer in self.hook_layers:
                random_vec_file = f"../../../../data_rand/llama-3.1-8b-{layer}-it/random_vector_0.pt"
                steer_vec = torch.load(random_vec_file, weights_only=True).to(self.token_device)
                steer_vecs.append(steer_vec)
        else:
            steer_vecs = []
            for layer in self.hook_layers:
                # random_vec_file = f"../../../../{self.cfg.sae_dir}/layer_{layer}/contrast{self.cfg.feat_type}_emo_{layer}_v1.pt"
                # random_vec_file = f"../../../../{self.cfg.sae_dir}/layer_{layer}/contrast{self.cfg.feat_type}_{layer}_v{self.cfg.current_v}.pt"
                # itv_W = torch.load(random_vec_file, weights_only=True).to(self.token_device)
                # steer_vecs.append(itv_W[self.label_dict[self.act]])
                index = self.hook_layers.index(layer)
                steer_vecs.append(self.saes[index].decoder.weight.T[self.label_dict[self.act]].to(self.token_device))

        handles = []
        for i in range(len(self.hook_layers)):
            layer = self.hook_layers[i]

            def modify_activations(module, input, activations, layer=layer, steer_vec = steer_vecs[i]):
                if activations.shape[1] == 1:
                    norm_orig = activations[:, -1, :].norm(dim=-1, keepdim=True)
                    norm_sae = steer_vec.norm(dim=-1, keepdim=True).to(activations.device)
                    scale = norm_orig / norm_sae

                    activations[:, -1, :] = activations[:, -1, :] + steer_vec.to(activations.device) * scale * self.str_str

                    norm_new = activations[:, -1, :].norm(dim=-1, keepdim=True)
                    activations[:, -1, :] = activations[:, -1, :] / norm_new * norm_orig

                    return activations
                else:
                    return activations

            target_layer = self.model.model.layers[layer]
            handle = target_layer.register_forward_hook(modify_activations)
            handles.append(handle)

        with torch.no_grad():
            rendered = [self.tokenizer.apply_chat_template(text, tokenize=False, add_generation_prompt=add_generation_prompt)]
            tokens = self.tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False).to(self.token_device)
            if return_prob:
                outputs = self.model.forward(tokens)
                outputs = outputs[:, -1, :]  # Get the logits for the last token
            else:
                output_text = self.model.generate(
                    **tokens, max_new_tokens=max_tokens, 
                    temperature=tmp, 
                    do_sample=True, 
                    pad_token_id=self.tokenizer.eos_token_id, 
                    use_cache=True, 
                    cache_implementation='dynamic', 
                    repetition_penalty=1.0
                )[:, tokens.input_ids.shape[1]:]
                outputs = self.tokenizer.batch_decode(output_text, skip_special_tokens=True)
        
        for handle in handles:
            handle.remove()
        
        torch.cuda.empty_cache()
        gc.collect()

        return outputs[0]