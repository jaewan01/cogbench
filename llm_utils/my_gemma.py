import os
import torch
from ..base_classes import LLM 

import pdb
import sys
import gc

class GEMMA(LLM):
    def __init__(self, llm_info):
        super().__init__(llm_info)
        engine, max_tokens, self.temperature = llm_info

        cur_dir = os.getcwd()
        os.chdir('../../../../')
        os.environ['HF_HOME'] = '../data/SooYong/hf_home'
        os.environ['HF_HUB_CACHE'] = '../data/SooYong/hub'
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] =  "5,6"

        from transformers import AutoTokenizer
        from transformer_lens import HookedTransformer

        sys.path.append('.')

        from utils_gemma_9b import Config, Data_Manager, load_label_dict, model_selection, remove_model_hooks

        self.cfg = Config()
        
        dm = Data_Manager(self.cfg)
        df, self.label_dict, sev_dict, act_max_dict, abbv_dict  = dm.load_data()

        hook_device = 'cuda:0'
        self.token_device = 'cuda:0'

        model_id = "microsoft/" + engine
        n_devices = len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) 
        self.model, self.tokenizer = model_selection(self.cfg.model_id, library='lens', n_devices=n_devices, dtype='float16')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        total_n_layers = self.model.cfg.n_layers
        self.hook_layers = []
        self.hook_layers.append(int(total_n_layers * 0.4) - 1)  
        self.hook_layers.append(int(total_n_layers * 0.6) - 1)
        self.hook_layers.append(int(total_n_layers * 0.8) - 1)
        
        # self.hook_layers = [9, 19, 29]  # For Gemma-9B, we use layers 9, 19, and 29

        # self.model = remove_model_hooks(self.model, keep_hook_layers=self.hook_layers)

        # self.sae = dm.load_pt('sae').eval().to(hook_device)

        self.saes = []

        for layer in self.hook_layers:
            self.cfg.hook_layer = layer
            dm = Data_Manager(self.cfg)
            self.saes.append(dm.load_pt('sae').eval().to(hook_device))
        
        os.chdir(cur_dir)
        
    def _generate_batched(self, text, tmp, max_tokens, return_prob=False, add_generation_prompt=True):

        input_prompt_batch = []
        
        for q in text:
            bench_prompt = [{
                'role': 'user',
                'content': q
            }]
            input_prompt_batch.append(bench_prompt)
            # input_prompt_batch.append(q)
        
        if self.act.startswith('random'):
            rand_num = self.act.split()[1]
            steer_vecs = []
            for layer in self.hook_layers:
                random_vec_file = f"../../../../data_rand/gemma-9b-{layer}-it/random_vector_{rand_num}.pt"
                steer_vec = torch.load(random_vec_file, weights_only=True).to(self.token_device)
                steer_vecs.append(steer_vec)
        elif self.act == 'none':
            steer_vecs = []
            for layer in self.hook_layers:
                random_vec_file = f"../../../../data_rand/gemma-9b-{layer}-it/random_vector_0.pt"
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
                

        def modify_activations0(activations, hook):
            norm_orig = activations[:, -1, :].norm(dim=-1, keepdim=True)
            norm_sae = steer_vecs[0].norm(dim=-1, keepdim=True).to(activations.device)
            scale = norm_orig / norm_sae

            activations[:, -1, :] = activations[:, -1, :] + steer_vecs[0].to(activations.device) * scale * self.str_str

            norm_new = activations[:, -1, :].norm(dim=-1, keepdim=True)
            activations[:, -1, :] = activations[:, -1, :] / norm_new * norm_orig

            return activations

        def modify_activations1(activations, hook):
            norm_orig = activations[:, -1, :].norm(dim=-1, keepdim=True)
            norm_sae = steer_vecs[1].norm(dim=-1, keepdim=True).to(activations.device)
            scale = norm_orig / norm_sae

            activations[:, -1, :] = activations[:, -1, :] + steer_vecs[1].to(activations.device) * scale * self.str_str

            norm_new = activations[:, -1, :].norm(dim=-1, keepdim=True)
            activations[:, -1, :] = activations[:, -1, :] / norm_new * norm_orig

            return activations
    
        def modify_activations2(activations, hook):
        
            norm_orig = activations[:, -1, :].norm(dim=-1, keepdim=True)
            norm_sae = steer_vecs[2].norm(dim=-1, keepdim=True).to(activations.device)
            scale = norm_orig / norm_sae

            activations[:, -1, :] = activations[:, -1, :] + steer_vecs[2].to(activations.device) * scale * self.str_str

            norm_new = activations[:, -1, :].norm(dim=-1, keepdim=True)
            activations[:, -1, :] = activations[:, -1, :] / norm_new * norm_orig

            return activations
        
        with torch.no_grad():
            for i in range(len(self.hook_layers)):
                hook = f'blocks.{self.hook_layers[i]}.hook_resid_post'
                self.model.add_hook(hook, modify_activations0 if i == 0 else (modify_activations1 if i == 1 else modify_activations2))
            
            if return_prob:
                tokens = self.tokenizer.apply_chat_template(input_prompt_batch, tokenize=True, add_generation_prompt=add_generation_prompt, return_tensors="pt", padding=True).to(self.token_device)
                outputs = self.model(tokens)
                self.model.remove_all_hook_fns()
                outputs = outputs[:, -1, :]  # Get the logits for the last token
            else:
                tokens = self.tokenizer.apply_chat_template(input_prompt_batch, tokenize=True, add_generation_prompt=add_generation_prompt, return_tensors="pt", padding=True).to(self.token_device)
                output_text = self.model.generate(tokens, max_new_tokens=max_tokens, temperature=tmp, do_sample=False)[:, tokens.shape[1]:]
                self.model.remove_all_hook_fns()
                outputs = self.tokenizer.batch_decode(output_text, skip_special_tokens=True)

        # del tokens, steer_vecs
        # gc.collect()
        # torch.cuda.empty_cache()

        return outputs
        
        

    def _generate(self, text, tmp, max_tokens, return_prob=False):

        str_str = self.str_str

        # input_prompt_batch = [{'role': 'user', 'content': text}]
        input_prompt_batch = [{
                'role': 'user',
                'content': text
            }] 
        
        # itv_str_tensor = torch.tensor([str_str]).unsqueeze(1).to(self.token_device)
        if self.act.startswith('random'):
            rand_num = self.act.split()[1]
            steer_vecs = []
            for layer in self.hook_layers:
                random_vec_file = f"../../../../data_rand/gemma-9b-{layer}-it/random_vector_{rand_num}.pt"
                steer_vec = torch.load(random_vec_file, weights_only=True).to(self.token_device)
                steer_vecs.append(steer_vec)
        elif self.act == 'none':
            steer_vecs = []
            for layer in self.hook_layers:
                random_vec_file = f"../../../../data_rand/gemma-9b-{layer}-it/random_vector_0.pt"
                steer_vec = torch.load(random_vec_file, weights_only=True).to(self.token_device)
                steer_vecs.append(steer_vec)
        else:
            steer_vecs = []
            for layer in self.hook_layers:
                # random_vec_file = f"../../../../{self.cfg.sae_dir}/layer_{layer}/contrast{self.cfg.feat_type}_emo_{layer}_v1.pt"
                random_vec_file = f"../../../../{self.cfg.sae_dir}/layer_{layer}/contrast{self.cfg.feat_type}_{layer}_v{self.cfg.current_v}.pt"
                itv_W = torch.load(random_vec_file, weights_only=True).to(self.token_device)
                steer_vecs.append(itv_W[self.label_dict[self.act]])
                # index = self.hook_layers.index(layer)
                # steer_vecs.append(self.saes[index].decoder.weight.T[self.label_dict[self.act]].to(self.token_device))
        
        def modify_activations0(activations, hook):
            norm_orig = activations[:, -1, :].norm(dim=-1, keepdim=True)
            norm_sae = steer_vecs[0].norm(dim=-1, keepdim=True).to(activations.device)
            scale = norm_orig / norm_sae

            activations[:, -1, :] = activations[:, -1, :] + steer_vecs[0].to(activations.device) * scale * self.str_str

            norm_new = activations[:, -1, :].norm(dim=-1, keepdim=True)
            activations[:, -1, :] = activations[:, -1, :] / norm_new * norm_orig

            return activations

        def modify_activations1(activations, hook):
            norm_orig = activations[:, -1, :].norm(dim=-1, keepdim=True)
            norm_sae = steer_vecs[1].norm(dim=-1, keepdim=True).to(activations.device)
            scale = norm_orig / norm_sae

            activations[:, -1, :] = activations[:, -1, :] + steer_vecs[1].to(activations.device) * scale * self.str_str

            norm_new = activations[:, -1, :].norm(dim=-1, keepdim=True)
            activations[:, -1, :] = activations[:, -1, :] / norm_new * norm_orig

            return activations
    
        def modify_activations2(activations, hook):
        
            norm_orig = activations[:, -1, :].norm(dim=-1, keepdim=True)
            norm_sae = steer_vecs[2].norm(dim=-1, keepdim=True).to(activations.device)
            scale = norm_orig / norm_sae

            activations[:, -1, :] = activations[:, -1, :] + steer_vecs[2].to(activations.device) * scale * self.str_str

            norm_new = activations[:, -1, :].norm(dim=-1, keepdim=True)
            activations[:, -1, :] = activations[:, -1, :] / norm_new * norm_orig

            return activations
        
        with torch.no_grad():
            for i in range(len(self.hook_layers)):
                hook = f'blocks.{self.hook_layers[i]}.hook_resid_post'
                self.model.add_hook(hook, modify_activations0 if i == 0 else (modify_activations1 if i == 1 else modify_activations2))
            
            if return_prob:
                tokens = self.tokenizer.apply_chat_template(input_prompt_batch, tokenize=True, add_generation_prompt=False, return_tensors="pt", padding=True).to(self.token_device)
                outputs = self.model(tokens)
                self.model.remove_all_hook_fns()
                outputs = outputs[:, -1, :]  # Get the logits for the last token
            else:
                tokens = self.tokenizer.apply_chat_template(input_prompt_batch, tokenize=True, add_generation_prompt=True, return_tensors="pt", padding=True).to(self.token_device)
                output_text = self.model.generate(tokens, max_new_tokens=max_tokens, temperature=tmp, do_sample=False)[:, tokens.shape[1]:]
                self.model.remove_all_hook_fns()
                outputs = self.tokenizer.batch_decode(output_text, skip_special_tokens=True)

        return outputs[0]