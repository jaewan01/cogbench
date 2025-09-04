import os
import torch
from ..base_classes import LLM 

import pdb
import sys
import gc

class PHI(LLM):
    def __init__(self, llm_info):
        super().__init__(llm_info)
        engine, max_tokens, self.temperature, self.act, self.str_str = llm_info

        cur_dir = os.getcwd()
        os.chdir('../../../../')
        os.environ['HF_HOME'] = '../data/SooYong/hf_home'
        os.environ['HF_HUB_CACHE'] = '../data/SooYong/hub'
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] =  "5,6"

        from transformers import AutoTokenizer
        from transformer_lens import HookedTransformer

        sys.path.append('.')

        from utils_phi import Config, Data_Manager, load_label_dict, model_selection

        self.cfg = Config()
        dm = Data_Manager(self.cfg)
        df, self.label_dict, sev_dict, act_max_dict, abbv_dict  = dm.load_data()

        hook_device = 'cuda:0'
        self.token_device = 'cuda:0'

        model_id = "microsoft/" + engine
        n_devices = len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) - 1
        self.model, self.tokenizer = model_selection(self.cfg.model_id, library='lens', n_devices=2, dtype='float16')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.sae = dm.load_pt('sae').eval().to(hook_device)
        
        os.chdir(cur_dir)
        
    def _generate_batched(self, text, tmp, max_tokens):


        str_list = [self.str_str] * len(text)
        input_prompt_batch = []
        
        for q in text:
            prompt = [{
                'role': 'user',
                'content': q
            }]
            input_prompt_batch.append(prompt)
        
        itv_str_tensor = torch.tensor(str_list).unsqueeze(1).to(self.token_device)
        if self.act == 'random':
            itv_W = self.sae.decoder.weight.T[self.label_dict['anger']].to(self.token_device)
            itv_W = torch.rand_like(itv_W).to(self.token_device)
        else:
            itv_W = self.sae.decoder.weight.T[self.label_dict[self.act]].to(self.token_device)
        itv_W_batch = itv_W.repeat(len(input_prompt_batch), 1).unsqueeze(1)

        def modify_activations(activations, hook):
            if activations.shape[1] > 1:
                return activations 
            else:
                base_norm = activations.norm(dim=2)
                itv_norm = itv_W_batch.norm(dim=2).to(activations.device)            
                alpha_itv = (base_norm / itv_norm)
                alpha_itv = torch.mul(alpha_itv, itv_str_tensor.to(activations.device))
                activations = activations + (itv_W_batch.to(activations.device) * alpha_itv[:, :, None]) 
                return activations 
        
        self.model.add_hook(self.cfg.hook_name, modify_activations)
        tokens = self.tokenizer.apply_chat_template(input_prompt_batch, tokenize=True, add_generation_prompt=True, return_tensors="pt", padding=True).to(self.token_device)
        output_text = self.model.generate(tokens, max_new_tokens=max_tokens, temperature=self.temperature, do_sample=False, verbose=True)[:, tokens.shape[1]:]
        self.model.remove_all_hook_fns()

        # convert the output tokens to text
        output_text = self.tokenizer.batch_decode(output_text, skip_special_tokens=True)

        # clean up memory
        gc.collect()
        torch.cuda.empty_cache()

        return output_text

    def _generate(self, text, tmp, max_tokens, str_str=None):

        if str_str is None:
            str_str = self.str_str

        input_prompt_batch = [{'role': 'user', 'content': text}]
        
        
        itv_str_tensor = torch.tensor([str_str]).unsqueeze(1).to(self.token_device)
        if self.act == 'random':
            itv_W = self.sae.decoder.weight.T[self.label_dict['anger']].to(self.token_device)
            itv_W = torch.rand_like(itv_W).to(self.token_device)
        else:
            itv_W = self.sae.decoder.weight.T[self.label_dict[self.act]].to(self.token_device)
        itv_W_batch = itv_W.repeat(1, 1).unsqueeze(1)

        def modify_activations(activations, hook):
            if activations.shape[1] > 1:
                return activations 
            else:
                base_norm = activations.norm(dim=2)
                itv_norm = itv_W_batch.norm(dim=2).to(activations.device)                
                alpha_itv = (base_norm / itv_norm)
                alpha_itv = torch.mul(alpha_itv, itv_str_tensor.to(activations.device))
                activations = activations + (itv_W_batch.to(activations.device) * alpha_itv[:, :, None]) 
                return activations 
        
        self.model.add_hook(self.cfg.hook_name, modify_activations)
        tokens = self.tokenizer.apply_chat_template(input_prompt_batch, tokenize=True, add_generation_prompt=True, return_tensors="pt", padding=True).to(self.token_device)
        output_text = self.model.generate(tokens, max_new_tokens=max_tokens, temperature=self.temperature, do_sample=False, verbose=False)[:, tokens.shape[1]:]
        self.model.remove_all_hook_fns()

        # convert the output tokens to text
        output_text = self.tokenizer.batch_decode(output_text, skip_special_tokens=True)

        # clean up memory
        gc.collect()
        torch.cuda.empty_cache()

        return output_text[0]