import argparse
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from .utils import remove_repeated_sentences, remove_excessive_newlines
import pdb
import torch

class Experiment:
    """Base class to represent a single experiment"""
    def __init__(self, get_llm):
        """Initialize the experiment"""
        self.parser = argparse.ArgumentParser(description="Run experiments.")
        self.get_llm = get_llm
        self.add_arguments()

    def add_arguments(self):
        self.parser.add_argument('--engines', nargs='+', default=['random'], help='List of engines to use')
        self.parser.add_argument('--max-tokens', type=int, default=2, help='Maximum number of tokens')
        self.parser.add_argument('--temp', type=int, default=0, help='Temperature for LLM')
        self.parser.add_argument('--debug', action='store_true', default=False, help='Debug mode to print the output of the LLM at each step of the experiment')
        #TODO: Add flag --num_runs and --version number in each subclass

    def run(self):
        """Run the experiments"""
        args = self.parser.parse_args()
        engines = args.engines
        print(f"Running experiment with engines: {engines}, num_runs: {args.num_runs} and max_tokens: {args.max_tokens}")
        for engine in engines:
            print(f'Engine :------------------------------------ {engine} ------------------------------------')
            # Load LLM
            llm = self.get_llm(engine, args.temp, args.max_tokens) 
            llm.debug = args.debug

            # If already some run_files in path, start from the last one
            start_run = 0
            version = 'V' + self.parser.parse_args().version_number
            path = f'./data/{engine}{version if version != "V1" else ""}.csv'
            # if os.path.exists(path):
            #     start_run = pd.read_csv(path).run.max() + 1
            #     print(f"Starting from run {start_run}")
            
            for run in tqdm(range(start_run, args.num_runs)):
                # Run experiment
                df = self.run_single_experiment(llm)
                #Store data
                df['run'] = run
                self.save_results(df, engine)
    
    def run_batched(self, batch_num, llm, engine, label, str_str):
        """Run the experiments"""
        args = self.parser.parse_args()
        
        if label == "none":
            engine = f"str_{label}+{engine}"
        else:
            engine = f"str_{label}_{str_str}+{engine}"

        llm.act = label
        llm.str_str = str_str

        print(f'Engine :------------------------------------ {engine} ------------------------------------')

        # If already some run_files in path, start from the last one
        version = 'V' + self.parser.parse_args().version_number
        path = f'./data/{engine}{version if version != "V1" else ""}.csv'
        # if os.path.exists(path):
        #     start_run = pd.read_csv(path).run.max() + 1
        #     print(f"Starting from run {start_run}")

        texts = []
        run_indices = []
        datas= []
        format_answers = []
        for run in range(args.num_runs):
            cur_texts, data, cur_format_answers = self.get_single_script(llm)
            texts += cur_texts
            for i in range(len(cur_texts)):
                run_indices.append(run)
            datas.append(data)
            format_answers += cur_format_answers
        
        indices = np.arange(len(texts))
        batch_indices = np.split(indices, batch_num)

        generated_texts = []

        Q_, A_ = llm.Q_A

        for i in range(batch_num):
            start = batch_indices[i][0]
            end = batch_indices[i][-1] + 1
            texts_batch = texts[start:end]
            format_answers_batch = format_answers[start:end].copy()
            # llm.default_query_specifications = "(You must first output the results of step-by-step reasoning, and finally output the most likely answer. Please limit your entire answer to 100 words or less.)\n"
            llm.Q_A = Q_, A_

            # prompts = []
            # for j in range(len(texts_batch)):
            #     prompt = [{"role": 'user', 'content': texts_batch[j]}]
            generated_batch = llm.generate_batched(texts_batch, format_answers_batch, return_prob=False, add_generation_prompt=True)
            # generated_batch1 = llm.generate_batched(texts_batch, format_answers_batch, return_prob=True, add_generation_prompt=True)
            # choices = [str(k) for k in range(10)] 
            # choice_ids = [llm.tokenizer(c, add_special_tokens=False)["input_ids"][0] for c in choices]
            # choice_ids_tensor = torch.tensor(choice_ids)
            # selected_logits = generated_batch1[:, choice_ids_tensor]
            # pred_indices = torch.argmax(selected_logits, dim=1)
            # generated_batch1 = [choices[k] for k in pred_indices.tolist()]
            # for j in range(len(generated_batch1)):
            #     generated_batch1[j] = generated_batch1[j] + " "
            #     texts_batch[j] = texts_batch[j] + generated_batch1[j]
            # generated_batch2 = llm.generate_batched(texts_batch, format_answers_batch, return_prob=True, add_generation_prompt=False)
            # selected_logits = generated_batch2[:, choice_ids_tensor]
            # pred_indices = torch.argmax(selected_logits, dim=1)
            # generated_batch2 = [choices[k] for k in pred_indices.tolist()]
            # generated_batch = [generated_batch1[i] + generated_batch2[i] for i in range(len(generated_batch1))]
            
            generated_texts += generated_batch

        total_rand_cnt = 0
        total_cnt = 0
        
        for run in range(args.num_runs):
            cur_texts = []
            cur_data = datas[run]
            for i in range(len(texts)):
                if run_indices[i] == run:
                    cur_texts.append(generated_texts[i])
            
            df, cur_rand_cnt = self.run_single_experiment_from_generated(cur_texts, cur_data)
            total_rand_cnt += cur_rand_cnt
            total_cnt += len(cur_texts)
            df['run'] = run 
            self.save_results(df, engine)
        
        with open(f'./data/rand_ratio.txt', 'a') as f:
            f.write(f'{engine} {total_rand_cnt / total_cnt}\n')
            f.close()

    def run_batched_horizon(self, batch_size, llm, engine, label, str_str):

        args = self.parser.parse_args()

        if label == "none":
            engine = f"str_{label}+{engine}"
        else:
            engine = f"str_{label}_{str_str}+{engine}"

        llm.act = label
        llm.str_str = str_str
        # llm.default_query_specifications = "(Please only give me the machine ID without 'Machine' without any explanation based on your thought.)\n"
        llm.format_answer = "Machine "

        print(f'Engine :------------------------------------ {engine} ------------------------------------')

        # If already some run_files in path, start from the last one
        version = 'V' + self.parser.parse_args().version_number
        path = f'./data/{engine}{version if version != "V1" else ""}.csv'

        envs = []
        instructions = []
        histories = []
        trials_lefts = []
        questions = []
        actions_list = []
        num_trial_left = np.zeros(args.num_runs, dtype=int)
        ts = []
        for run in range(args.num_runs):
            instruction, history, trials_left, question, env = self.make_single_env(llm)
            envs.append(env)
            num_trial_left[run] = env.rewards.shape[0] - 4
            instructions.append(instruction)
            histories.append(history)
            trials_lefts.append(trials_left)
            questions.append(question)
            actions_list.append([None, None, None, None])
            ts.append(0)
        
        while(np.sum(num_trial_left) > 0):
            print(f"Number of trials left: {np.sum(num_trial_left)}")
            cur_indies_to_run = np.where(num_trial_left > 0)[0]
            if len(cur_indies_to_run) > batch_size:
                cur_indies_to_run = cur_indies_to_run[:batch_size]
            prompts = []
            for i in range(len(cur_indies_to_run)):
                run = cur_indies_to_run[i]
                prompt = instructions[run] + histories[run] + "\n" + trials_lefts[run] + questions[run]
                prompts.append(prompt)
            format_answers_batch = []
            for i in range(len(cur_indies_to_run)):
                format_answers_batch.append(llm.format_answer)
            generated_batch = llm.generate_batched(prompts, format_answers_batch, return_prob=False, add_generation_prompt=True)
            # choices = ["F", "J"]
            # choice_ids = [llm.tokenizer(c, add_special_tokens=False)["input_ids"][0] for c in choices]
            # choice_ids_tensor = torch.tensor(choice_ids)
            # selected_logits = generated_batch[:, choice_ids_tensor]
            # pred_indices = torch.argmax(selected_logits, dim=1)
            # generated_batch = [choices[i] for i in pred_indices.tolist()]
            for i in range(len(cur_indies_to_run)):
                run = cur_indies_to_run[i]
                actions, history, trials_left = self.run_single_experiment_from_generated(generated_batch[i], actions_list[run], histories[run], envs[run], ts[run])
                histories[run] = history
                trials_lefts[run] = trials_left
                actions_list[run] = actions
                num_trial_left[run] -= 1
                ts[run] += 1
            
        for run in range(args.num_runs):
            # Run experiment
            df = self.post_process_single_experiment(envs[run], actions_list[run])
            #Store data
            df['run'] = run
            self.save_results(df, engine)
        
        # pdb.set_trace()
    
    def run_batched_bandit(self, llm, engine, label, str_str):

        args = self.parser.parse_args()

        if label == "none":
            engine = f"str_{label}+{engine}"
        else:
            engine = f"str_{label}_{str_str}+{engine}"

        llm.act = label
        llm.str_str = str_str

        print(f'Engine :------------------------------------ {engine} ------------------------------------')

        # If already some run_files in path, start from the last one
        version = 'V' + self.parser.parse_args().version_number
        path = f'./data/{engine}{version if version != "V1" else ""}.csv'

        envs = []
        histories = []
        data_list = []
        index_to_run = np.ones(args.num_runs, dtype=bool)
        for run in range(args.num_runs):
            history, env, data = self.make_single_env()
            envs.append(env)
            histories.append(history)
            data_list.append(data)
        

        while(np.sum(index_to_run) > 0):
            print(f"Cur trial: {envs[0].t+1}")
            llm.format_answer = f"Machine " 
            llm.default_query_specifications = f"(Think carefully remembering that exploration of both machines is required for optimal rewards. Give the answer in the form ’Machine <your choice>’.)"
            prompt_letter = []
            format_answers_batch = []
            for i in range(len(index_to_run)):
                if index_to_run[i]:
                    prompt = self.make_prompt_letter(histories[i], llm, envs[i])
                    prompt_letter.append(prompt)
                    format_answers_batch.append(llm.format_answer)
            generated_batch = llm.generate_batched(prompt_letter, format_answers_batch, return_prob=False, add_generation_prompt=True)
            selected = []
            for s in generated_batch:
                if "F" in s:
                    selected.append("F")
                elif "J" in s:
                    selected.append("J")
                else:
                    random_choice = np.random.choice(["F", "J"])
                    selected.append(random_choice)

            # choices = ["F", "J"]
            # choice_ids = [llm.tokenizer(c, add_special_tokens=False)["input_ids"][0] for c in choices]
            # choice_ids_tensor = torch.tensor(choice_ids)
            # selected_logits = generated_batch[:, choice_ids_tensor]
            # pred_indices = torch.argmax(selected_logits, dim=1)
            # selected =[choices[i] for i in pred_indices.tolist()]
            llm.default_query_specifications = "Q: Give the answer in the form 'Machine <your choice> with confidence <your confidence>' (where your confidence of your choice being the best is given on a continuous "\
                "scale running from 0 representing "\
                f"\'this was a guess\' to 1 representing \'very certain\' should be given to two decimal places. If your estimate is 1, please answer as 0.99)."
            format_answers_batch = []
            for i in range(len(selected)):
                prompt_letter[i] = prompt_letter[i] + f"\n\nA: Machine {selected[i]}.\n\n" 
                format_answers_batch.append(f"Machine {selected[i]} with confidence 0.")
            generated_batch = llm.generate_batched(prompt_letter, format_answers_batch, return_prob=False, add_generation_prompt=True)
            batch_idx = 0
            for i in range(len(index_to_run)):
                if index_to_run[i]:
                    history, env, data = self.run_single_experiment_from_generated(generated_batch[batch_idx], selected[batch_idx], envs[i], data_list[i], histories[i])
                    histories[i] = history
                    envs[i] = env
                    data_list[i] = data
                    if env.done:
                        index_to_run[i] = False
                    batch_idx += 1
            
        for run in range(args.num_runs):
            # Run experiment
            df = pd.DataFrame(data_list[run], columns=['trial', 'block', 'block_trial', 'choice', 'reward', 'confidence', 'accurate'])
            #Store data
            df['run'] = run
            self.save_results(df, engine)
    
    def run_batched_instrumental(self, llm, engine, label, str_str):

        args = self.parser.parse_args()

        if label == "none":
            engine = f"str_{label}+{engine}"
        else:
            engine = f"str_{label}_{str_str}+{engine}"

        llm.act = label
        llm.str_str = str_str

        print(f'Engine :------------------------------------ {engine} ------------------------------------')

        # If already some run_files in path, start from the last one
        version = 'V' + self.parser.parse_args().version_number
        path = f'./data/{engine}{version if version != "V1" else ""}.csv'

        envs = []
        instructions_list = []
        histories = []
        trials_lefts = []
        questions = []
        current_machines = []
        data_list = []
        dones = np.zeros(args.num_runs, dtype=bool)
        for run in range(args.num_runs):
            env, instructions, history, trials_left, question, current_machine = self.make_single_env(llm)
            envs.append(env)
            instructions_list.append(instructions)
            histories.append(history)
            trials_lefts.append(trials_left)
            questions.append(question)
            current_machines.append(current_machine)
            data_list.append([])
        
        num_trials = envs[0].max_steps

        llm.format_answer = "Machine "
        llm.default_query_specifications = '(Give the answer in the form \"Machine <your choice>.\")'

        for t in range(num_trials):
            print(f"Cur trial: {t+1}")
            prompts = []
            format_answers_batch = []
            for i in range(args.num_runs):
                prompt = instructions_list[i] + trials_lefts[i] + "\n" + histories[i] + "\n"+ questions[i]
                prompts.append(prompt)
                format_answers_batch.append(llm.format_answer)
            generated_batch = llm.generate_batched(prompts, format_answers_batch, return_prob=False, add_generation_prompt=True)
            for i in range(args.num_runs):
                env, data, history, trials_left, current_machine, question, done = self.run_single_experiment_from_generated(generated_batch[i], envs[i], t, data_list[i], histories[i], current_machines[i], dones[i])
                envs[i] = env
                histories[i] = history
                trials_lefts[i] = trials_left
                current_machines[i] = current_machine
                questions[i] = question
                data_list[i] = data
                dones[i] = done
            
        for run in range(args.num_runs):
            # Run experiment
            df = pd.DataFrame(data_list[run], columns=['trial', 'task', 'mean0', 'mean1', 'reward0', 'reward1', 'choice', 'reward'])
            #Store data
            df['run'] = run
            self.save_results(df, engine)
    
    def run_batched_twostep(self, llm, engine, label, str_str):
        """Run the experiments"""
        args = self.parser.parse_args()

        if label == "none":
            engine = f"str_{label}+{engine}"
        else:
            engine = f"str_{label}_{str_str}+{engine}"

        llm.act = label
        llm.str_str = str_str

        print(f'Engine :------------------------------------ {engine} ------------------------------------')

        # If already some run_files in path, start from the last one
        version = 'V' + self.parser.parse_args().version_number
        path = f'./data/{engine}{version if version != "V1" else ""}.csv'
        # if os.path.exists(path):
        #     start_run = pd.read_csv(path).run.max() + 1
        #     print(f"Starting from run {start_run}")

        num_trials = args.num_trials
        num_runs = args.num_runs    

        Q_, A_ = llm.Q_A   

        previous_interactions = []
        datas = []
        reward_probs = []
        action_to_index = {"D": 0, "F": 1, "J": 2, "K": 3}

        for _ in range(num_runs):
            previous_interactions.append([])    
            datas.append([])
            reward_probs.append(np.random.uniform(0.25, 0.75, 4))  # Initialize reward probabilities for each alien
        
        for i in range(num_trials):
            total_texts = []
            format_answers_batch = []
            for j in range(num_runs):
                total_texts.append("You will travel to foreign planets in search of treasures.\n"\
            "When you visit a planet, you can choose an alien to trade with.\n"\
            "The chance of getting treasures from these aliens changes over time.\n"\
            "Your goal is to maximize the number of received treasures.\n\n")
                if len(previous_interactions[0]) > 0:
                    total_texts[j] += "Your previous space travels went as follows:\n"
                for count, interaction in enumerate(previous_interactions[j]):
                    days = " day" if (len(previous_interactions[j]) - count) == 1 else " days"
                    total_texts[j] += "- " + str(len(previous_interactions[j]) - count) + days + " ago, "
                    total_texts[j] += interaction
                
                total_texts[j] += f"\n{Q_} Do you want to take the spaceship to planet X or planet Y?\n"
                format_answers_batch.append("Planet ")
            # llm.default_query_specifications = "(Please only give me the planet name without 'Planet' without any explanation based on your thought.)\n"
            generated_batch = llm.generate_batched(total_texts, format_answers_batch, return_prob=False, add_generation_prompt=True)

            format_answers_batch = []
            states = []
            action1s = []
            for j in range(num_runs):
                action1 = generated_batch[j]
                if "X" in action1:
                    action1 = "X"
                elif "Y" in action1:
                    action1 = "Y"
                else:
                    action1 = np.random.choice(["X", "Y"])
                total_texts[j] += "A: Planet " + action1 + ".\n"
                action1s.append(action1)
                state = self.transition(action1)
                states.append(state)
                
                if state == "X":
                    feedback = "You arrive at planet " + state + ".\n"\
                    f"{Q_} Do you want to trade with alien D or F?\n"
                elif state == "Y":
                    feedback = "You arrive at planet " + state + ".\n"\
                    f"{Q_} Do you want to trade with alien J or K?\n"
                
                total_texts[j] +=  feedback
                format_answers_batch.append("Alien ")
            
            # llm.default_query_specifications = "(Please only give me the alien name without 'Alien' without any explanation based on your thought.)\n"
            generated_batch = llm.generate_batched(total_texts, format_answers_batch, return_prob=False, add_generation_prompt=True)

            for j in range(num_runs):
                action2 = generated_batch[j]
                if states[j] == "X":
                    if "D" in action2:
                        action2 = "D"
                    elif "F" in action2:
                        action2 = "F"
                    else:
                        action2 = np.random.choice(["D", "F"])
                elif states[j] == "Y":
                    if "J" in action2:
                        action2 = "J"
                    elif "K" in action2:
                        action2 = "K"
                    else:
                        action2 = np.random.choice(["J", "K"])
                treasure = np.random.binomial(1, reward_probs[j][action_to_index[action2]], 1)[0]

                row = [i, action1s[j], states[j], action2, treasure, reward_probs[0], reward_probs[1], reward_probs[2], reward_probs[3]]
                datas[j].append(row)

                reward_probs[j] += np.random.normal(0, 0.025, 4)
                reward_probs[j] = np.clip(reward_probs[j], 0.25, 0.75)
                if treasure:
                    feedback_item = "you boarded the spaceship to " + action1s[j] + ", arrived at planet " + states[j] + ", traded with alien " + action2 + ", and received treasures.\n"
                else:
                    feedback_item = "you boarded the spaceship to " + action1s[j] + ", arrived at planet " + states[j] + ", traded with alien " + action2 + ", and received junk.\n"
                previous_interactions[j].append(feedback_item)

        
        for run in range(num_runs):
            df = pd.DataFrame(datas[run], columns=['trial', 'action1', 'state', 'action2', 'reward', 'probsA', 'probsB', 'probsC', 'probsD'])
            df['run'] = run
            self.save_results(df, engine)
    
    def run_temporal(self, llm, engine, label, str_str):
        """Run the experiments"""
        args = self.parser.parse_args()

        if label == "none":
            engine = f"str_{label}+{engine}"
        else:
            engine = f"str_{label}_{str_str}+{engine}"

        llm.act = label
        llm.str_str = str_str

        print(f'Engine :------------------------------------ {engine} ------------------------------------')

        # If already some run_files in path, start from the last one
        version = 'V' + self.parser.parse_args().version_number
        path = f'./data/{engine}{version if version != "V1" else ""}.csv'
        # if os.path.exists(path):
        #     start_run = pd.read_csv(path).run.max() + 1
        #     print(f"Starting from run {start_run}")
        
        for run in range(args.num_runs):
            # Run experiment
            df = self.run_single_experiment(llm)
            #Store data
            df['run'] = run
            self.save_results(df, engine)

    def run_batched_bart(self, llm, engine, label, str_str):

        args = self.parser.parse_args()

        if label == "none":
            engine = f"str_{label}+{engine}"
        else:
            engine = f"str_{label}_{str_str}+{engine}"

        llm.act = label
        llm.str_str = str_str

        print(f'Engine :------------------------------------ {engine} ------------------------------------')

        # If already some run_files in path, start from the last one
        version = 'V' + self.parser.parse_args().version_number
        path = f'./data/{engine}{version if version != "V1" else ""}.csv'

        envs = []
        instructions_list = []
        histories = []
        trials_list = []
        trials_left = []
        pursuing_list = []
        no_pumps_list = []
        data_list = []
        observation_list = []
        actions_list = []
        balloon_idx_list = []
        dones = np.zeros(args.num_runs, dtype=bool)

        for run in range(args.num_runs):
            env, data, trials = self.make_single_env(llm)
            envs.append(env)
            data_list.append(data)
            trials_list.append(0)
            trials_left.append(trials * env._no_balloons)
            pursuing_list.append(True)
            no_pumps_list.append(0)
            alphabet = 'ABCDEFGHIJ'[:env._no_balloons]
            instructions = f"In this game you will encounter {env._no_balloons} different balloons labeled {', '.join(alphabet[:-1]) + ' and ' + alphabet[-1]}."\
            f" There will be a total of {trials} balloons for each type of balloon. Your goal is to accumulate as many points as possible without popping the balloon."\
            " You will be presented with a balloon and given the option to inflate it or not."\
            " Each inflation increases the balloon's size and potential points, but also carries a risk of the balloon popping. "\
            "Your task is to decide whether to inflate the balloon or not knowing that a successful inflation adds 1 point from that balloon."\
            " Once you decide to stop inflating the balloon, you can no longer earn points "\
            "from that balloon. If the balloon pops before you stop inflating (skip), you will lose all the points accumulated for that balloon. "\
            f"Your final score will be determined by the total number of points earned across all {trials*env._no_balloons} balloons. "\
            "Your goal is to maximize your final score. "
            history = ""
            instructions_list.append(instructions)
            histories.append(history)
            observation_list.append("")
            balloon_idx_list.append(env.randomly_sample())

        llm.format_answer = "Option "

        Q_, A_ = llm.Q_A  

        while(np.sum(dones) < args.num_runs):   
            trial_diff_list = [(trials_left[i] - trials_list[i]) for i in range(args.num_runs)]
            print(trial_diff_list)
            prompts = []
            format_answers_batch = []
            for i in range(args.num_runs):
                if dones[i]:
                    continue
                actions_list.append(np.random.choice([1, 2], size=2, replace=False))
                actions = np.random.choice([1, 2], size=2, replace=False)
                prompt = instructions_list[i] + histories[i] + observation_list[i] + f"{Q_} You are currently with balloon {trials_list[i]+1} which is a balloon of type {alphabet[balloon_idx_list[i]]}. "\
                    f"What do you do? (Option {actions[0]} for 'skip' or Option {actions[1]} for 'inflate') You must answer in the option number."
                if (trials_list[i] == 0) & (no_pumps_list[i] == 0):
                    histories[i] += f"\n\n You observed the following previously where the type of balloon is given in parenthesis:"
                prompts.append(prompt)
                format_answers_batch.append(llm.format_answer)
            generated_batch = llm.generate_batched(prompts, format_answers_batch, return_prob=False, add_generation_prompt=True)
            valid_idx = 0
            for i in range(args.num_runs):
                if dones[i]:
                    continue
                env, data, pursuing, no_pumps, observations, history = self.run_single_experiment_from_generated(generated_batch[valid_idx], envs[i], data_list[i], actions_list[i], balloon_idx_list[i], histories[i], trials_list[i], no_pumps_list[i], pursuing_list[i])
                envs[i] = env
                data_list[i] = data
                pursuing_list[i] = pursuing
                no_pumps_list[i] = no_pumps
                observation_list[i] = observations
                histories[i] = history  

                if not pursuing:
                    trials_list[i] += 1
                    if trials_list[i] == trials_left[i]:
                        dones[i] = True
                        continue
                    balloon_idx_list[i] = env.randomly_sample()
                    no_pumps_list[i] = 0
                    observation_list[i] = ""
                    pursuing_list[i] = True

                valid_idx += 1
            
        for run in range(args.num_runs):
            # Run experiment
            df = pd.DataFrame(data_list[run], columns=['trial', 'pumps', 'reward', 'max_pumps', 'exploded', 'balloon_trial', 'balloon_idx'])
            #Store data
            df['run'] = run
            self.save_results(df, engine)


    def save_results(self, df, engine):
            """Save the results in a CSV format"""
            # Path name
            version = 'V' + self.parser.parse_args().version_number
            folder = f"data{version if version != 'V1' else ''}"
            os.makedirs(folder, exist_ok=True)
            path = f'./{folder}/{engine}.csv'

            if os.path.exists(path):
                existing_df = pd.read_csv(path)
                existing_columns = existing_df.columns.tolist()

                # Reorder columns in df to match the existing CSV file's columns
                df = df[existing_columns]

            df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)

    def run_single_experiment(self):
        """Run a single experiment"""
        raise NotImplementedError()

class LLM:
    def __init__(self, llm_info):
        self.llm_info = llm_info
        self.Q_A = ('\n\nQ:', '\n\nA:')
        self.format_answer = "" #Default format of the answer is empty
        self.default_query_specifications = "" #Default query specifications is empty

    def generate(self, text, temp=None, max_tokens=None, return_prob=False, add_generation_prompt=True):
        """ Generate a response from the LLM. 'temp' and 'max_tokens' are made optional to be able to use the same function for all LLMs."""
        if temp is None:
            temp = self.temperature
        if max_tokens is None:
            max_tokens = self.max_tokens

        Q_, A_ = self.Q_A    
        # Set the default text for default queries, and alternative text if cot and cb can't be processed
        default_text = text + self.default_query_specifications
        default_text += f"{A_} {self.format_answer}"

        # Append the additional text from prompt engineering techniques
        if self.step_back:
            text += " First step back and think in the following two steps to answer this:"\
                "\nStep 1) Abstract the key concepts and principles relevant to this question in maximum 60 words."\
                "\nStep 2) Use the abstractions to reason through the question in maximum 60 words."\
                    f" Finally, give your final answer in the format 'Final answer: {self.format_answer}<your choice>'. It is very important that you always answer in the right format even if you have no idea or you believe there is not enough information."\
                    f"{A_} Step 1)\n"
        elif self.cot:
            text += f" You must first output the results of step-by-step reasoning, and finally output the most likely answer. Please limit your entire answer to 100 words or less. It is very important that you always answer in the right format even if you have no idea or you believe there is not enough information." 
        else:
            text = default_text
            # text += f" Please answer the question in the format 'Final answer: {self.format_answer}<your choice>'. It is very important that you always answer in the right format even if you have no idea or you believe there is not enough information."\
                # f"{A_} "
        
        prompt = [{"role": 'user', 'content': text}]
        
        llm_output = self._generate(prompt, temp, max_tokens, return_prob, add_generation_prompt)

        # pdb.set_trace()

        if self.cot:
            format_answer = self.format_answer
            prompt = [{"role": 'user', 'content': text}]
            prompt.append({"role": 'assistant', 'content': llm_output})
            prompt.append({"role": 'user', 'content': f"Based on your reasoning, please answer in the format 'Final answer: {format_answer}<your choice>' without reasoning or any additional text."})

            llm_output = self._generate(prompt, temp, 100, return_prob, add_generation_prompt)
    
        # pdb.set_trace()

        if return_prob:
            return llm_output
        
        #output processing
        processed_output = self.postprocess(llm_output, text, default_text, self.format_answer, temp, 100)

        # pdb.set_trace()

        return processed_output

    def generate_batched(self, text_batch, format_answers_batch, temp=None, max_tokens=None, return_prob=False, add_generation_prompt=True):
        """ Generate a response from the LLM. 'temp' and 'max_tokens' are made optional to be able to use the same function for all LLMs."""
        if temp is None:
            temp = self.temperature
        if max_tokens is None:
            max_tokens = self.max_tokens

        Q_, A_ = self.Q_A    
        # Set the default text for default queries, and alternative text if cot and cb can't be processed

        for i in range(len(text_batch)):
            text = text_batch[i] + self.default_query_specifications
            format_answer = format_answers_batch[i]

            default_text = text_batch[i] + self.default_query_specifications
            default_text += f"{A_} {format_answer}"

            # Append the additional text from prompt engineering techniques
            if self.step_back:
                text += " First step back and think in the following two steps to answer this:"\
                    "\nStep 1) Abstract the key concepts and principles relevant to this question in maximum 60 words."\
                    "\nStep 2) Use the abstractions to reason through the question in maximum 60 words."\
                        f" Finally, give your final answer in the format 'Final answer: {format_answer}<your choice>'. It is very important that you always answer in the right format even if you have no idea or you believe there is not enough information."\
                        f"{A_} Step 1)\n"
            elif self.cot:
                text += f" You must first output the results of step-by-step reasoning, and finally output the most likely answer. Please limit your entire answer to 100 words or less. It is very important that you always answer in the right format even if you have no idea or you believe there is not enough information." 
            else:
                text = default_text
                # text += f" Please answer the question in the format 'Final answer: {format_answer}<your choice>'. It is very important that you always answer in the right format even if you have no idea or you believe there is not enough information."\
                #     f"{A_} "
            
            text_batch[i] = text

        prompts = []
        for i in range(len(text_batch)):
            prompt = [{"role": 'user', 'content': text_batch[i]}]
            prompts.append(prompt)
        
        llm_output = self._generate_batched(prompts, temp, max_tokens, return_prob, add_generation_prompt)

        # pdb.set_trace()

        if self.cot:
            prompts = []
            for i in range(len(text_batch)):
                format_answer = format_answers_batch[i]
                prompt = [{"role": 'user', 'content': text_batch[i]}]
                prompt.append({"role": 'assistant', 'content': llm_output[i]})
                prompt.append({"role": 'user', 'content': f"Based on your reasoning, please answer in the format 'Final answer: {format_answer}<your choice>' without reasoning or any additional text."})
                prompts.append(prompt)
        
            llm_output = self._generate_batched(prompts, temp, 100, return_prob, add_generation_prompt)
        
        # pdb.set_trace()

        if return_prob:
            return llm_output

        outputs = []

        for i in range(len(text_batch)):

            default_text = text_batch[i] + self.default_query_specifications
            default_text += f"{A_} {self.format_answer}"

            #output processing
            processed_output = self.postprocess(llm_output[i], text_batch[i], default_text, format_answers_batch[i], temp, 100)

            outputs.append(processed_output)
        
        # pdb.set_trace()

        return outputs

    def _generate(self, text, temperature, max_tokens):
        raise NotImplementedError("Subclasses should implement this!")

    def postprocess(self, text, text_input, default_text, format_answer, temp, max_tokens):
        """
        Postprocess the output of the LLM. This is used to extract the answer from the LLM output in case of CoT and Step_back
        where the answer is given in the format 'Final answer: <answer> after a few sentences.
        Args:
            text (str): The output of the LLM
            text_input (str): The input of the LLM
            default_text (str): The default text is fed to LLM for the simpler standard query if none of the postprocessing techniques for prompt engineering outputs work.
        Returns:
            str: Extracting the answer from the output.
        """
        text = remove_excessive_newlines(text)
        if (self.step_back) or (self.cot):
            # Remove the bold format of the answer from some LLMs
            text = text.replace('**Final answer:**', 'Final answer:')
            text = text.replace('Final answer:\n\n', 'Final answer: ')
            text = text.replace('Final answer:\n', 'Final answer: ')

            # Get rid of multiple "Let's think step by step:" if more than 1 just keep first
            text = text.replace("Let's think step by step:", "")
            #Same idea with repeating Step 1)
            text = text.replace('**Step 1)', '')
            text = text.replace('Step 1)', '')



            # Split the text on "Final answer: "
            parts = text.split(f"Final answer: {format_answer}")

            # if answer not in right format, or no answer was given, then generate again
            if (len(parts) == 1) or (len(parts[1].split('.')[0].split('\n')[0].split("'")[0].replace(' ', '')) == 0):
                # import ipdb; ipdb.set_trace()
                print(f"Answer:{text} not in right format, generating again by appending the format at the end...")
                # First delete possible repetitions of the question which is sometimes the issue
                text = remove_repeated_sentences(text)
                
                # Append the generated answer to the text and generate again by hardcoding the format of the answer conditioned on the previously generated answer.
                # This is handy when the llm does not understand the limit of 100 words and gives a long answer without the final answer. We therefore do a two step generation.
                if self.Q_A[0] in text:
                    # Sometimes it does not give in the format 'Final answer: <answer>', so gives the answer and then generates itself a new question, and therefore here, we get rid of everything after the question.
                    text = text.split(self.Q_A[0])[0]

                # Generate the answer 
                reasoning_text = text.rsplit('\n', 1)[0] + f"\n\nFinal answer: {format_answer}"
                prompt = [{"role": 'user', 'content': text_input + reasoning_text}]
                new_text = self._generate(prompt, temp, max_tokens, return_prob=False, add_generation_prompt=True)
                # import ipdb; ipdb.set_trace()
                # print(new_text)

                # If the answer is not in the right format, ask again by:
                # 1. appending the reasoning text and the wrong answer to the input and generating again.
                # 2. intervening, saying that the reasoning was too long. 
                if len(new_text.split('.')[0].split('\n')[0].split("'")[0].replace(' ', '')) == 0: 
                    print(f"2-Answer still not in right format, something is wrong with the reasoning steps. Generating again by telling the LLM, the response was wrong...")
                    #1. appending the reasoning text and the wrong answer to the input and generating again.
                    wrong_output = new_text[:3]
                    re_ask_question = f"{self.Q_A[0]} I don't understand your final answer, the limit was a maximum of 1{'0' if  self.cot else '2'}0 words. Remember you are a reliable and helpful assistant. Just give me your answer without any more steps in the format 'Final answer: {format_answer}<your choice>'.{self.Q_A[1]} Sorry, it was a mistake. Final answer: {format_answer}"
                    prompt = [{"role": 'user', 'content': text_input + reasoning_text + wrong_output + re_ask_question}]
                    new_text = self._generate(prompt, temp, max_tokens, return_prob=False, add_generation_prompt=True)
                    if len(new_text.split('.')[0].split('\n')[0].split("'")[0].replace(' ', '')) == 0:
                        print(f"3-Answer still not in right format, something is wrong with the reasoning steps. Generating again by telling the LLM, the response was wrong but without appendix the right format this time because sometimes there is a confusion...")
                        #2. intervening, saying that the reasoning was too long without
                        prompt = [{"role": 'user', 'content': text_input + text.rsplit('\n', 1)[0] + re_ask_question}]
                        new_text = self._generate(prompt, temp, max_tokens, return_prob=False, add_generation_prompt=True)
                        
                        # If can't generate again, then return the default answer
                        if len(new_text.split('.')[0].split('\n')[0].split("'")[0].replace(' ', '')) == 0:
                            new_text = default_text
                            return self._generate(default_text, self.temperature, 2)[:5]
                parts = ['na', new_text]
            
            # Return the answer until '.' or '\' or '''
            text = parts[1].split('.')[0]
            text = text.split("\n")[0]
            text = text.split("'")[0]
            text = text[:5]
        return text
        

class RandomLLM(LLM):
    def __init__(self, llm_info):
        super().__init__(llm_info)
        print("Random agent is used!")

    def _generate(self, text, temp, max_tokens):
        return self.random_fct()
    
    def random_fct(self):
        raise NotImplementedError("Should set this random function depending on the task!")
    
class InteractiveLLM(LLM):
    def __init__(self, llm_info):
        super().__init__(llm_info)
        print("Interactive agent is used!")

    def _generate(self, text, temp, max_tokens):
        return input(f"{text}")
        
class StoringScores:
    """Analyze the results of the run"""
    def __init__(self):
        self.columns = ['engine','run','performance_score1','performance_score1_name','behaviour_score1','behaviour_score1_name']
        self.parser = argparse.ArgumentParser(description="Get behavioral and performance scores of experiments and store them.")
        self.add_arguments()

    def add_arguments(self):
        self.parser.add_argument('--engines', nargs='+', default=['all'], help='List of engines to use')
        self.parser.add_argument('--no_perf_score', nargs='+',  default=[], help='Number of performance scores')
        self.parser.add_argument('--version_number', type=str, default='1', help='Version number of the experiment. This is used to store the results of the experiment with have some changes seperately')
    def get_all_scores(self):
        args = self.parser.parse_args()
        engines = args.engines
        scores_csv_name = f"scores_data{'V' + args.version_number if args.version_number != '1' else ''}.csv"
        # self.columns += args.columns
        data_folder = f'data{"V"+args.version_number if args.version_number != "1" else ""}'
        if 'all' in args.engines:
            # if the csv file in format ./data/{engine}.csv exists then add to the list of engines
            engines = [os.path.splitext(file)[0] for file in os.listdir(data_folder)]
        # Check if scores_data exists, else, add the column names 
        storing_df =  pd.read_csv(scores_csv_name) if os.path.isfile(scores_csv_name) else pd.DataFrame(columns=self.columns)
             
        # Loop across engines and runs and store the scores
        for engine in tqdm(engines):
            print(f'Fitting for engine: {engine}-------------------------------------------')
            path = f'{data_folder}/{engine}.csv'
            full_df = pd.read_csv(path)
            no_runs = full_df['run'].max() + 1 
            for run in range(no_runs):
                df_run = full_df[full_df['run'] == run].reset_index(drop=True)
                storing_df = self.get_scores(df_run, storing_df, engine, run)
                storing_df.to_csv(scores_csv_name, index=False)




