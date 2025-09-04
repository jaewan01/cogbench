import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
# import gymnasium as gym
import envs
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))) #allows to import CogBench as a package
from CogBench.base_classes import Experiment
from CogBench.llm_utils.llms import get_llm
import pdb
import torch

class LearningToInferExpForLLM(Experiment):
    """
    This class represents the experiment in the Theory of Learning to Infer paper (Dasgupta, 2020) for an LLM. It extends the base Experiment class.

    The class is initialized with a function `get_llm` that is used to get the LLM generate function.

    The class has two main methods:
    - `__init__(self, get_llm)`: Initializes the class and adds the command line arguments.
    - `add_arguments_(self)`: Adds the command line arguments using argparse.
    - `run_single_experiment(self, llm)`: Runs an LLM on a single experiment of the Theory of Learning to Infer task.
    - `random_fct(self)`: Returns a random number between 1 and 99, with a leading zero if the number is less than 10.
    - `del_letters_at_end(self, text)`: Deletes letters from the end of a given text.

    The `run_single_experiment` method generates a series of tasks, each with a different set of conditions, and collects the LLM's responses. The responses are then stored in a DataFrame and returned.

    The `random_fct` method is used to generate random numbers for the tasks.

    The `del_letters_at_end` method is used to process the LLM's responses by removing any trailing letters.
    """
    def __init__(self):
        self.add_arguments_()

    def add_arguments_(self):
        # Add any additional arguments here
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--num_runs', type=int, default=100, help='Number of runs')
        self.parser.add_argument('--version_number', type=str, default='1', help='Version number of the experiment.')
        self.parser.add_argument('--engine', type=str, default="Llama-3.1-8b-Instruct", help='LLM engine to use')
        self.parser.add_argument('--gpus', type=str, default="0", help='GPU to use for the experiment')

    
    def run_single_experiment(self, llm):
        """Runs an LLM on single experiment of the Theory of Learning to Infer task.
        Args:
            llm (LLM): LLM object which will be used to generate the responses (choice and confidence).
        Returns:
            df (pd.DataFrame): Dataframe with the results of the experiment
        """
        Q_, A_ = llm.Q_A        
        llm.random_fct = self.random_fct
        llm_choice = lambda x: self.del_letters_at_end(llm.generate(x))
        data = []

        # Do one with informative likelihood and one with informative prior
        for informative_lh in [True, False]:
        # randomly toss a coin for which urn is left or right to avoid bias towards one urn
            if np.random.rand() < 0.5:
                left_urn = 'F'
                right_urn = 'J'
            else:
                left_urn = 'J'
                right_urn = 'F'
            
            if self.parser.parse_args().version_number == '1':
                #Wheel of fortune and urn
                instructions = f"You are participating in an experiment where you are provided with a wheel of fortune and two urns."\
                    f" The wheel of fortune contains 10 evenly sized sections labelled either {left_urn} or {right_urn}, corresponding to the urns {left_urn} and {right_urn}."\
                    " Another person will spin the wheel of fortune, select an urn based on the outcome of the spin, and then randomly pick a ball from the selected urn."\
                    f" Your goal is to give your best estimate of the probability of the urn being {left_urn} after observing the ball drawn from the urn." 
            elif self.parser.parse_args().version_number == '2':
                instructions = f"You are participating in an experiment where you are provided with a wheel of fortuneand two ten-sided dice, {left_urn} and {right_urn}."\
                    f" The coin is a biased coin where heads corresponds to dice {left_urn} and tail to dice {right_urn}."\
                    " Another person will toss the coin, select one of the two dice based on the outcome of the toss, and then randomly throw the selected dice which has only red or blue faces."\
                    f" Your goal is to give your best estimate of the probability of the dice being dice {left_urn} after observing the face of the dice thrown." 
            elif self.parser.parse_args().version_number == '3':
                # Gem Mining Operation
                instructions = f"You are participating in an experiment simulating a gem mining operation. There are two mines, mine {left_urn} and mine {right_urn},"\
                    " where valuable gems can be found. The mines contain a mix of real gems and fake stones. After extracting stones from a mine, they go through a sorting process to separate the real gems from the fakes."\
                    f" Your goal is to give your best estimate of the probability that a sorted gem came from mine {left_urn} after observing the result of the sorting process."

            env = envs.urns_exp.UrnTask(informative_lh)
            prior = env.left_prob
            lh = env.red_prob
            obs, urn = env.step(left_urn, right_urn)

            if self.parser.parse_args().version_number == '1':
                query = f"{Q_} The wheel of fortune contains {int(prior*10)} section{'s' if int(prior*10) > 1 else ''} labelled {left_urn} and {10 - int(prior*10)} section{'s' if 10 - int(prior*10)> 1 else ''} labelled {right_urn}."\
                    f" The urn {left_urn} contains ({int(lh*10)}, {10 - int(lh*10)}) and the urn {right_urn} contains ({10 - int(lh*10)}, { int(lh*10)}) red/blue balls."\
                    f" A {obs} ball was drawn. What is the probability that it was drawn from Urn {left_urn}? (Give your probability estimate on the scale from 0 to 1 rounded to two decimal places)"
                llm.format_answer = f"I estimate the probability of the {obs} ball to be drawn from the urn {left_urn} to be 0."
            elif self.parser.parse_args().version_number == '2':
                query = f"{Q_} The coin has a bias of {prior} towards heads (representing dice {left_urn})."\
                    f" The dice {left_urn} contains ({int(lh*10)}, {10 - int(lh*10)}) and the dice {right_urn} contains ({10 - int(lh*10)}, { int(lh*10)}) red/blue faces respectively."\
                    f" A {obs} face was observed. What is the probability that it was from the throw of dice {left_urn}? (Give your probability estimate on the scale from 0 to 1 rounded to two decimal places)"
                llm.format_answer = f"I estimate the probability of the {obs} face to be drawn from the dice {left_urn} to be 0."
            elif self.parser.parse_args().version_number == '3':
                gemobs = 'real' if obs == 'red' else 'fake'
                query = f"{Q_} The sorting process uses a large container, which contains {int(prior*10)} section{'s' if int(prior*10) > 1 else ''} labelled {left_urn} and {10 - int(prior*10)} section{'s' if 10 - int(prior*10)> 1 else ''} labelled {right_urn} representing the mines that the gems were extracted from."\
                    f" The mine {left_urn} contains ({int(lh*10)}, {10 - int(lh*10)}) real/fake gems and the mine {right_urn} contains ({10 - int(lh*10)}, { int(lh*10)}) real/fake gems."\
                    f" A {gemobs} gem was sorted. What is the probability that it was sorted from mine {left_urn}? (Give your probability estimate on the scale from 0 to 1 rounded to two decimal places)"
                llm.format_answer = f"I estimate the probability of the {gemobs} gem to be sorted from mine {left_urn} to be 0."
            #! Sometimes the following line is needed unhashed in this experiment because the LLM reformulates the question and therefore gives the answer in the format 0.XX whereas the code is made for an answer in the form XX only
            # llm.format_answer = llm.format_answer.replace(' 0.', '')
            
            prompt = instructions + query # add history if not very first trial
            pred = llm_choice(prompt)

            # Data storage
            # Process observation into boolean for storing
            if obs == 'red':
                obs = True
            elif obs == 'blue':
                obs = False
            else:
                raise ValueError(f"Invalid observation: '{obs}'")

            data.append([informative_lh, prior, lh, pred, obs])
        df = pd.DataFrame(data, columns=['informative_lh', 'prior', 'lh', 'left_pred', 'red_observation'])
        return df
    
    def get_single_script(self, llm):
        """Runs an LLM on single experiment of the Theory of Learning to Infer task.
        Args:
            llm (LLM): LLM object which will be used to generate the responses (choice and confidence).
        Returns:
            df (pd.DataFrame): Dataframe with the results of the experiment
        """
        Q_, A_ = llm.Q_A        
        llm.random_fct = self.random_fct
        llm_choice = lambda x: self.del_letters_at_end(llm.generate(x))
        data = []

        prompts = []

        format_answers = []

        # Do one with informative likelihood and one with informative prior
        for informative_lh in [True, False]:
        # randomly toss a coin for which urn is left or right to avoid bias towards one urn
            if np.random.rand() < 0.5:
                left_urn = 'F'
                right_urn = 'J'
            else:
                left_urn = 'J'
                right_urn = 'F'
            
            if self.parser.parse_args().version_number == '1':
                #Wheel of fortune and urn
                instructions = f"You are participating in an experiment where you are provided with a wheel of fortune and two urns."\
                    f" The wheel of fortune contains 10 evenly sized sections labelled either {left_urn} or {right_urn}, corresponding to the urns {left_urn} and {right_urn}."\
                    " Another person will spin the wheel of fortune, select an urn based on the outcome of the spin, and then randomly pick a ball from the selected urn."\
                    f" Your goal is to give your best estimate of the probability of the urn being {left_urn} after observing the ball drawn from the urn." 
            elif self.parser.parse_args().version_number == '2':
                instructions = f"You are participating in an experiment where you are provided with a wheel of fortuneand two ten-sided dice, {left_urn} and {right_urn}."\
                    f" The coin is a biased coin where heads corresponds to dice {left_urn} and tail to dice {right_urn}."\
                    " Another person will toss the coin, select one of the two dice based on the outcome of the toss, and then randomly throw the selected dice which has only red or blue faces."\
                    f" Your goal is to give your best estimate of the probability of the dice being dice {left_urn} after observing the face of the dice thrown." 
            elif self.parser.parse_args().version_number == '3':
                # Gem Mining Operation
                instructions = f"You are participating in an experiment simulating a gem mining operation. There are two mines, mine {left_urn} and mine {right_urn},"\
                    " where valuable gems can be found. The mines contain a mix of real gems and fake stones. After extracting stones from a mine, they go through a sorting process to separate the real gems from the fakes."\
                    f" Your goal is to give your best estimate of the probability that a sorted gem came from mine {left_urn} after observing the result of the sorting process."

            env = envs.urns_exp.UrnTask(informative_lh)
            prior = env.left_prob
            lh = env.red_prob
            obs, urn = env.step(left_urn, right_urn)

            if self.parser.parse_args().version_number == '1':
                query = f"{Q_} The wheel of fortune contains {int(prior*10)} section{'s' if int(prior*10) > 1 else ''} labelled {left_urn} and {10 - int(prior*10)} section{'s' if 10 - int(prior*10)> 1 else ''} labelled {right_urn}."\
                    f" The urn {left_urn} contains ({int(lh*10)}, {10 - int(lh*10)}) and the urn {right_urn} contains ({10 - int(lh*10)}, { int(lh*10)}) red/blue balls."\
                    f" A {obs} ball was drawn. What is the probability that it was drawn from Urn {left_urn}? (Give your probability estimate on the scale from 0 to 1 rounded to two decimal places. If your estimate is 1, please answer as 0.99)"
                format_answer = f"I estimate the probability of the {obs} ball to be drawn from the urn {left_urn} to be 0."
                # format_answer = f""
            elif self.parser.parse_args().version_number == '2':
                query = f"{Q_} The coin has a bias of {prior} towards heads (representing dice {left_urn})."\
                    f" The dice {left_urn} contains ({int(lh*10)}, {10 - int(lh*10)}) and the dice {right_urn} contains ({10 - int(lh*10)}, { int(lh*10)}) red/blue faces respectively."\
                    f" A {obs} face was observed. What is the probability that it was from the throw of dice {left_urn}? (Give your probability estimate on the scale from 0 to 1 rounded to two decimal places)"
                llm.format_answer = f"I estimate the probability of the {obs} face to be drawn from the dice {left_urn} to be 0."
            elif self.parser.parse_args().version_number == '3':
                gemobs = 'real' if obs == 'red' else 'fake'
                query = f"{Q_} The sorting process uses a large container, which contains {int(prior*10)} section{'s' if int(prior*10) > 1 else ''} labelled {left_urn} and {10 - int(prior*10)} section{'s' if 10 - int(prior*10)> 1 else ''} labelled {right_urn} representing the mines that the gems were extracted from."\
                    f" The mine {left_urn} contains ({int(lh*10)}, {10 - int(lh*10)}) real/fake gems and the mine {right_urn} contains ({10 - int(lh*10)}, { int(lh*10)}) real/fake gems."\
                    f" A {gemobs} gem was sorted. What is the probability that it was sorted from mine {left_urn}? (Give your probability estimate on the scale from 0 to 1 rounded to two decimal places)"
                llm.format_answer = f"I estimate the probability of the {gemobs} gem to be sorted from mine {left_urn} to be 0."
            #! Sometimes the following line is needed unhashed in this experiment because the LLM reformulates the question and therefore gives the answer in the format 0.XX whereas the code is made for an answer in the form XX only
            # llm.format_answer = llm.format_answer.replace(' 0.', '')
            
            prompt = instructions + query # add history if not very first trial
            prompts.append(prompt)

            data.append([informative_lh, prior, lh, obs])

            format_answers.append(format_answer)
            
        return prompts, data, format_answers
    
    def run_single_experiment_from_generated(self, cur_texts, data):
        """Runs an LLM on single experiment of the Theory of Learning to Infer task.
        Args:
            llm (LLM): LLM object which will be used to generate the responses (choice and confidence).
        Returns:
            df (pd.DataFrame): Dataframe with the results of the experiment
        """

        preds = []
        rand_cnt = 0
        for i in range(len(cur_texts)): 
            pred_, is_random = self.del_letters_at_end(cur_texts[i])
            preds.append(pred_)
            if is_random:
                rand_cnt += 1

        df = pd.DataFrame(data, columns=['informative_lh', 'prior', 'lh', 'red_observation'])
        df['left_pred'] = preds

        return df, rand_cnt

    def random_fct(self):
        """Returns a random number between 1 and 99, with a leading zero if the number is less than 10."""
        number = np.random.randint(1, 100)
        if number < 10:
            return "0.0" + str(number)
        else:
            return "0." + str(number)
        
    def del_letters_at_end(self, text):
        '''
        Args:
            text (str): text to delete letters from end
        Returns:
            text (str): text with letters deleted from end
        '''
        original_text = text
        if ',' in text:
            text = text.split(',')[0]
        text = text.replace('0.','').replace(',', '').replace('.', '').replace(' ', '').replace('\'', '').replace(':"', '').replace('?', '').replace('*', '') #The 0. is because for gemini we have to remove the 0. from the prompt and therefore the LLM will generate 0. at the beginning of the confidence
        if len(text) == 0:
            # print(f'{original_text} for prediction so storing random number')
            return self.random_fct(), True
            # If text is empty, have to choose what to do. For now hits debugger.
        while text[-1].isalpha():
            if len(text) > 1:
                text = text[:-1]
            else:
            # If text is empty, have to choose what to do. For now hits debugger.
                # print(f'{original_text} for prediction so storing random number')
                return self.random_fct(), True
                # import ipdb; ipdb.set_trace()
                # break
        try:
            if len(text) == 1:
                text = '0' + text
            return float('0.'+text), False
        except:
            # print(f'{original_text} for prediction so storing random number')
            return self.random_fct(), True

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument('--engine', type=str, default="Llama-3.1-8b-Instruct", help='LLM engine to use')
    parser.add_argument('--max-tokens', type=int, default=500, help='Maximum number of tokens')
    parser.add_argument('--temp', type=int, default=0.3, help='Temperature for LLM')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode to print the output of the LLM at each step of the experiment')
    parser.add_argument('--gpus', type=str, default="0", help='GPU to use for the experiment')
    args = parser.parse_args()

    llm = get_llm(args.engine, args.temp, args.max_tokens, args.gpus)
    llm.cot = True
    llm.debug = args.debug
    experiment = LearningToInferExpForLLM()
    np.random.seed(0)
    torch.manual_seed(0)
    experiment.run_batched(1, llm, args.engine, "none", 0.0)

    labels = ['depressed mood', 'low self-esteem', 'negativity bias', 'guilt', 'risk-aversion', 'self-destruction', 'manic mood', 'grandiosity', 'positivity bias', 'lack of remorse', 'risk-seeking', 'hostility']
    str_strs = [0.25]

    for label in labels:
        for str_str in str_strs:
            experiment = LearningToInferExpForLLM()
            np.random.seed(0)   
            torch.manual_seed(0)
            experiment.run_batched(1, llm, args.engine, label, str_str)

    labels = ['random 0', 'random 1', 'random 2', 'random 3', 'random 4', 'random 5', 'random 6', 'random 7', 'random 8', 'random 9']
    str_strs = [0.25]

    for label in labels:
        for str_str in str_strs:
            experiment = LearningToInferExpForLLM()
            np.random.seed(0)   
            torch.manual_seed(0)
            experiment.run_batched(1, llm, args.engine, label, str_str)
    