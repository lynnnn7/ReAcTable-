import dotenv
from joblib import Parallel, delayed
from tabqa.GptCOTPrompter_BeamSeach import *
import os

config = dotenv.dotenv_values("../.env")
openai.api_key = config['OPENAI_API_KEY']
dataset = pd.read_csv('../dataset/WikiTableQuestions/data/pristine-unseen-tables.tsv', sep='\t')

NNDemo = False
max_demo = 5
gpt_model = 'Qwen/Qwen2.5-7B-Instruct'
program = 'sql-py'
template = 'original-sql-py-no-intermediate'

def parallel_func(i):
    max_retry = 3
    while max_retry>0:
        try:
            codex_prompter = CodexAnswerCOTExecutor_HighTemperaturMajorityVote(
                                                f'prompt_template/{template}.json',
                                                dataset.iloc[i]['id'], 
                                                dataset.iloc[i]['utterance'], 
                                                dataset.iloc[i]['context'], 
                                                dataset.iloc[i]['targetValue'],  
                                                base_path='../dataset/WikiTableQuestions/',
                                                demo_file=f'few-shot-demo/WikiTQ-{program}.json',
                                                )
            codex_prompter.max_demo = max_demo
            codex_prompter.model = gpt_model
            codex_prompter._gen_gpt_prompt(NNDemo)
            codex_prompter._get_gpt_prediction_majority_vote(repeat_times=5)
            log = codex_prompter._log_dict()
            break
        except Exception as e:
            log = {
                'id': dataset.iloc[i]['id'],
                'uncaught_err': str(e)
            }
            if "model's maximum context length" in str(e):
                return log
            max_retry -= 1
    return log
    
n_threads = 3
maxLimit = 5

output_result_file = f'../dataset/WikiTableQuestions/results/CodexAnswerCOTExecutor_HighTemperaturMajorityVote_{template}_{program}_NNDemo={NNDemo}_results_pristine-unseen-tables_limit{maxLimit}_model{gpt_model}.json'
os.makedirs(os.path.dirname(output_result_file), exist_ok=True)
logs = Parallel(
    n_jobs=n_threads, require='sharedmem'
    )(
        delayed(parallel_func)(i) for i in tqdm(range(min(maxLimit, dataset.shape[0])))
    )    
json.dump(logs, open(output_result_file, 'w'), indent=4)