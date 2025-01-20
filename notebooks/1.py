datasetname='tabfact_cot'
import pandas as pd
import openai
import os
import json
from tqdm import tqdm
from tabqa.GptPrompter import *
from tabqa.GptCOTPrompter import *
from tabqa.GptCOTPrompter_BeamSeach import *
from tabqa.AutoReasoner import *
import dotenv
import ast  # 导入 ast 模块用于解析字符串为列表
import time
import numpy as np

config = dotenv.dotenv_values("../.env")
openai.api_key = config['OPENAI_API_KEY']
openai.base_url = config['GPT_BASE_URL']

dataset = pd.read_csv('../dataset/'+str(datasetname)+'/data/test.jsonl', sep='\t')

from dataset.tabfact_cot.load_data import load_tabfact_dataset
dataset_path='../dataset/'+str(datasetname)+'/data/test.jsonl'
raw2clean_path='../dataset/'+str(datasetname)+'/data/raw2clean.jsonl'
first_n=10
dataset = load_tabfact_dataset(dataset_path, raw2clean_path, first_n=first_n)
# if isinstance(dataset, list):
#     dataset = np.array(dataset)
dataset = pd.DataFrame(dataset)

# 在加载数据后，解析 table_text 列
# dataset['table_text'] = dataset['table_text'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

NNDemo = False
max_demo = 8
# gpt_model = 'gpt-4o-mini'
gpt_model = 'Qwen/Qwen2.5-7B-Instruct'

program = 'sql-py'
template = 'original-sql-py-no-intermediate'

def parallel_codex_func_formatv1(i):
    max_retry = 3
    attempt = 0  # 尝试次数
    while attempt < max_retry:
        try:
            # print('--------try: '+str(max_retry))
            # print(f"Index {i}: statement={dataset.iloc[i]['statement']}, table_text={dataset.iloc[i]['table_text']}, label={dataset.iloc[i]['label']}")
            codex_prompter = CodexAnswerCOTExecutor_HighTemperaturMajorityVote(
                f'prompt_template/{template}.json',
                dataset.iloc[i]['id'], 
                dataset.iloc[i]['statement'], 
                dataset.iloc[i]['table_text'],
                dataset.iloc[i]['label'],  
                base_path=f'../dataset/{datasetname}/',
                demo_file=f'few-shot-demo/WikiTQ-{program}.json',
            )
            codex_prompter.max_demo = max_demo
            codex_prompter.model = gpt_model
            codex_prompter._gen_gpt_prompt(NNDemo) #使用 table_formater 函数格式化 source_table_df 
            # print(str(max_retry)+'-4')
            codex_prompter._get_gpt_prediction_majority_vote(repeat_times=5)
            if codex_prompter.predicted_result is None:
                print(f"Error: No prediction result for index {i}.")
                continue  # 跳过当前索引
            # print(str(max_retry)+'-3')
            log = codex_prompter._log_dict()
            break
        except Exception as e:
            print(f"Error processing index {i}: {str(e)}")
            if "Request was rejected due to rate limiting" in str(e):
                print("Rate limit reached, waiting before retrying...")
                time.sleep(5)  # 等待5秒后重试
                attempt += 1  # 增加尝试次数
                if attempt >= max_retry:
                    print(f"Max retry attempts reached for index {i}.")
            else:
                log = {
                    'id': dataset.iloc[i]['id'],
                    'uncaught_err': str(e)
                }
                if "model's maximum context length" in str(e):
                    print("YES")
                    return log
    print("YES")
    return log
    
n_threads = 3
maxLimit = 2

from joblib import Parallel, delayed
output_result_file = f'../dataset/{datasetname}/results/11.json'
output_dir = os.path.dirname(output_result_file)  # 获取文件的目录部分
if not os.path.exists(output_dir):  # 如果目录不存在
    os.makedirs(output_dir)  # 创建目录
logs = Parallel(n_jobs=n_threads, require='sharedmem')(delayed(parallel_codex_func_formatv1)(i) for i in tqdm(range(min(maxLimit, dataset.shape[0]))))    

# 将 logs 中的 numpy.int64 转换为 int
logs = [{k: int(v) if isinstance(v, np.int64) else v for k, v in log.items()} for log in logs]

json.dump(logs, open(output_result_file, 'w'), indent=4)
print("1")

