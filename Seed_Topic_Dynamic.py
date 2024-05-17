from collections import Counter
import time
from tqdm import tqdm
import pandas as pd
from sys import exit
from transformers import AutoTokenizer, LlamaTokenizer
import transformers
import torch
billsss = pd.read_csv('bills1000.csv')
wikisss = pd.read_csv('wiki1000.csv')
ng20_spsss = pd.read_csv('20NG_special1000.csv')
ng20 = pd.read_csv('20NG1000.csv')

llama7b = 'meta-llama/Llama-2-7b-chat-hf'
llama13b = 'meta-llama/Llama-2-13b-chat-hf'
mistral_vanilla = 'mistralai/Mistral-7B-Instruct-v0.1'

bills='''<s>[INST] Based on the provided text, identify one to three general topics.
Topics should match the granularity of seed topics such as 'Health' and 'Education'. 
Each identified topic should consist of no more than three words. 
Please return only the topics. 
The desired output format:
Topic 1: xxx
Topic 2: xxx
Topic 3: xxx 
The Given text:
{list_of_text}[/INST]
Topics:
'''

bills_dpo='''<s>[INST] Based on the provided text, identify one to three general topics.
Topics should match the granularity of seed topics such as 'Health' and 'Education'. 
Please return only the topics. 
The desired output format:
1. xxx
2. xxx
3. xxx
The Given text:
{list_of_text}[/INST]
Topics:
'''


wiki='''<s>[INST] Based on the provided text, identify one to three general topics. 
Topics should match the granularity of seed topics such as 'History' and 'Music'.
Each identified topic should consist of no more than three words.  
Please return only the topics. 
The desired output format:
Topic 1: xxx
Topic 2: xxx
Topic 3: xxx 
The Given text:
{list_of_text}[/INST]
Topics:
'''

wiki_dpo='''<s>[INST] Based on the provided text, identify one to three general topics. 
Topics should match the granularity of seed topics such as 'History' and 'Music'.
Please return only the topics.
The desired output format:
1. xxx
2. xxx
3. xxx
The Given text:
{list_of_text}[/INST]
Topics:
'''


bills_dpo_dynamic='''<s>[INST] Based on the provided text, identify one to three general topics.
Topics should match the granularity of seed topics such as [{seedsss}]. 
Please return only the topics. 
The desired output format:
1. xxx
2. xxx
3. xxx
The Given text:
{list_of_text}[/INST]
Topics:
'''

wiki_dpo_dynamic='''<s>[INST] Based on the provided text, identify one to three general topics. 
Topics should match the granularity of seed topics such as [{seedsss}].
Please return only the topics.
The desired output format:
1. xxx
2. xxx
3. xxx
The Given text:
{list_of_text}[/INST]
'''
#'Computer' and 'Sports'.
prompt_20ng_dynamic='''<s>[INST] Based on the provided text, identify one to three general topics. 
Topics should match the granularity of seed topics such as [{seedsss}].
Please return only the topics.
The desired output format:
1. xxx
2. xxx
3. xxx
The Given text:
{list_of_text}[/INST]
'''
#'Baseball' and 'Hockey'
prompt_20ng_sp_dynamic='''<s>[INST] Based on the provided text, identify one to three specific topics related to sports and recreation. 
Topics should match the granularity of seed topics such as [{seedsss}].
Please return only the topics.
The desired output format:
1. xxx
2. xxx
3. xxx
The Given text:
{list_of_text}[/INST]
'''

#billsss = pd.read_csv('bills1000.csv')
#wikisss = pd.read_csv('wiki1000.csv')
#ng20_spsss = pd.read_csv('20NG_special1000.csv')
#ng20 = pd.read_csv('20NG1000.csv')

#llama7b = 'meta-llama/Llama-2-7b-chat-hf'
#llama13b = 'meta-llama/Llama-2-13b-chat-hf'
#mistral_vanilla = 'mistralai/Mistral-7B-Instruct-v0.1'
# new model path
mistral_dpo = [model_path]

pipeline = transformers.pipeline(
    "text-generation",
    model=mistral_dpo,
    torch_dtype=torch.float16,
    device_map="auto",
)
#sortedDictKey = list(dict(sorted(past_summarise_topics.items(), key=lambda item: item[1], reverse=True)).keys())
tokenizer = AutoTokenizer.from_pretrained(mistral_dpo, use_fast=False)

# build the dynamic topics function
def add_topics(response_text, past_summarise_topics):
    #response_text_tok = response_text.split(',')
    for each_topic in response_text:
        #each_topic_tok = each_topic_raw.split(':')
        #each_topic = each_topic_tok[-1].lower()
        if each_topic in past_summarise_topics:
            past_summarise_topics[each_topic] += 1
        else:
            past_summarise_topics[each_topic] = 1
    return past_summarise_topics
        
def gpt4396_dynamic_seed_topics(data, prompts, task):
    from nltk.stem import LancasterStemmer, WordNetLemmatizer
    lemmer = WordNetLemmatizer()
    output=[]
    past_summarise_topics = {'Computer': 0, 'Sports': 0}
    
    #batch_dynamic = []
   # batch_raw = []
    batch_clean = []
    j = 0
    for i in tqdm(data):
        sortedDictKey = list(dict(sorted(past_summarise_topics.items(), key=lambda item: item[1], reverse=True)).keys()) 
        dynamic_seeds = ', '.join(sortedDictKey[:10])
        if j > 10:
            print('--------dynamic_seeds---------')
            print(dynamic_seeds)
            print('--------dynamic_seeds---------')
        formatted_prompt = prompts.format(seedsss = dynamic_seeds, list_of_text = i)
        #print(formatted_prompt)
        content = formatted_prompt
        sequences = pipeline(
            content,
            # set do_sample = False; as this parameter enables decoding strategies such as multinomial sampling, beam-search multinomial sampling, Top-K sampling and Top-p sampling. (see: https://huggingface.co/docs/transformers/generation_strategies)
            do_sample=False,
            #top_k=10,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens = 50,
        )
        for seq in sequences:
            chat_response = seq['generated_text'][len(formatted_prompt):]
        print(chat_response)
        j += 1    
        print('jjjjjjjjjjjj')
        print(j)                      
        output.append(chat_response)
        df_output = pd.DataFrame(data={"raw_data": output})
        df_output.to_csv(f"dynamic_{task}.csv", sep = ',',index=False)
        chat_x = chat_response.replace('\n\n','\n')
        batch_raw = []
        for kk in chat_x.split('\n'):
            if kk[1:2] == '.':
                kk = kk.replace('Topic 1: ','').replace('Topic 2: ','').replace('Topic 3: ','').replace('\n','').replace('\r','').replace('Topic 1:','').replace('Topic 2:','').replace('Topic 3:','').replace('1. ','').replace('2. ','').replace('3. ','')
                if kk == 'Cars':
                    kk = 'Car'                                   
                batch_raw.append(kk)
                past_summarise_topics = add_topics(batch_raw, past_summarise_topics)
        if j > 50:
            print(past_summarise_topics)               
    return output#, batch_clean, batch_raw

output = gpt4396_dynamic_seed_topics(ng20.text, prompt_20ng, task = 'ng20')
from sys import exit
exit(0)

def gpt2200_fixed_seed_topics(data, prompts, task, model):
    from nltk.stem import LancasterStemmer, WordNetLemmatizer
    output=[]
    #seeds=[]
    #tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    for i in tqdm(data): 
        #a = random.sample(candidate, 2)
        formatted_prompt = prompts.format(list_of_text = i)
        #print(formatted_prompt)
        content = formatted_prompt
        sequences = pipeline(
            content,
            # set do_sample = False; as this parameter enables decoding strategies such as multinomial sampling, beam-search multinomial sampling, Top-K sampling and Top-p sampling. (see: https://huggingface.co/docs/transformers/generation_strategies)
            do_sample=False,
            #top_k=10,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            #max_length=4000,
            max_new_tokens = 35,
        )
        for seq in sequences:
            chat_response = seq['generated_text'][len(formatted_prompt):]
       # print(chat_response)
        output.append(chat_response)
        df_output = pd.DataFrame(data={task: output})
        df_output.to_csv(f"{task}.csv", sep = ',', index=False)     
    return output #, batch_clean, batch_raw

output = gpt2200_fixed_seed_topics(wikisss.clean_text, bad, 'wiki_L13_bad', llama13b)



