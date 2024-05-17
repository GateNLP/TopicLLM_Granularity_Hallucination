import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import re
import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine

def compute_mean_cosine_similarity(list1, list2):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    overall_similarities = []

    for item1, item2 in zip(list1, list2):
        topics1 = item1.split('\n')
        topics2 = item2.split('\n')
        
        item_similarities = []
        for topic1 in topics1:
            for topic2 in topics2:
                # Encode topics
                embeddings1 = model.encode([topic1], convert_to_tensor=True)
                embeddings2 = model.encode([topic2], convert_to_tensor=True)
                
                # Compute cosine similarity and convert tensor to CPU
                cosine_sim = cosine_similarity(embeddings1.cpu().numpy(), embeddings2.cpu().numpy())
                item_similarities.append(cosine_sim[0][0])
        
        # Mean similarity for the item
        item_mean_similarity = np.mean(item_similarities)
        overall_similarities.append(item_mean_similarity)
    
    # Overall mean similarity
    overall_mean_similarity = np.mean(overall_similarities)
    return overall_similarities, overall_mean_similarity


def extract_first_three_lines(input_string):
    # Split the string into lines
    lines = input_string.splitlines()
    
    # Slice to get the first three lines
    first_three_lines = lines[:3]
    
    # Join the first three lines into a single string
    return '\n'.join(first_three_lines)


def take_first_three(text):
    # Use regular expression to match lines starting with 1., 2., or 3.
    topics = re.findall(r'^\d+\.\s.*$', text, re.M)
    first_three_topics = topics[:3]  # Get the first three topics
    a = "\n".join(first_three_topics)
    return a
    # print("\n".join(first_three_topics))

def match_ori(data, test):
    #     full_names_list = [label_to_fullname[label] for label in test['title']]
    #     test['title'] = full_names_list
    #spsp = [i.replace(' ','\n') for i in test]
    #take_first_three(i).replace('1. ', '').replace('2. ', '').replace('3. ', '').split('\n')
    dx = pd.DataFrame({'topics': data.outputs.values, 'labels': test})
    clean = [extract_first_three_lines(i).replace('1. ', '').replace('2. ', '').replace('3. ', '').replace('\n\n','\n').replace('Topic 1: ', '').replace('Topic 2: ', '').replace('Topic 3: ', '') for i in dx.topics]
    ex_clean = [i if i[0] != '\n' else i.replace('\n','',1) for i in clean]
    dx['topics'] = ex_clean
    item_similarities, overall_mean_similarity = compute_mean_cosine_similarity(dx.topics, dx.labels)
    #print("Overall mean similarity:", overall_mean_similarity)
    return overall_mean_similarity
    


def get_bert_embedding(sentence):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    inputs = tokenizer(sentence, return_tensors="pt", max_length=128, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs['last_hidden_state'][:, 0, :].squeeze().numpy()

def similar_n(final_list):
    embeddings = [get_bert_embedding(topic) for topic in final_list]
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            similarity = cosine(embeddings[i], embeddings[j])
            similarities.append(similarity)
    avg_similarity = sum(similarities) / len(similarities)
    #print("Average semantic similarity:", avg_similarity)
    return avg_similarity    
#########    
import os

# Specify the directory containing the CSV files
#directory = './bills_results_all'
# Original list of newsgroup labels
newsgroup_labels = [
    'alt.atheism',
    'comp.graphics',
    'comp.os.ms-windows.misc',
    'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware',
    'comp.windows.x',
    'misc.forsale',
    'rec.autos',
    'rec.motorcycles',
    'rec.sport.baseball',
    'rec.sport.hockey',
    'sci.crypt',
    'sci.electronics',
    'sci.med',
    'sci.space',
    'soc.religion.christian',
    'talk.politics.guns',
    'talk.politics.mideast',
    'talk.politics.misc',
    'talk.religion.misc',
]

# Mapping of newsgroup labels to their full descriptive names
label_to_fullname = {
    'alt.atheism': 'Atheism',
    'comp.graphics': 'Computer Graphics',
    'comp.os.ms-windows.misc': 'Computer OS Microsoft Windows Miscellaneous',
    'comp.sys.ibm.pc.hardware': 'Computer Systems IBM PC Hardware',
    'comp.sys.mac.hardware': 'Computer Systems Mac Hardware',
    'comp.windows.x': 'Computer Windows',
    'misc.forsale': 'Sale',
    'rec.autos': 'Autos',
    'rec.motorcycles': 'Motorcycles',
    'rec.sport.baseball': 'Sport Baseball',
    'rec.sport.hockey': 'Sport Hockey',
    'sci.crypt': 'Science Cryptography',
    'sci.electronics': 'Science Electronics',
    'sci.med': 'Science Medicine',
    'sci.space': 'Science Space',
    'soc.religion.christian': 'Social Religion Christian',
    'talk.politics.guns': 'Politics Guns',
    'talk.politics.mideast': 'Politics Middle East',
    'talk.politics.misc': 'Politics Miscellaneous',
    'talk.religion.misc': 'Religion Miscellaneous',
}


directory = './all_results'
for filename in os.listdir(directory):
    if filename.endswith(".csv"):  
        filepath = os.path.join(directory, filename)
        print(f"Processing {filename}")
        #dx = pd.read_csv('bills1000.csv') # 
        dx = pd.read_csv('wiki1000.csv')
        data = pd.read_csv(filepath)
        data.columns=['outputs']
        all_topics=[]
        for i in data.outputs:
            for ii in extract_first_three_lines(i).replace('1. ', '').replace('2. ', '').replace('3. ', '').replace('Topic 1: ', '').replace('Topic 2: ', '').replace('Topic 3: ', '').split('\n'):
                all_topics.append(ii)
        #model = SentenceTransformer('all-MiniLM-L6-v2')
        from collections import Counter
        counter = Counter(all_topics)
        # Convert the counter to a DataFrame
        df = pd.DataFrame(counter.items(), columns=['Item', 'Frequency'])
        # Sort the DataFrame by frequency in descending order and take the top 10
        top_10 = df.sort_values(by='Frequency', ascending=False)
        print(top_10.Item[:10].values)
        print(len(top_10))
        #top_10.head() supercategory  topic
        a=match_ori(data, dx.supercategory.values)
        b=similar_n(top_10.Item[:10].values)
        print(len(top_10)), print(np.round(b, 3)), print(np.round(a, 3))
#a=match_ori(data, dx.topic.values)
#a=match_ori(data, dx.topic.values)
#b=similar_n(top_10.Item[:10].values)
#print(len(top_10)), print(np.round(b, 3)), print(np.round(a, 3))

