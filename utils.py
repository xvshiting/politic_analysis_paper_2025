import os 
import pandas as pd
import seaborn as sns
import spacy
import numpy as np
from collections import Counter
import os
from tqdm import notebook
import tqdm 

nlp = spacy.load("zh_core_web_md")
def init_dir(dir_path):
    if os.path.exists(dir_path):
        print("dir exists!")
    else:
        os.makedirs(dir_path)

def clean_text(text):
    #delete \u3000
    text = text.replace("\u3000","")
    return text

def load_stop_words(stop_word_path = "./stopwords.txt"):
    with open(stop_word_path) as f:
        lines = f.readlines()
    stop_word_list = []
    for line in lines:
        if line.strip() !="":
            stop_word_list.append(line.strip())
    return set(stop_word_list)

def get_paragraphs(text_list,
                   date_list,
                   province_list):
    paragraphs = []
    assert len(text_list)==len(date_list)==len(province_list)
    ind = 0
    for text, d, pro in zip(text_list, date_list, province_list):
        # print(pro)
        try:
            d = d.split("/")[0] #year
            d = int(d)
        except :
            d = 9999
        paras = text.split("\n")
        for p in paras:
            p = p.strip()
            if p.strip() and len(p)>15:
                paragraphs.append([p.strip(), int(d), ind, pro[:2]])
        ind += 1
    return paragraphs

def build_corpus(paragraphs, all_holder):
    removal = set(['ADV','PRON','CCONJ','PUNCT','PART','DET','ADP','SPACE', 'NUM', 'SYM'])
    ret = []
    for para in  tqdm.tqdm(paragraphs):
        # print(para)
        p, d, ind, _ = para
        doc = nlp(p)
        filted_p = []
        for token in doc:
            if not token.is_stop and token.is_alpha and str(token) not in all_holder.stop_word_set and len(str(token))>1:
                filted_p.append(str(token))
        ret.append(filted_p)
    return ret

def get_topic_dataframe(holder):
    rows = []
    topic_str_list = ["topic-{num}".format(num=num) for num in range(holder.lda_model.num_topics)]
    for ind, item in enumerate(holder.lda_model[holder.lda_corpus]):
        para_info = holder.paragraphs[ind]
        topic_score_list = [0]* holder.lda_model.num_topics
        for score in item:
            topic_score_list[score[0]] = score[1]
        tmp_dict = dict()
        for name, score in zip(topic_str_list, topic_score_list):
            tmp_dict[name] = score
        tmp_dict["text"] = para_info[0]
        tmp_dict["date"] = para_info[1]
        tmp_dict["doc_id"] = para_info[2]
        tmp_dict["province"] = para_info[3]
        rows.append(tmp_dict)
    df = pd.DataFrame(rows)
    return df

class Holder:
    pass