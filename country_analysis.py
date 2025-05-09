import pandas as pd
import seaborn as sns
import spacy
import numpy as np
from collections import Counter
import os
import wordcloud
from PIL import Image
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore
from gensim.models import CoherenceModel
from matplotlib import pyplot as plt
import pickle as pk
from tqdm import notebook
import pyLDAvis.gensim_models
pyLDAvis.enable_notebook()
from utils import *

all_holder = Holder
nlp = spacy.load("zh_core_web_md")
work_dir = "20241222/country"
init_dir(work_dir)

#Load Data
raw_xlsx_path = "./data/Policies Data.xlsx"
df_sd = pd.read_excel(raw_xlsx_path, sheet_name="Central Policies Table")
df_all = pd.concat([df_sd])

CONTENT_LABEL_STR = "Policy Content"
YEAR_LABEL_STR = "Release Time"
PROVINCE_LABEL_STR = "province"
WORD_STR = "word"
FREQUENCY_STR = "frequency"
text_list = df_all[CONTENT_LABEL_STR].to_list()
date_list = df_all[YEAR_LABEL_STR].to_list()
province_list = ["国家"]*len(text_list) #df_all[PROVINCE_LABEL_STR].to_list()
text_list = [ clean_text(text) for text in text_list]
stop_word_set = load_stop_words()
all_holder.stop_word_set = stop_word_set

#word frequency statistic
corpus_path = os.path.join(work_dir,"./all_corpus.pk")
words_freq_excel_path = os.path.join(work_dir,"./所有文档词频top200.xlsx")
if os.path.exists(corpus_path):
    paragraphs = get_paragraphs(text_list, date_list, province_list)
    corpus = pk.load(open(corpus_path, "rb"))
else:
    paragraphs = get_paragraphs(text_list, date_list, province_list)
    corpus = build_corpus(paragraphs, all_holder)
    pk.dump(corpus, open(corpus_path, "wb"))
all_holder.corpus = corpus
all_holder.paragraphs = paragraphs
if os.path.exists(words_freq_excel_path):
    words_freq_df = pd.read_excel(words_freq_excel_path)
else:
    mycounter = Counter()
    total_num = 0
    for c in corpus:
        total_num+= len(c)
        mycounter.update(c)
    words = []
    freqs = []
    for item in mycounter.most_common(200):
        words.append(item[0])
        freqs.append(item[1]/total_num)
    words_freq_df = pd.DataFrame()
    words_freq_df[WORD_STR] = words
    words_freq_df[FREQUENCY_STR] = freqs  
    words_freq_df.to_excel(words_freq_excel_path, index=False)
all_holder.words_freq_df = words_freq_df

#Topic analysis
dictionary = Dictionary(corpus)
dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=2000)
all_holder.dictionary = dictionary
lda_corpus = [dictionary.doc2bow(doc) for doc in corpus]
all_holder.lda_corpus = lda_corpus
lda_model = LdaMulticore(corpus=lda_corpus,
                         id2word=dictionary,
                         iterations=50,
                         num_topics=15,
                         workers = 4,
                         passes=50,
                         random_state=300,
                        alpha= "asymmetric",
                        gamma_threshold=0.002,
                        minimum_phi_value=0.02)
all_holder.lda_model = lda_model
topic_describ_path = os.path.join(work_dir,"topic_words_country.xlsx")
topic_describ_list = []
for i in range(10):
    words = lda_model.get_topic_terms(i, topn=15)
    word_str = " , ".join([dictionary.id2token[word[0]] for word in words])
    topic_id = i
    topic_describ_list.append({"topic_id":i, "words":word_str})
topic_words_df = pd.DataFrame(topic_describ_list)
topic_words_df.to_excel(topic_describ_path, index=False)
lda_display = pyLDAvis.gensim_models.prepare(lda_model, lda_corpus, dictionary)
pyLDAvis.save_html(lda_display,os.path.join(work_dir,"country_lda_topic.html"))
all_holder_path = os.path.join(work_dir,"all_holder.pk")
pk.dump(all_holder, open(all_holder_path,"wb"))

all_topic_df = get_topic_dataframe(all_holder)
all_topic_excel_path = os.path.join(work_dir, "all_topic_country.xlsx")
all_topic_df.to_excel(all_topic_excel_path, index = False)

#analysis topic trend through time
topic_str_list = ["topic-{num}".format(num=num) for num in range(all_holder.lda_model.num_topics)]
date_topic_score_list = []
df_list = []
for _, group in all_topic_df.groupby("date"):
    group = group.iloc[:,:-4]
    _df = pd.DataFrame({"topic":topic_str_list, "score":list(group.sum()/group.sum().sum())})
    _df["date"] = _
    df_list.append(_df)
plot_df = pd.concat(df_list)
all_plot_excel_path = os.path.join(work_dir,"./all_plot.xlsx")
plot_df.to_excel(all_plot_excel_path, index=False)
f, ax = plt.subplots(1, 1, figsize=[20,10])
sns.lineplot(ax =ax, x = "date", y="score", hue="topic", data=plot_df)

#word cloud
bg_path = "./chinese.png"
bg_mask = Image.open(bg_path)
wc = wordcloud.WordCloud(font_path = "./SimSun.ttf",
                         mask = np.array(bg_mask),
                         max_words=500,
                         max_font_size=400,
                         mode="RGBA",
                         scale=0.5,
                         # width=200,
                         # height=200
                         background_color=None
                        )
word_freq_dict = {item1:item2 for item1, item2 in zip(words_freq_df[WORD_STR].tolist(),words_freq_df[FREQUENCY_STR].tolist())}
wc.generate_from_frequencies(word_freq_dict)
wc_img = wc.to_image()
fig = plt.figure()
fig.set_size_inches(10,6)
# # plt.figure(figsize=(20,20))
plt.imshow(wc_img,)