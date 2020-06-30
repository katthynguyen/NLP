import re 
import nltk
import scrapy
import numpy as np
import urllib.request
from pathlib import Path
from scipy import sparse
from scipy import spatial
from heapq import nlargest
from bs4 import BeautifulSoup
from string import punctuation
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from collections import defaultdict
from sklearn import neighbors, datasets
from nltk.stem import WordNetLemmatizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.summarization.bm25 import get_bm25_weights
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score

my_stopwords = set(stopwords.words('english') + list(punctuation)+ list("0123456789"))

#loại bỏ thẻ html
def Clean_html(request):
    soup= BeautifulSoup(request, "html.parser")    
    text = soup.get_text(strip=True)
    return text

#Xóa bỏ ký tự đặc biệt
def remove_special_character(text):
    strings = []
    string = re.sub('[^\w\s\.]',' ',text)    
    string = re.sub('\s+',' ',string)    
    string = string.strip()
    strings.append(string)
    return strings

# tách câu.
def separate_sentence(string):
    tokens = [t for t in string.split('.')]
    freq = nltk.FreqDist(tokens)
    sent = [key for key,val in freq.items()]
    return sent


# tách từ - loại bỏ stopword
def token_word(sentences): 
    words = word_tokenize(sentences)
    words = [w.lower() for w in words ]
    words = [word for word in words if word not in my_stopwords]    
    return words

# stemming đưa về động từ và tính từ nguyên mẫu
def stemming_word(words):
    lemmatizer = WordNetLemmatizer()
    words1 = [lemmatizer.lemmatize(word, pos = "a") for word in words]
    words3 = [lemmatizer.lemmatize(word, pos = "n") for word in words1]
    words4 = [lemmatizer.lemmatize(word, pos = "v") for word in words3]
    return words4

# tần xuất hiện các từ trong tài liệu
def Count_frequently(stem):    
    return Counter(stem)

# BAG OF WORDS
def BagOfWords(string):
    print("\n---Bag of words---\n")
    result = CountVectorizer()
    dense = result.fit_transform(string).todense()
    print("\n---Vocabulary---\n")
    print(result.vocabulary_)
    return dense 

# Ghi ra file độ đo Bag of words
def convertToList_BOW(string):
    listBag = []
    result = CountVectorizer() 
    Bag = result.fit_transform(string).todense()     
    for item in Bag:
        # np.savetxt('BOW.txt',item,fmt='%.0f')
        listBag.append(item.tolist())
    return listBag

# Độ đo Tf-idf - xuất độ đo ra màn hình
def Tf_IDF(string):    
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, stop_words='english')
    tf_idf_matrix = tf.fit_transform(string)
    feature_names = tf.get_feature_names()
    dense = tf_idf_matrix.todense()
    print("\n---feature names---\n")
    print('\n'.join(feature_names))
    print("\n---TF_IDF---\n")
    return dense
    
# Ghi  ra Độ đo Tf-idf - xuất ra độ đo
def convertToList_TF_IDF(string):
    list_TF_IDF = []
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, stop_words='english')
    tf_idf_matrix = tf.fit_transform(string)
    dense = tf_idf_matrix.todense()
    for item in dense:
        # np.savetxt('TF-IDF.txt',item,fmt='%.20f')
        list_TF_IDF.append(item.tolist())
    return list_TF_IDF

# độ đo tương quan BOW dùng cosine 
def CosineSimilar_BOW(string):    
    listcosine = []
    list_dense = convertToList_BOW(string)
    matrix_cosine=np.zeros(shape=(len(list_dense),len(list_dense)))
    irow =0
    for irow in range(len(list_dense)): 
        icol =0
        for icol in range(len(list_dense)):
            result = 1 - spatial.distance.cosine(list_dense[irow],list_dense[icol])
            matrix_cosine[irow][icol]=result;           
    listcosine=matrix_cosine.tolist()
    return listcosine

# độ đo tương quan TF_IDF dùng cosine             
def CosineSimilar_TF_IDF(string):    
    listcosine = []
    list_dense = convertToList_TF_IDF(string)
    matrix_cosine=np.zeros(shape=(len(list_dense),len(list_dense)))
    irow =0
    for irow in range(len(list_dense)): 
        icol =0
        for icol in range(len(list_dense)):
            result = 1 - spatial.distance.cosine(list_dense[irow],list_dense[icol])
            matrix_cosine[irow][icol]=result
    listcosine=matrix_cosine.tolist()
    return listcosine

def menu():
    print("\n\tTHUẬT TOÁN \n\t1. Bag of words.\n\t2. TF_IDF\n\t0. Thoát Chương trình.")
    print("\n\tYour choice: ")
  
