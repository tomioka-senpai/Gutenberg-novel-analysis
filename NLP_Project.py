#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import nltk
import string
import matplotlib.pyplot as plt


# In[2]:


# Function which returns string containing contents of .txt file at file_path
def txt_file_to_string(file_path):  
    with open(file_path, "r", encoding ="utf8") as curr:
        text = curr.read()
        text = text.replace("\n", " ").replace("\r", " ")
    return text


# In[3]:


# Book 1 is Pride and Prejudice stored in string T1
book_one_path = "E:/Pride and Prejudice.txt"
T1 = txt_file_to_string(book_one_path)
T1


# In[4]:


# Book 2 is The Adventures of Sherlock Holmes stored in string T2
book_two_path = "E:/The Adventures of Sherlock Holmes.txt"
T2 = txt_file_to_string(book_two_path)
T2


# In[5]:


#Function to do basic pre-processing on the text
def pre_process(str):
    
    # Converting entire string to lowercase
    str = str.lower()

    # Removing all punctuations by replacing everyhting other than whitespace characters, a-z, A-Z, 0-9 and '_' by empty string 
    # followed by replacing '_' by empty string
    str = re.sub(r"[^\w\s]", "", str)
    str = re.sub(r"_", "", str)

    #Removing chapter number headings if any
    str = re.sub(r"chapter [0-9]{1,3}", "", str)

    # Replacing whitespace characters by space
    str = re.sub(r"[\s]", " ", str)
    
    return str


# In[6]:


# Pre-processing T1
T1 = pre_process(T1)
T1


# In[7]:


# Tokenizing T1
from nltk.tokenize import word_tokenize
Tokenized_T1 = word_tokenize(T1)
print(Tokenized_T1)


# In[8]:


# Pre-processing T2
T2 = pre_process(T2)
T2


# In[9]:


# Tokenizing T2
Tokenized_T2 = word_tokenize(T2)
print(Tokenized_T2)


# In[10]:


# Finding frequency of tokens in T1
from nltk.probability import FreqDist
T1_frequency_distribution = FreqDist(Tokenized_T1)
T1_frequency_distribution


# In[11]:


# Finding frequency of tokens in T2
T2_frequency_distribution = FreqDist(Tokenized_T2)
T2_frequency_distribution


# In[12]:


from matplotlib.pyplot import figure
figure(figsize=(15, 15), dpi=80)

freq_graph = T1_frequency_distribution.plot(40, title = "A")
#labelling left
# type(freq_graph)
# font = {'family':'serif','color':'darkred','size':25}
# figure.set_xlabel("Word Length", fontdict = font)
# freq_graph.set_ylabel("Frequency", fontdict = font)
# freq_graph.show()


# In[13]:


figure(figsize=(15, 15), dpi=80)

freq_graph = T2_frequency_distribution.plot(40)
#labelling left


# In[16]:


import sys
print(sys.executable)


# In[26]:


anaconda3 -m pip install wordcloud


# In[14]:


from collections import Counter
dictionary = Counter(T1_frequency_distribution)
from wordcloud import WordCloud

cloud = WordCloud(max_font_size=60, max_words=80, background_color="white").generate_from_frequencies(dictionary)
plt.figure(figsize=(20,20))
plt.imshow(cloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[15]:


dictionary = Counter(T2_frequency_distribution)
cloud = WordCloud(max_font_size=60, max_words=80, background_color="white").generate_from_frequencies(dictionary)
plt.figure(figsize=(20,20))
plt.imshow(cloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[16]:


from nltk.corpus import stopwords
remove_these = set(stopwords.words('english'))
Cleaned_T1 = [w for w in Tokenized_T1 if not w in remove_these]
Cleaned_T1


# In[17]:


T1_frequency_distribution = FreqDist(Cleaned_T1)
T1_frequency_distribution


# In[18]:


dictionary = Counter(T1_frequency_distribution)
cloud = WordCloud(max_font_size=60, max_words=80, background_color="white").generate_from_frequencies(dictionary)
plt.figure(figsize=(20,20))
plt.imshow(cloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[19]:


remove_these = set(stopwords.words('english'))
Cleaned_T2 = [w for w in Tokenized_T2 if not w in remove_these]
Cleaned_T2


# In[20]:


T2_frequency_distribution = FreqDist(Cleaned_T2)
T2_frequency_distribution


# In[21]:


dictionary = Counter(T2_frequency_distribution)
cloud = WordCloud(max_font_size=60, max_words=80, background_color="white").generate_from_frequencies(dictionary)
plt.figure(figsize=(20,20))
plt.imshow(cloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[22]:



length_frequency_T1 = {}
for i, j in T1_frequency_distribution.items():
    x = len(i)
    if(x in length_frequency_T1):
        length_frequency_T1[x] += j
    else:
        length_frequency_T1[x] = j
length_frequency_T1


# In[23]:


plt.figure(figsize=(20,20))
plt.bar(range(len(length_frequency_T1)), list(length_frequency_T1.values()), align='center')
plt.xticks(range(len(length_frequency_T1)), list(length_frequency_T1.keys()))
# ax.bar_label(p1, label_type='center')
# ax.set_title("Amount Frequency")
font = {'family':'serif','color':'darkred','size':25}
plt.title("Relationship between Word Length and Frequency", fontdict = font, loc = "center")
plt.xlabel("Word Length", fontdict = font)
plt.ylabel("Frequency", fontdict = font)
plt.show()
# sort keys


# In[24]:



length_frequency_T2 = {}
for i, j in T2_frequency_distribution.items():
    x = len(i)
    if(x in length_frequency_T2):
        length_frequency_T2[x] += j
    else:
        length_frequency_T2[x] = j
length_frequency_T2


# In[25]:


plt.figure(figsize=(20,20))
plt.bar(range(len(length_frequency_T2)), list(length_frequency_T2.values()), align='center')
plt.xticks(range(len(length_frequency_T2)), list(length_frequency_T2.keys()))
# ax.bar_label(p1, label_type='center')
# ax.set_title("Amount Frequency")
font = {'family':'serif','color':'darkred','size':25}
plt.title("Relationship between Word Length and Frequency", fontdict = font, loc = "center")
plt.xlabel("Word Length", fontdict = font)
plt.ylabel("Frequency", fontdict = font)
plt.show()


# In[ ]:




