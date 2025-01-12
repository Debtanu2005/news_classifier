#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf 
import pandas as pd
import seaborn as sns
import numpy as np


# In[3]:


print(tf.config.list_physical_devices())


# In[9]:


df = pd.read_csv(r"C:\Users\aspir\Downloads\train.csv")
df


# In[11]:


df["News with title"]="The title "+ df['Title']+" with news "+ df["Description"]


# In[13]:


df


# In[15]:


sns.histplot(df["Class Index"])


# In[16]:


import re


# In[19]:


def cleaning(text):
    text = text.lower()
    text= text.lower()
    text=re.sub(r'<.*?>', '', text)
    text= re.sub(r'\n', '', text)
    text=re.sub(r'[\x80-\xFF]', '', text)
    text = re.sub(r'[?/]', '', text)
    text= re.sub(r'[#/]', '', text)
    text= re.sub(r'[./:/]','', text)
    text= re.sub(r'[-]','', text)
    text= re.sub(r'[-/-]', '', text)
    text= re.sub(r'[,/]','', text)
    text= re.sub(r'[./]','', text)
    text= re.sub(r'[\']','', text)
    text= re.sub(r'%', ' percent', text)
    text= re.sub(r'["]','', text)
    text= re.sub(r'[$]', '', text)
    text= re.sub (r'[(ap)]', '', text)
    text= re.sub(r'[$]', '', text)
    text= re.sub(r'&', '', text)
    text =re.sub(r'[././.]', '', text)
    text= re.sub(r';', '', text)
    return text


# In[21]:


df['News with title']=df["News with title"].apply(cleaning)


# In[22]:


df['News with title'][16]


# In[25]:


from nltk.corpus import stopwords


# In[26]:


stopwords.words('english')


# In[29]:


def remove_stopword(text):
    new_text=[]
    for i in text.split():
        if i  in stopwords.words('english'):
            new_text.append('')
        else:
            new_text.append(i)
    return " ".join(new_text)       


# In[31]:


df['News with title']= df['News with title'].apply(remove_stopword)


# In[32]:


df["News with title"][0]


# In[33]:


import string


# In[34]:


def clean_punctuations(text):
    arr=[]
    for word in text.split():
        if word not in string.punctuation:
            arr.append(word)
    return " ".join(arr) 


# In[35]:


df['News with title']= df['News with title'].apply(clean_punctuations)


# In[36]:


from nltk.stem import WordNetLemmatizer


# In[37]:


def lem(text):
    ws = WordNetLemmatizer()
    new_text = []
    for word in text.split():
        new_text.append(ws.lemmatize(word))
    return " ".join(new_text)


# In[38]:


df['News with title']= df['News with title'].apply(lem)


# In[39]:


from keras.preprocessing.text import Tokenizer


# In[40]:


tok= Tokenizer(oov_token="<nothing>")


# In[41]:


tok.fit_on_texts(df["News with title"])


# In[42]:


tok.word_index


# In[43]:


df


# In[57]:


x= df['News with title']
y= df['Class Index']


# In[59]:


from sklearn.model_selection import train_test_split


# In[61]:


def text_length(text):
    return len(text.split())


# In[63]:


df['length']= df['News with title'].apply(text_length)


# In[64]:


max_text_len=max(df['length'])


# In[67]:


from keras.utils import pad_sequences


# In[69]:


x_train,x_test, y_train,y_test= train_test_split(x, y, test_size=0.2, random_state=42)


# In[71]:


x_tokenizer = Tokenizer(oov_token="<nothing>") 
x_tokenizer.fit_on_texts(list(x))

x_tr   =   x_tokenizer.texts_to_sequences(x_train) 
x_val =   x_tokenizer.texts_to_sequences(x_test)


x_train    =   pad_sequences(x_tr,  maxlen=200, padding='post')
x_test   =   pad_sequences(x_val, maxlen=200, padding='post')

x_voc   =  len(x_tokenizer.word_index) + 1


# In[72]:


x_train= np.stack(x_train )
x_test= np.stack(x_test)
y_train = np.stack(y_train)
y_test = np.stack(y_test)


# In[73]:


x_test


# In[74]:


y_train.shape
from tensorflow.keras.utils import to_categorical

# Example for one-hot encoding
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)


# In[79]:


from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping


# In[81]:


embedding_dim = 100
lstm_unit= 48


# In[ ]:





# In[84]:


y_train_one_hot = y_train_one_hot.astype('float32')
y_test_one_hot = y_test_one_hot.astype('float32')


# In[86]:


x_train= np.array(x_train)
x_train
max_text_len


# In[88]:


inputs = Input(shape=(200,))
emb = Embedding(x_voc, 100, input_length=200)(inputs)
lstm1 = LSTM(40, return_sequences=True)(emb)
lstm2=  LSTM(40, return_sequences=True)(lstm1)
pool = GlobalAveragePooling1D()(lstm2)
output = Dense(5, activation="softmax")(pool)

model = Model(inputs, output)
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[ ]:





# In[91]:


history=model.fit(x_train, y_train_one_hot,
          batch_size=32,
          validation_data=(x_test, y_test_one_hot),
          epochs=10)


# In[93]:


x_train_reshaped = x_test[2].reshape(1, -1)

prediction = model.predict(x_train_reshaped)


# In[95]:


prediction


# In[97]:


np.argmax(prediction)


# # Prediction with custom data

# In[100]:


#according to the data 
data= ["None", "world", "sports", "business",  "Science"]


# In[109]:


def pred(text):
    text=cleaning(text)
    text=clean_punctuations(text)
    text= remove_stopword(text)
    text= lem(text)
    x_tokenizer.fit_on_texts(text)
    text=x_tokenizer.texts_to_sequences(text)
    text= pad_sequences(text, maxlen=200, padding='post')
    predi=model.predict(text.reshape(1,-1))
    arg= np.argmax(predi)
    print(f"The news is {data[arg]} news")
    
    
    


# In[115]:


text= """named the men’s player of the year at the FIFA’s “The Best” awards. File | Photo Credit: AP

Real Madrid star Vinícius Júnior finally got his hands on a big global player award on Tuesday.

Vinícius was named the men’s player of the year at the FIFA’s “The Best” awards, where Barcelona playmaker Aitana Bonmati continued to clean up in the prizes for women’s soccer.

The 24-year-old Vinícius was so disappointed to lose out to Manchester City midfielder Rodri for the Ballon d’Or in October that he and his Madrid team snubbed the ceremony in Paris in protest.

This time Rodri ended up second to Vinícius by five points. The Brazil forward was at the FIFA ceremony to collect his award having travelled to Doha on Monday with Madrid for the Intercontinental Cup final against Pachuca.

“I don’t even know where to begin," Vinícius said in Portuguese. "It was so far away that it seemed impossible to get here. I was a kid who only played football barefoot on the streets of São Gonçalo, close to poverty and crime.

"Getting here is something very important to me. I’m doing it for many children who think that everything is impossible and who think they can’t get here.”

Vinícius echoed those sentiments in an Instagram post, where he took a thinly-disguised dig at presumably the Ballon d’Or voters — journalists from the top 100 countries in the FIFA rankings.

“Today I am writing to that boy who saw so many idols lift this trophy... your time has come,” he wrote. "Or rather, my time has come. The time to say ... yes, I am the best player in the world and I fought hard for it.

“They tried and still try to invalidate me, to diminish me. But they are not prepared. No one is going to tell me who I should fight for, how I should behave.”

Vinícius has been subjected to racist abuse in Spain and at one point earlier this year said he was “losing my desire to play” but added “I’ll keep fighting.”

Barcelona’s Aitana Bonmati won FIFA best women’s player of the year. File
Barcelona’s Aitana Bonmati won FIFA best women’s player of the year. File | Photo Credit: AP

Bonmati won the award for best women’s player of the year making it back-to-back prizes at FIFA's version of the older and more prestigious Ballon d’Or prize.

The 26-year-old Spain midfielder has won the Ballon d’Or for two straight years, and won the Spanish league, Spanish cup and Champions League with Barcelona in 2024.

“I am grateful to receive this award. As I always say, this is a team effort," Bonmati said. "It was a great year, very difficult to repeat. I am grateful to the people who help me to be better every day, from the club, to my teammates, who always help me to be better.”

FIFA said the award winners were decided by an “equally weighted voting system" by fans, the current captains and coaches of all national teams, and media representatives.

There was an 11-player shortlist for both awards, with the contenders selected based on their performances from Aug. 21, 2023, to Aug. 10. 2024.

Vinícius had the best season of his career, scoring 24 goals in 39 appearances for Madrid and helping the Spanish team to a record-extending 15th European Cup — and his second Champions League trophy.

He also netted in the final, becoming the youngest player to score in two Champions League finals.

Madrid teammate Jude Bellingham was third, ahead of Dani Carvajal and Lamine Yamal, with Lionel Messi — who had won the award the previous two years — sixth.

Madrid manager Carlo Ancelotti was named best men’s coach, and United States coach Emma Hayes took the women’s prize.

Hayes steered the USWNT to Olympic gold in Paris in August. Their shotstopper, Alyssa Naeher, was given the women’s goalkeeper award. Aston Villa and Argentina goalkeeper Emiliano Martínez won the men’s prize for the second straight year.

Alejandro Garnacho won the FIFA Puskás Award for the best goal, for his sensational overhead strike for Manchester United against Everton in November 2023.

Marta won the award that is named after her — the inaugural FIFA Marta Award — for her goal for Brazil against Jamaica in June.

Published - December 18, 2024 03:52 am IST

Related Topics
soccer

"""


# In[117]:


text= pred(text)


