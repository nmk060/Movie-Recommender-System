#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits= pd.read_csv('tmdb_5000_credits.csv')


# movies.head()

# In[62]:


dd


# In[4]:


movies =movies.merge(credits,on='title')


# In[5]:


movies['crew'] = movies


# In[6]:


#genres
#id(for poster of movies on website)
#keywords
#title(always in english)
#overview
#cast 
#crew

movies =movies[['genres','id','keywords','title','overview','cast','crew']]


# In[7]:


#preprocessing of last 5coloumns and merging them for tagline


# In[8]:


movies.isnull().sum()


# In[ ]:





# In[9]:


movies.shape


# In[10]:


#processing generes


# In[11]:


movies.iloc[0].genres


# In[12]:


movies.loc[0]


# In[13]:


movies= movies[['genres','id','keywords','title','overview','cast','crew']]


# In[ ]:





# In[14]:


#['Action','Adventure','Fantasy','SciFi']


# In[15]:


#helper fn


# In[16]:


import ast


# In[17]:


def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L    


# In[18]:


movies['genres']= movies['genres'].apply(convert)
movies['genres']


# In[ ]:





# In[19]:


movies['keywords']= movies['keywords'].apply(convert)


# In[20]:


movies['keywords']


# In[21]:


movies.head()


# In[ ]:





# In[22]:


def convert3(obj):
    L=[]
    counter =0
    for i in ast.literal_eval(obj):
        if counter !=3:
         L.append(i['name'])
         counter =counter +1
        else:
            break
    return L


# In[23]:


movies['cast']=movies['cast'].apply(convert3)


# In[24]:


movies['crew'][0]


# In[25]:


def convertdic(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
    return L 


# In[26]:


movies['crew']=movies['crew'].apply(convertdic)


# In[27]:


movies.head()


# In[28]:


movies


# In[ ]:





# In[ ]:





# In[29]:


movies.isnull().sum()


# In[30]:


movies = movies[movies['overview'].notnull()]
movies.shape


# In[31]:


movies['overview']= movies['overview'].apply(lambda x: x.split())


# 
# movies['overview']= movies['overview'].apply(lambda x: x.split())

# In[32]:


movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])


# In[ ]:





# In[33]:


movies.head(1)


# In[ ]:





# In[34]:


movies['tags'] =  movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew'] 


# In[35]:


new_df  = movies[['id','title','tags']]


# In[36]:


new_df.head()


# In[ ]:





# In[37]:


new_df['tags'] =new_df['tags'].apply(lambda x: " ".join(x))


# In[38]:


new_df['tags'][0]


# In[ ]:





# In[39]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[40]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
new_df.shape


# In[41]:


new_df.head()


# In[42]:


vector = cv.fit_transform(new_df['tags']).toarray()


# In[43]:


vector.shape


# In[44]:


movies.shape


# In[45]:


vector


# In[46]:


import nltk


# In[47]:


from nltk.stem.porter import PorterStemmer


# In[48]:


ps = PorterStemmer()


# In[49]:


def stem(text):
    y = []
    
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)    


# In[50]:


new_df['tags']=new_df['tags'].apply(stem)


# In[51]:


from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vector)


# In[52]:


cosine_similarity(vector)


# In[53]:


cosine_similarity(vector).shape


# In[54]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index] 
    movies_list =sorted(list(enumerate(similarity[movie_index])),reverse=True,key = lambda x: x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
        


# In[55]:


recommend('Avatar')


# In[56]:


import pickle


# In[60]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[59]:


new_df.to_dict()


# In[61]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




