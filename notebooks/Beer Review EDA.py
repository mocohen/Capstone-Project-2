#!/usr/bin/env python
# coding: utf-8

# Data can be found here: https://www.kaggle.com/ehallmar/beers-breweries-and-beer-reviews

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


reviews = pd.read_csv('../data/reviews.csv')


# In[4]:


reviews.info()


# In[5]:


reviews.head(10)


# In[6]:


beers = pd.read_csv('../data/beers.csv')


# In[7]:


beers.info()


# In[8]:


beers.head()


# In[9]:


breweries = pd.read_csv('../data/breweries.csv')
breweries.info()


# In[10]:


breweries.head()


# In[11]:


# row in which value of 'Age' column is more than 30
has_text_review_mask = reviews.apply(lambda x: True if len(x['text'].strip()) > 0 else False , axis=1)

number_with_text_reviews = has_text_review_mask.sum()

print('Number of Rows in dataframe: ', len(reviews))
print('Number of Rows in dataframe with text reviews: ', number_with_text_reviews)
print('Percent of Rows in dataframe with text reviews: ', 100.0*number_with_text_reviews/len(reviews))


# ### Example Review

# In[12]:


reviews.text.iloc[0]


# ## Number of reviews for each beer

# In[13]:


# reviews_per_beer = reviews.groupby('beer_id').count()


# # In[14]:


# fig, ax = plt.subplots(1)

# sns.distplot(reviews_per_beer.username.values, kde=False,
#              bins=np.logspace(0,5,15),ax=ax)
# #reviews.groupby('beer_id').count().username.hist(ax=ax,bins=(0,1,10,100,1000, 10000))
# plt.xscale('log')
# plt.xlabel('Number of Reviews')
# plt.yscale('log')
# plt.title('How many reviews a typical beers receives')
# plt.ylabel('Number of Beers')
# plt.show()


# # In[15]:


# beer_reviews = beers.merge(reviews_per_beer.score, left_on='id', right_on='beer_id').rename(columns={'score':'n_reviews'})


# # In[16]:


# beer_reviews.head()


# # # Most rated beers

# # In[17]:


# beer_reviews.sort_values('n_reviews', ascending=False).head(10)


# # # average of ratings by type of beer
# # 

# # In[18]:


# beer_reviews.groupby('style').agg({'n_reviews':"mean",
#                                  'id':'count'}).sort_values('n_reviews', ascending=False).rename(columns={'id':'n_beers'})


# # In[19]:


# fig, ax = plt.subplots()

# sns.distplot(beer_reviews.groupby('style').sum().n_reviews.values, kde=False,
#              bins=np.logspace(0,5,15),ax=ax)
# #reviews.groupby('beer_id').count().username.hist(ax=ax,bins=(0,1,10,100,1000, 10000))
# plt.xscale('log')
# plt.xlabel('Number of Styles')
# plt.yscale('log')
# plt.title('How many reviews a style of beer generates')
# plt.ylabel('Number of Users')
# plt.show()


# # ## Number of reviews for each user

# # In[20]:


# fig, ax = plt.subplots()

# sns.distplot(reviews.groupby('username').count().text.values, kde=False,
#              bins=np.logspace(0,5,15),ax=ax)
# #reviews.groupby('beer_id').count().username.hist(ax=ax,bins=(0,1,10,100,1000, 10000))
# plt.xscale('log')
# plt.xlabel('Number of Reviews')
# plt.yscale('log')
# plt.title('How many reviews a typical user generates')
# plt.ylabel('Number of Users')
# plt.show()


# # ## Correlations for beer reviews

# # In[21]:


# average_reviews = reviews.dropna(axis=0).groupby('beer_id').mean()


# # In[22]:


# sns.pairplot(average_reviews)


# # In[23]:


# # g = sns.PairGrid(average_reviews)
# # g.map_diag(sns.kdeplot)
# # g.map_offdiag(sns.kdeplot, n_levels=6);


# # In[24]:


# average_reviews_user = reviews.dropna(axis=0).groupby('username').mean()


# # In[25]:


# sns.pairplot(average_reviews_user.drop('beer_id', axis=1))


# # In[26]:


# # g = sns.PairGrid(average_reviews_user.drop('beer_id', axis=1))
# # g.map_diag(sns.kdeplot)
# # g.map_offdiag(sns.kdeplot, n_levels=6);


# In[27]:


reviews.head(2)


# In[28]:


reviews_with_text = reviews[has_text_review_mask]
reviews_with_text.head()


# In[29]:


beers.head(2)


# In[30]:


reviews_with_beer_info = reviews_with_text.merge(beers, left_on='beer_id', right_on='id')
reviews_with_beer_info.head()


# In[32]:


all_text = " ".join(review for review in reviews_with_beer_info.text)


# In[ ]:


wordcloud = WordCloud().generate(all_text)
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.savefig('../images/all_words_cloud.png')


# In[ ]:


good_text = ' '.join(review for review in reviews_with_beer_info[reviews_with_beer_info.overall >= 4.0].text)


# In[ ]:


wordcloud = WordCloud().generate(good_text)
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.savefig('../images/good_words_cloud.png')


# In[ ]:


bad_text = ' '.join(review for review in reviews_with_beer_info[reviews_with_beer_info.overall <= 3.0].text)


# In[ ]:


wordcloud = WordCloud().generate(bad_text)
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.savefig('../images/bad_words_cloud.png')

