# NLP Lesson 2 Text Data Cleaning 


import nltk
nltk.download('punkt')
nltk.download('stopwords')

import pandas as pd 

pd.options.display.max_rows = 4000
pd.set_option('display.max_colwidth', 4000)

json_2017 = pd.read_json('condensed_2017.json')
json_2018 = pd.read_json('condensed_2018.json')

df=json_2017.append(json_2018, ignore_index=True)
df.head()

# NaN means data is missing 

# [IMPORTANT] outer [] means select, inner [] means list 
df[['text', 'created_at']].head()

print("how many records ? ", len(df))
print("general information about the data: ")
df.describe() # only for numeric features


############### Basic Data Cleaning ##################

del df['source']
# Or use 
df = df.drop(['id_str', 'in_reply_to_user_id_str'], axis=1)

# only keep records where is not retweet , creating condition as a Boolean data frame
condition = df['is_retweet'] == False
# Use this condition to select eligible rows 
df = df[condition]
# Finally remove the column is_retweet 
del df['is_retweet']


# check if data type is correct 
df.dtypes 


############### Standardize Text ##################
# regular expression
import re

example = 'XXX aaa the safety of American people is my aosolute.....https:// ....'

def standardize_text(text):
	text = text.lower()
	text = re.sub("[.,!?'&$'\"\-()]", "", text)
	text = re.sub(r"http.+$", " ", text)
	text = text.strip()
	
	return text

	
example _clean = standardize_text(example)

############### Tokenization, Stemming and Lemmatisation ##################

from nltk.tokenize import work_tokenize
from nltk.stem import PorterStemmer 

# split words 
example_tokens= word_tokenize(example_clean)
print(example_tokens)

def tokenize_stem_token(text):
	tokens = work_tokenize(text)
	
	stemmer = PorterStemmer()
	stemmed_tokens = []
	for token in tokens:
		stemmed_tokens.append(stemmer.stem(token))
		
	return stemmed_tokens 
	
############### remove stopwords ##################
from nltk.corpus import stopwords 
stopwords.words()
len(stopwords.words())# 6800

def remove_stopwords(tokens):
	sw = stopwords.words()
	# list comprehension
	lean_tokens = [t for t in tokens if t not in sw]
	return lean_tokens 

############### Clean all tweets  ##################

# Apply(a function) - equal to a for loop on rows of a dataframe  
df['text'] = df['text'].apply(standardize_text)
df['text'] = df['text'].apply(tokenize_stem_token)
df['text'] = df['text'].apply(remove_stopwords)

# Persist dataframe to disk

df.to_pickel("df_cleaned.pickle")


# NLP Lesson 3 Feature Extraction

!pip install plotly
!pip install wordcloud

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px # popular, highlevel API, interactive graph
import plotly.graph_objects as go
import pandas as pd


# 1. Read in DF
df=pd.read_pickle('df2_cleaned.pickle')

# transform a list of words to a string of sentence 
# this is for CountVectorizer to transform sentences into bag of words 

df['text'] = df['text'].apply(lambda x: " ".join(x))
df.head()
len(df)

# 2. Generate Bag of Words

# <IMPORTANT> https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
from sklearn.feature_extraction.text import CountVectorizer

# CountVectorizer can be used to configure NLP methods
# eg use ngram_range to configure unigram, bigram or combination of both
vectorizer = CountVectorizer() # this API has functions : fit, transform and predict
text = vectorizer.fit_transform(df['text'])

text.shape
[output] (5341, 6248) # row, col

# transform the result into a dataframe
# Each row is one tweet, each column is one word
df_text = pd.DataFrame(text.toarray(), columns=vectorizer.get_feature_names())
df_text.head()


# 3. Top Words Visualization
# in the sum function, we can define how to sum either along row or col 
# axis = 0 refers to the index axis - so we sum along this axis for each column
word_sum=df_text.sum(axis=0).sort_values(ascending=False)
word_sum[:10]
[output]
great 1324
amp 911
wa 792
....]

top_100_word_df=pd.DataFrame(word_sum[:100])
top_100_word_df=top_100_word_df.reset_index() # otherwise it will has index value of the words above
top_100_word_df.columns=['word', 'count'] # name the columns 

# draw a bar plot
fig = px.bar(top_100_word_df, x='word', y='count')
fig.show()

# draw a pie plot
fig=px.pie(
	top_100_word_df.head(30),
	values='count',
	names='word',
	title="Top 30 Word Count of Trump's Tweets")
fig.show()

# 4. China-USA Trend Visualization

# Add back column 'created_at' column
df_text['created_at']=df['created_at']

# we consider use of word 'china' or 'xi' related to China 
# use of 'usa', 'america' or 'american' related to the US
# (Chinese was transformed to China during lemmatization)

# create boolean selector
is_china=(df_text['china']>0 | df_text['xi']>0)
print(sum(is_china))
is_china

is_usa=(df_text['usa']>0 | df_text['america']>0 | df_text['american']>0)
print(sum(is_usa))

# create dataframe that counts china related and usa related for each row
df_text_2=df_text[is_china | is_usa]
print(len(df_text_2))

df_text_2['china_related'] = df_text_2['china'] + df_text_2['xi']
df_text_2['usa_related'] = df_text_2['usa'] + df_text_2['america'] + df_text_2['american']

df_text_2 = df_text_2[['created_at', 'china_related', 'usa_related']]
df_text_2 = df_text_2.set_index('created_at') # so later we can use the value of index to group
df_text_2.head()

# group by month
df_groupby = df_text_2.groupby(pd.Grouper(freq='M'))
sum_by_month=df_groupby.sum()
# change index to 'year-month', using list comprehension API
sum_by_month.index=[str(x.year) + '-' + str(x.month) for x in sum_by_month.index]
sum_by_month.head()

# plotting line chart
fig = go.Figure()

fig.add_trace(go.Scatter(x=sum_by_month.index, y=sum_by_month.usa_related,
			 mode='lines+markers',
			 name='usa-related count'
			))

fig.add_trace(go.Scatter(x=sum_by_month.index, y=sum_by_month.china_related,
			 mode='lines+markers',
			 name='china-related count'
			))


# 5. Word Cloud
wc = WordCloud(background_color='white', colormap='Dark2', max_font_size=150, random_state=42)

# similarly, transform df to string separated by spaces
whole_text = " ".join(text for text in df['text'])
cloud = wc.generate(whole_text)

# plot the cloud using matplotlib
plt.imshow(cloud)
plt.axis("off")
plt.show()

# remove words that are not of interest 
whole_text_2 = whole_text.replace("amp", "")
cloud2 = wc.generate(whole_text_2)

plt.imshow(cloud2)
plt.axis("off")
plt.show()



# BIGRAM analysis
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer(ngram_range=(2, 2)) #bigram configuration
text=vectorizer.fit_transform(df['text'])

bigram_sum=df_text.sum(axis=0).sort_values(ascending=False)
top_100_bigram_df=pd.DataFrame(bigram_sum[:100])
top_100_bigarm_df=top_100_bigram_df.reset_index()
top_100_bigram_df.columns=['word', 'count']
top_100_bigram_df[:10]

# Dictionary Word Cloud Creation
freq_dict=top_100_bigram_df.set_index('word')['count'].to_dict()


# BIGRAM Word Cloud 
wc = WordCloud(background_color="while", colormap="Dark2", max_font_size=150, random_state=42)
cloud=wc.generate_from_frequencies(freq_dict)

plt.imshow(cloud)
plt.axis("off")
plt.show()

############ Lesson 4 Sentiment Analysis ############
! pip install textblob

from textblob import TextBlob

import matplotlib.pyplot as plt 
import plotly.express as px
impoer plotly.graph_objects as go 
import pandas as pd

pd.options.display.max_rows = 4000
pd.set_option('display.max_colwidth', 4000)

# How to use TextBlob
df = pd.read_pickle('df2_cleaned.pickle')
df['text']=df['text'].apply(lambda x: " ".join(x))

df['polarity'] = df['text'].apply(lambda x:TextBblob(x).sentiment[0])
df['subjectivity'] = df['text'].apply(lambda x: TextBlob(x).sentiment[1])

df.head()


# Sentiment Visualization

# See the average scores of polarity and subjectivity 
print("average polarity score: ", df['polarity'].mean())
print("average subjectivity score: ", df['subjectivity'].mean())

fig = px.histogram(df['polarity'], x='polarity', title='Histogram of Polarity of Trump\'s Tweets')
fig.show()

fig = px.histogram(df['subjectivity'], x='subjectivity', title='Histogram of Subjectivity of Trump\'s Tweets')
fig.show()

# Similarly get subset of China/USA related tweets
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
bow = vectorizer.fit_transform(df['text'])

df_bow = pd.DataFrame(bow.toarray(), columns=vectorizer.get_feature_names())
df_bow.head()

is_china = (df_bow['china'] > 0 | df_bow['xi'] > 0)
is_usa = (df_bow['usa'] > 0 | df_bow['america'] > 0 | df_bow['american'] > 0)

df.index = is_china.index
df_small = df[is_china | is_usa]
# create field 'related_to', set default value to usa, and then if related to China, set to china
df_small['related_to'] = 'USA'
df_small.loc[is_china, 'related_to'] = 'China'

df_small.head()

# visualization
fig=px.histogram(df_small, x='polarity', color='related_to', 
		marginal='rug', hover_data=['text', 'polarity'])
fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.75)
fig.show()


########################################################
####### NLP Lesson 5 Latent Dirichlet Allocation #######
########################################################








 
 
 


