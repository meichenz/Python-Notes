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

# outer [] means select, inner [] means list 
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




	


	









 
 
 


