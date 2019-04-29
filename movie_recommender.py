import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
###### helper functions. Use them when needed #######
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]
##################################################

##Step 1: Read CSV File
df=pd.read_csv("movie_dataset.csv")
print(df.head())
print(df.columns)
##Step 2: Select Features
features=['keywords','cast','genres','director']
##Step 3: Create a column in DF which combines all selected features

#when a NaN feature value is encountered fill it with blank string
for feature in features:
	df[feature]=df[feature].fillna('')

def combine_features(row):
	try:
		return row['keywords']+" "+row['cast']+" "+row["genres"]+" "+row["director"]
	except:
		print("Error:",row)	

df["combined_features"]=df.apply(combine_features,axis=1)
#Combine the features col wise so axis=1. Bydefault it combines row wise.
#I need all keywords together so i will combine col wise

print(df["combined_features"])

##Step 4: Create count matrix from this new combined column

cv = CountVectorizer()
count_matrix= cv.fit_transform(df["combined_features"])

##Step 5: Compute the Cosine Similarity based on the count_matrix
cosine_sim = cosine_similarity(count_matrix)
movie_user_likes = "Avatar"

## Step 6: Get index of this movie from its title
movie_index = get_index_from_title(movie_user_likes)  #Gives index of Avatar in this case

#check 1:25 of video
#similar_movies=cosine_sim[movie_index] 
#go to the cosine matrix and fine the movie row Avatar
#Convert this into a list and provide enumeration to it
similar_movies = list(enumerate(cosine_sim[movie_index]))



## Step 7: Get a list of similar movies in descending order of similarity score
#sorting from second element of the tuple so use lambda expression
#sort in descending ......... reverse=True
sorted_similar_movies = sorted(similar_movies,key= lambda x:x[1], reverse=True)

## Step 8: Print titles of first 50 movies
#[(0,1),(1,0.8),(3,0.5)]	movie[0] will return 0 for 1st rec

i=0
for movie in sorted_similar_movies:
	print(get_title_from_index(movie[0]))
	i=i+1
	if i>50:
		break