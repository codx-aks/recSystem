import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ratings = pd.read_csv('/Users/akshayv/Desktop/ratings.csv', usecols=range(8))
events = pd.read_csv('/Users/akshayv/Desktop/events.csv', usecols=range(8))
users = pd.read_csv('/Users/akshayv/Desktop/users.csv', usecols=range(8))

print(ratings.head())
print(events.head())


n_ratings = len(ratings)
n_events = len(events)
n_users = ratings['UserId'].nunique()

print(f"Number of ratings: {n_ratings}")
print(f"Number of unique eventId's: {n_events}")
print(f"Number of unique users: {n_users}")
print(f"Average number of ratings per user: {round(n_ratings/n_users, 2)}")
print(f"Average number of ratings per event: {round(n_ratings/n_events, 2)}")

sns.countplot(x='rating', data=ratings)
plt.title("Distribution of event ratings", fontsize=14)
plt.show()

print(f"Mean global rating: {round(ratings['rating'].mean(),2)}.")

mean_ratings = ratings.groupby('UserId')['rating'].mean()
print(f"Mean rating per user: {round(mean_ratings.mean(),2)}.")

print(ratings['eventId'].value_counts())

event_ratings = ratings.merge(events,on='eventId')
# movie_ratings
print(event_ratings['name'].value_counts()[0:10])

mean_ratings = ratings.groupby('eventId')[['rating']].mean()
lowest_rated = mean_ratings['rating'].idxmin()

highest_rated = mean_ratings['rating'].idxmax()

event_stats = ratings.groupby('eventId')['rating'].agg(['count','mean'])
event_stats.head()

C = event_stats['count' ].mean()
m = event_stats['mean' ].mean()
print(f"Average number of ratings for a given movie: {C:.2f}")
print(f"Average rating for a given movie: {m:.2f}")

def bayesian_avg(ratings):
    bayesian_avg = (C*m + ratings.sum())/(C+ ratings.count())
    return round(bayesian_avg, 3)
lamerica = pd.Series([5,5])
bayesian_avg(lamerica)
bayesian_avg_ratings = ratings.groupby('eventId')['rating'].agg(bayesian_avg).reset_index()
#reset_index used to convert panda series to dataframe
bayesian_avg_ratings.columns = ['eventId', 'bayesian_avg']
event_stats = event_stats.merge(bayesian_avg_ratings, on='eventId')

event_stats = event_stats.merge(events[['eventId', 'name']])
event_stats.sort_values(by='bayesian_avg',ascending=False)

event_stats.sort_values('bayesian_avg', ascending=True).head()
print(events.head())


events['type'] = events['type'].apply(lambda x: x.split('|'))
from collections import Counter
type_frequency = Counter(t for types in events['type'] for t in types)
print(f"There are {len(type_frequency)} event types.")
print(type_frequency)
# print("The 5 most common event types : \n", type_frequency.most_common(5))
type_frequency_df = pd.DataFrame([type_frequency]).T.reset_index()
type_frequency_df.columns=(['type','count'])
print(type_frequency_df[0:5])

sns.barplot(x='type', y='count', hue='type', data=type_frequency_df.sort_values(by='count', ascending=False), palette='viridis', legend=False)
plt.xticks(rotation=90)
plt.show()

from scipy.sparse import csr_matrix

def create_X(df):
    M = df['UserId'].nunique()
    N = df['eventId'].nunique()

    user_mapper = dict(zip(np.unique(df["UserId"]), list(range(M))))
    event_mapper = dict(zip(np.unique(df["eventId"]), list(range(N))))

    user_inv_mapper = dict(zip(list(range(M)), np.unique(df["UserId"])))
    event_inv_mapper = dict(zip(list(range(N)), np.unique(df["eventId"])))

    user_index = [user_mapper[i] for i in df['UserId']]
    item_index = [event_mapper[i] for i in df['eventId']]

    X = csr_matrix((df["rating"], (user_index,item_index)), shape=(M,N))

    return X, user_mapper, event_mapper, user_inv_mapper, event_inv_mapper

X, user_mapper, event_mapper, user_inv_mapper, event_inv_mapper = create_X(ratings)

n_total = X.shape[0]*X.shape[1]
n_ratings = X.nnz
sparsity = n_ratings/n_total
print(f"Matrix sparsity: {round(sparsity*100,2)}%")

n_ratings_per_user = X.getnnz(axis=1)

print(len(n_ratings_per_user))

print(f"Most active user rated {n_ratings_per_user.max()} events.")
print(f"Least active user rated {n_ratings_per_user.min()} events.")

n_ratings_per_event = X.getnnz(axis=0)

len(n_ratings_per_event)

print(f"Most rated event has {n_ratings_per_event.max()} ratings.")
print(f"Least rated event has {n_ratings_per_event.min()} ratings.")

plt.figure(figsize=(16,4))
plt.subplot(1,2,1)
sns.kdeplot(n_ratings_per_user, fill=True)
plt.xlim(0)
plt.title("Number of Ratings Per User", fontsize=14)
plt.xlabel("number of ratings per user")
plt.ylabel("density")
plt.subplot(1,2,2)
sns.kdeplot(n_ratings_per_event, fill=True)
plt.xlim(0)
plt.title("Number of Ratings Per event", fontsize=14)
plt.xlabel("number of ratings per event")
plt.ylabel("density")
plt.show()

from sklearn.neighbors import NearestNeighbors

def find_similar_events(event_id, X, event_mapper, event_inv_mapper, k, metric='cosine'):
    X = X.T
    neighbour_ids = []
    event_ind = event_mapper[event_id]
    event_vec = X[event_ind]
    if isinstance(event_vec, (np.ndarray)):
        event_vec = event_vec.reshape(1,-1)
    # use k+1 since kNN output includes the eventId of interest
    kNN = NearestNeighbors(n_neighbors=k+1, algorithm="brute", metric=metric)
    kNN.fit(X)
    neighbour = kNN.kneighbors(event_vec.reshape(1, -1), return_distance=False)
    for i in range(0,k):
        n = neighbour.item(i)
        neighbour_ids.append(event_inv_mapper[n])
    neighbour_ids.pop(0)
    return neighbour_ids


similar_events = find_similar_events(1, X, event_mapper, event_inv_mapper, k=10)
print(similar_events)
event_names = dict(zip(events['eventId'], events['name']))


n_events = events['eventId'].nunique()
print(f"There are {n_events} unique events in our events dataset.")

types = set(t for T in events['type'] for t in T)
for t in types:
    events[t] = events.type.transform(lambda x: int(t in x))

event_types = events.drop(columns=['eventId', 'name','type','peoplecount','daysleft','location','agerecommended','description'])
print(event_types.head())


from sklearn.metrics.pairwise import cosine_similarity

cosine_sim_type = cosine_similarity(event_types, event_types)
print(f"Dimensions of our types cosine similarity matrix: {cosine_sim_type.shape}")


from fuzzywuzzy import process
def event_finder(name):
    all_names = events['name'].tolist()
    closest_match = process.extractOne(name, all_names)
    return closest_match[0]

name = event_finder('Event6')
event_idx = dict(zip(events['name'], list(events.index)))
idx =event_idx[name]
print(f"Event index for Event6: {idx}")

n_recommendations=5
sim_scores = list(enumerate(cosine_sim_type[idx]))
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
print(sim_scores[:5])

print(events['name'].iloc[similar_events])





from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=12, n_iter=10 )
Q = svd.fit_transform(X.T)

event_id = 1
similar_events = find_similar_events(event_id, Q.T, event_mapper, event_inv_mapper, metric='cosine', k=10)
event_name = event_names[event_id]

print(f"Because you liked {event_name}:")
for i in similar_events:
    print(event_names[i])





def knowledge_based_recommendations(user_interests, top_n=5):
    filtered_events = events[events['type'].apply(lambda types: any(interest in types for interest in user_interests))]
    mean_event_ratings = ratings.groupby('eventId')['rating'].mean()
    top_rated_events = mean_event_ratings.sort_values(ascending=False).index
    filtered_top_rated_events = filtered_events[filtered_events['eventId'].isin(top_rated_events)]
    recommended_events = filtered_top_rated_events.head(top_n)

    return recommended_events

user_id = 1
user_interests_str = users.loc[users['UserId'] == user_id, 'Interests'].values[0]
user_interests = user_interests_str.split('|')

recommendations = knowledge_based_recommendations(user_interests)

print("Knowledge-based recommendations:")
print(recommendations[['eventId', 'name', 'type']])