import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

# Load data
ratings = pd.read_csv('/Users/akshayv/Desktop/ratings.csv', usecols=range(3))
events = pd.read_csv('/Users/akshayv/Desktop/events.csv', usecols=range(8))
users = pd.read_csv('/Users/akshayv/Desktop/users.csv', usecols=range(5))

# Print basic information about the datasets
print(ratings.head())
print(events.head())

n_ratings = len(ratings)
n_events = len(events)
n_users = ratings['UserId'].nunique()

print(f"Number of ratings: {n_ratings}")
print(f"Number of unique eventId's: {n_events}")
print(f"Number of unique users: {n_users}")
print(f"Average number of ratings per user: {round(n_ratings / n_users, 2)}")
print(f"Average number of ratings per event: {round(n_ratings / n_events, 2)}")

sns.countplot(x='rating', data=ratings)
plt.title("Distribution of event ratings", fontsize=14)
plt.show()

print(f"Mean global rating: {round(ratings['rating'].mean(), 2)}.")

mean_ratings = ratings.groupby('UserId')['rating'].mean()
print(f"Mean rating per user: {round(mean_ratings.mean(), 2)}.")

print(ratings['eventId'].value_counts())

event_ratings = ratings.merge(events, on='eventId')
print(event_ratings['name'].value_counts()[0:10])

mean_ratings = ratings.groupby('eventId')[['rating']].mean()
lowest_rated = mean_ratings['rating'].idxmin()
highest_rated = mean_ratings['rating'].idxmax()

event_stats = ratings.groupby('eventId')['rating'].agg(['count', 'mean'])
event_stats.head()

C = event_stats['count'].mean()
m = event_stats['mean'].mean()
print(f"Average number of ratings for a given event: {C:.2f}")
print(f"Average rating for a given event: {m:.2f}")


def bayesian_avg(ratings):
    bayesian_avg = (C * m + ratings.sum()) / (C + ratings.count())
    return round(bayesian_avg, 3)


bayesian_avg_ratings = ratings.groupby('eventId')['rating'].agg(bayesian_avg).reset_index()
bayesian_avg_ratings.columns = ['eventId', 'bayesian_avg']
event_stats = event_stats.merge(bayesian_avg_ratings, on='eventId')
event_stats = event_stats.merge(events[['eventId', 'name']])
event_stats.sort_values(by='bayesian_avg', ascending=False)

event_stats.sort_values('bayesian_avg', ascending=True).head()


def create_event_profiles(events):
    events['type'] = events['type'].apply(lambda x: x.split('|'))
    type_frequency = Counter(t for types in events['type'] for t in types)
    type_frequency_df = pd.DataFrame([type_frequency]).T.reset_index()
    type_frequency_df.columns = ['type', 'count']

    sns.barplot(x='type', y='count', hue='type', data=type_frequency_df.sort_values(by='count', ascending=False),
                palette='viridis', legend=False)
    plt.xticks(rotation=90)
    plt.show()

    types = list(type_frequency.keys())
    event_profiles = pd.DataFrame(index=events.index, columns=types, data=0)
    for idx, row in events.iterrows():
        event_profiles.loc[idx, row['type']] = 1
    return event_profiles


event_profile = create_event_profiles(events)

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


def create_user_profile(user_id, ratings, events, event_profile):
    user_ratings = ratings[ratings['UserId'] == user_id].merge(events, on='eventId', how='left')
    user_ratings = user_ratings.drop(
        columns=['UserId', 'eventId', 'name', 'peoplecount', 'daysleft', 'location', 'agerecommended', 'description'])
    print(user_ratings)
    event_type_scores = {}
    for idx, row in user_ratings.iterrows():
        event_types = row['type']
        rating = row['rating']
        for event_type in event_types:
            if event_type not in event_type_scores:
                event_type_scores[event_type] = []
            event_type_scores[event_type].append(rating)

    mean_rating_per_type = {}
    for event_type, ratings in event_type_scores.items():
        mean_rating_per_type[event_type] = np.mean(ratings)

    for event_type in event_profile.columns:
        if event_type not in mean_rating_per_type:
            mean_rating_per_type[event_type] = 0

    event_type_scores = [mean_rating_per_type[event_type] for event_type in event_profile.columns]
    user_profile = np.array([event_type_scores])

    print(user_profile)
    print(event_profile)
    return user_profile

from sklearn.preprocessing import MinMaxScaler
def generate_recommendations(user_id, ratings, events, event_profile):
    user_profile = create_user_profile(user_id, ratings, events, event_profile)

    similarity_scores = cosine_similarity(user_profile, event_profile)

    scaler = MinMaxScaler()
    similarity_scores_normalized = scaler.fit_transform(similarity_scores.reshape(-1, 1)).flatten()

    event_ids = events['eventId']
    event_similarity = pd.DataFrame({'eventId': event_ids, 'similarity': similarity_scores_normalized})
    print(event_similarity)
    ranked_events = event_similarity.sort_values(by='similarity', ascending=False)

    return ranked_events


user_id = 1
recommendations_user_profile = generate_recommendations(user_id, ratings, events, event_profile)
print(recommendations_user_profile.head(16))


































def knowledge_based_recommendations(user_interests, top_n=5):
    filtered_events = events[events['type'].apply(lambda types: any(interest in types for interest in user_interests))]
    mean_event_ratings = ratings.groupby('eventId')['rating'].mean()
    top_rated_events = mean_event_ratings.sort_values(ascending=False).index
    filtered_top_rated_events = filtered_events[filtered_events['eventId'].isin(top_rated_events)]
    recommended_events = filtered_top_rated_events.head(top_n)

    return recommended_events

user_id = 1
user_interests_str = users.loc[users['userId'] == user_id, 'Interests'].values[0]
user_interests = user_interests_str.split('|')

recommendations = knowledge_based_recommendations(user_interests,10)

print("Knowledge-based recommendations:")
print(recommendations[['eventId', 'name', 'type']])