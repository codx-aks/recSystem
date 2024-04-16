import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

ratings = pd.read_csv('/Users/akshayv/Desktop/ratings.csv', usecols=range(3))
events = pd.read_csv('/Users/akshayv/Desktop/events.csv', usecols=range(8))
users = pd.read_csv('/Users/akshayv/Desktop/users.csv', usecols=range(5))

print(ratings.head())
print(events.head())
print(users.head())

user_id = 101

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

    X = csr_matrix((df["rating"], (user_index, item_index)), shape=(M, N))

    return X, user_mapper, event_mapper, user_inv_mapper, event_inv_mapper


X, user_mapper, event_mapper, user_inv_mapper, event_inv_mapper = create_X(ratings)

n_total = X.shape[0] * X.shape[1]
n_ratings = X.nnz
sparsity = n_ratings / n_total
print(f"Matrix sparsity: {round(sparsity * 100, 2)}%")

n_ratings_per_user = X.getnnz(axis=1)

print(len(n_ratings_per_user))

print(f"Most active user rated {n_ratings_per_user.max()} events.")
print(f"Least active user rated {n_ratings_per_user.min()} events.")

n_ratings_per_event = X.getnnz(axis=0)

len(n_ratings_per_event)

print(f"Most rated event has {n_ratings_per_event.max()} ratings.")
print(f"Least rated event has {n_ratings_per_event.min()} ratings.")

plt.figure(figsize=(16, 4))
plt.subplot(1, 2, 1)
sns.kdeplot(n_ratings_per_user, fill=True)
plt.xlim(0)
plt.title("Number of Ratings Per User", fontsize=14)
plt.xlabel("number of ratings per user")
plt.ylabel("density")
plt.subplot(1, 2, 2)
sns.kdeplot(n_ratings_per_event, fill=True)
plt.xlim(0)
plt.title("Number of Ratings Per event", fontsize=14)
plt.xlabel("number of ratings per event")
plt.ylabel("density")
plt.show()


def bayesian_avg(ratings):
    bayesian_avg = (C * m + ratings.sum()) / (C + ratings.count())
    return round(bayesian_avg, 3)


bayesian_avg_ratings = ratings.groupby('eventId')['rating'].agg(bayesian_avg).reset_index()
bayesian_avg_ratings.columns = ['eventId', 'bayesian_avg']
event_stats = event_stats.merge(bayesian_avg_ratings, on='eventId')
event_stats = event_stats.merge(events[['eventId', 'name']])
event_stats.sort_values(by='bayesian_avg', ascending=False)

event_stats.sort_values('bayesian_avg', ascending=True).head()


def create_event_profiles_eventtype(events):
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


event_profile_eventtype = create_event_profiles_eventtype(events)
print(event_profile_eventtype)


def create_user_profile_eventtype(user_id, ratings, events, event_profile):
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


def generate_recommendations_eventtype(user_id, ratings, events, event_profile):
    user_profile = create_user_profile_eventtype(user_id, ratings, events, event_profile)

    similarity_scores = cosine_similarity(user_profile, event_profile)

    scaler = MinMaxScaler()
    similarity_scores_normalized = scaler.fit_transform(similarity_scores.reshape(-1, 1)).flatten()

    event_ids = events['eventId']
    event_similarity = pd.DataFrame({'eventId': event_ids, 'similarity': similarity_scores_normalized})

    return event_similarity


recommendations_user_profile_type = generate_recommendations_eventtype(user_id, ratings, events,
                                                                       event_profile_eventtype)
print(recommendations_user_profile_type.head(16))


def create_event_profiles_count(events):
    count_ranges = [0, 20, 50, 100, 250, 500, 1000, 5000, 100000]
    count_profile = pd.DataFrame(index=events.index, columns=count_ranges, data=0)

    for idx, row in events.iterrows():
        for count_range in count_ranges:
            if row['peoplecount'] >= count_range:
                count_profile.loc[idx, count_range] = 1

    plt.figure(figsize=(10, 6))
    sns.countplot(x='peoplecount', hue='peoplecount', data=events, palette='viridis')
    plt.title("Distribution of people count", fontsize=14)
    plt.xlabel("people count")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

    return count_profile


event_profile_count = create_event_profiles_count(events)
print(event_profile_count)


def create_user_profile_count(user_id, ratings, events, event_profile):
    user_ratings = ratings[ratings['UserId'] == user_id].merge(events, on='eventId', how='left')
    user_ratings = user_ratings.drop(
        columns=['UserId', 'eventId', 'name', 'type', 'daysleft', 'location', 'agerecommended', 'description'])

    count_ranges = [0, 20, 50, 100, 250, 500, 1000, 5000, 100000]
    count_profile = pd.DataFrame(index=np.arange(len(user_ratings)), columns=count_ranges, data=0)

    for idx, row in user_ratings.iterrows():
        for count_range in count_ranges:
            if row['peoplecount'] >= count_range:
                count_profile.loc[idx, count_range] = 1

    mean_rating_per_count = {}
    for count_range in count_ranges:
        ratings = user_ratings[user_ratings['peoplecount'] >= count_range]['rating']
        mean_rating_per_count[count_range] = ratings.mean()

    for count_range in count_ranges:
        if count_range not in mean_rating_per_count:
            mean_rating_per_count[count_range] = 0

    user_profile = pd.Series([mean_rating_per_count[count_range] for count_range in count_ranges]).values.reshape(1, -1)
    user_profile = np.nan_to_num(user_profile, nan=0)
    print(user_profile)
    return user_profile


def generate_recommendations_count(user_id, ratings, events, event_profile):
    user_profile = create_user_profile_count(user_id, ratings, events, event_profile)

    similarity_scores = cosine_similarity(user_profile, event_profile)

    scaler = MinMaxScaler()
    similarity_scores_normalized = scaler.fit_transform(similarity_scores.reshape(-1, 1)).flatten()

    event_ids = events['eventId']
    event_similarity = pd.DataFrame({'eventId': event_ids, 'similarity': similarity_scores_normalized})

    return event_similarity


recommendations_user_profile_count = generate_recommendations_count(user_id, ratings, events, event_profile_count)
print(recommendations_user_profile_count.head(16))


def create_event_profiles_agerecommended(events):
    age_ranges = [0, 6, 12, 16, 18, 21, 30, 45]
    age_profile = pd.DataFrame(index=events.index, columns=age_ranges, data=0)

    for idx, row in events.iterrows():
        for age_range in age_ranges:
            if row['agerecommended'] <= age_range:
                age_profile.loc[idx, age_range] = 1

    plt.figure(figsize=(10, 6))
    sns.countplot(x='agerecommended', hue='agerecommended', data=events, palette='viridis')
    plt.title("Distribution of Age Recommendations", fontsize=14)
    plt.xlabel("Age Recommended")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

    return age_profile


event_profile_agerecommended = create_event_profiles_agerecommended(events)
print(event_profile_agerecommended)


def create_user_profile_agerecommended(user_id, ratings, events, event_profile):
    user_ratings = ratings[ratings['UserId'] == user_id].merge(events, on='eventId', how='left')
    user_ratings = user_ratings.drop(
        columns=['UserId', 'eventId', 'name', 'type', 'daysleft', 'location', 'peoplecount', 'description'])

    agerecommended_ranges = [0, 6, 12, 16, 18, 21, 30, 45]
    agerecommended_profile = pd.DataFrame(index=np.arange(len(user_ratings)), columns=agerecommended_ranges, data=0)

    for idx, row in user_ratings.iterrows():
        for agerecommended_range in agerecommended_ranges:
            if row['agerecommended'] <= agerecommended_range:
                agerecommended_profile.loc[idx, agerecommended_range] = 1

    mean_rating_per_agerecommended = {}
    for agerecommended_range in agerecommended_ranges:
        ratings = user_ratings[user_ratings['agerecommended'] >= agerecommended_range]['rating']
        mean_rating_per_agerecommended[agerecommended_range] = ratings.mean()

    for agerecommended_range in agerecommended_ranges:
        if agerecommended_range not in mean_rating_per_agerecommended:
            mean_rating_per_agerecommended[agerecommended_range] = 0

    user_profile = pd.Series([mean_rating_per_agerecommended[agerecommended_range] for agerecommended_range in
                              agerecommended_ranges]).values.reshape(1, -1)
    user_profile = np.nan_to_num(user_profile, nan=0)
    print(user_profile)
    return user_profile


def generate_recommendations_agerecommended(user_id, ratings, events, event_profile):
    user_profile = create_user_profile_agerecommended(user_id, ratings, events, event_profile)

    similarity_scores = cosine_similarity(user_profile, event_profile)

    scaler = MinMaxScaler()
    similarity_scores_normalized = scaler.fit_transform(similarity_scores.reshape(-1, 1)).flatten()

    event_ids = events['eventId']
    event_similarity = pd.DataFrame({'eventId': event_ids, 'similarity': similarity_scores_normalized})

    return event_similarity


recommendations_user_profile_agerecommended = generate_recommendations_agerecommended(user_id, ratings, events,
                                                                                      event_profile_agerecommended)
print(recommendations_user_profile_agerecommended.head(16))


def create_event_profiles_daysleft(events):
    daysleft_ranges = [1, 3, 7, 15, 30, 60, 100, 1000]
    daysleft_profile = pd.DataFrame(index=events.index, columns=daysleft_ranges, data=0)

    for idx, row in events.iterrows():
        for daysleft_range in daysleft_ranges:
            if row['daysleft'] >= daysleft_range:
                daysleft_profile.loc[idx, daysleft_range] = 1

    plt.figure(figsize=(10, 6))
    sns.countplot(x='daysleft', hue='daysleft', data=events, palette='viridis')
    plt.title("Distribution of days left", fontsize=14)
    plt.xlabel("days left")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

    return daysleft_profile


event_profile_daysleft = create_event_profiles_daysleft(events)
print(event_profile_daysleft)


def create_user_profile_daysleft(user_id, ratings, events, event_profile):
    user_ratings = ratings[ratings['UserId'] == user_id].merge(events, on='eventId', how='left')
    user_ratings = user_ratings.drop(
        columns=['UserId', 'eventId', 'name', 'type', 'peoplecount', 'location', 'agerecommended', 'description'])

    daysleft_ranges = [1, 3, 7, 15, 30, 60, 100, 1000]
    daysleft_profile = pd.DataFrame(index=np.arange(len(user_ratings)), columns=daysleft_ranges, data=0)

    for idx, row in user_ratings.iterrows():
        for daysleft_range in daysleft_ranges:
            if row['daysleft'] >= daysleft_range:
                daysleft_profile.loc[idx, daysleft_range] = 1

    mean_rating_per_daysleft = {}
    for daysleft_range in daysleft_ranges:
        ratings = user_ratings[user_ratings['daysleft'] >= daysleft_range]['rating']
        mean_rating_per_daysleft[daysleft_range] = ratings.mean()

    for daysleft_range in daysleft_ranges:
        if daysleft_range not in mean_rating_per_daysleft:
            mean_rating_per_daysleft[daysleft_range] = 0

    user_profile = pd.Series(
        [mean_rating_per_daysleft[daysleft_range] for daysleft_range in daysleft_ranges]).values.reshape(1, -1)
    user_profile = np.nan_to_num(user_profile, nan=0)
    print(user_profile)
    return user_profile


def generate_recommendations_daysleft(user_id, ratings, events, event_profile):
    user_profile = create_user_profile_daysleft(user_id, ratings, events, event_profile)

    similarity_scores = cosine_similarity(user_profile, event_profile)

    scaler = MinMaxScaler()
    similarity_scores_normalized = scaler.fit_transform(similarity_scores.reshape(-1, 1)).flatten()

    event_ids = events['eventId']
    event_similarity = pd.DataFrame({'eventId': event_ids, 'similarity': similarity_scores_normalized})

    return event_similarity


recommendations_user_profile_daysleft = generate_recommendations_daysleft(user_id, ratings, events,
                                                                          event_profile_daysleft)
print(recommendations_user_profile_daysleft['similarity'].head(16))

rec_result = recommendations_user_profile_daysleft
rec_result['similarity'] *= 0.13
rec_result['similarity'] += (0.13 * recommendations_user_profile_agerecommended['similarity'])
rec_result['similarity'] += (0.13 * recommendations_user_profile_count['similarity'])
rec_result['similarity'] += (0.61 * recommendations_user_profile_type['similarity'])


def quick_select_linear(arr, k):
# QuickSelect algorithm with linear time complexity.

    if len(arr) == 1:
        return arr[0]

    pivot = median_of_medians(arr)
    left, mid, right = partition(arr, pivot)

    if k <= len(left):
        return quick_select_linear(left, k)
    elif k <= len(left) + len(mid):
        return pivot
    else:
        return quick_select_linear(right, k - len(left) - len(mid))


def median_of_medians(arr):
# Find the median of medians of the input array.

    n = len(arr)
    if n <= 5:
        return sorted(arr)[n // 2]

    chunks = [arr[i:i + 5] for i in range(0, n, 5)]
    medians = [sorted(chunk)[len(chunk) // 2] for chunk in chunks]
    pivot = median_of_medians(medians)
    return pivot


def partition(arr, pivot):
# Partition the input array into three parts based on the pivot.
    left = []
    mid = []
    right = []
    for similarity in arr:
        if similarity < pivot:
            left.append(similarity)
        elif similarity == pivot:
            mid.append(similarity)
        else:
            right.append(similarity)
    return left, mid, right

k=50

top_k_threshold = quick_select_linear(rec_result['similarity'], k)
top_k_similarity = rec_result['similarity'][rec_result['similarity'] >= top_k_threshold]

top_k_similarity = top_k_similarity.sort_values(ascending=False)

top_k_events = rec_result.loc[top_k_similarity.index[:k]]

print(top_k_events)

def knowledge_based_recommendations(user_interests, top_n=5):
    filtered_events = events[events['type'].apply(lambda types: any(interest in types for interest in user_interests))]
    mean_event_ratings = ratings.groupby('eventId')['rating'].mean()
    top_rated_events = mean_event_ratings.sort_values(ascending=False).index
    filtered_top_rated_events = filtered_events[filtered_events['eventId'].isin(top_rated_events)]
    recommended_events = filtered_top_rated_events.head(top_n)

    return recommended_events


user_interests_str = users.loc[users['userId'] == user_id, 'Interests'].values[0]
user_interests = user_interests_str.split('|')

recommendations = knowledge_based_recommendations(user_interests, 10)

print("Knowledge-based recommendations:")
print(recommendations[['eventId', 'name', 'type']])
