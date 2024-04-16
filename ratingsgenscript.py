import random
import pandas as pd

def generate_rating(user_interests, event_types):

    if any(interest in event_types for interest in user_interests):
        return 2
    else:
        return random.choices([1, -1], weights=[3, 1])[0]

def read_user_interests(user_id, user_data):

    user_row = user_data[user_data['userId'] == user_id]
    if not user_row.empty:
        interests_str = user_row['Interests'].iloc[0]
        return interests_str.split("|") if interests_str else []
    return []

def read_event_types(event_id, event_data):

    event_row = event_data[event_data['eventId'] == event_id]
    if not event_row.empty:
        types_str = event_row['type'].iloc[0]
        return types_str.split("|") if types_str else []
    return []

def generate_ratings(num_ratings, chunk_size=10000):

    data = []

    user_interests_data = pd.read_csv("users.csv")
    event_types_data = pd.read_csv("events.csv")

    for i in range(0, num_ratings, chunk_size):
        print(i)
        chunk_ratings = min(chunk_size, num_ratings - i)
        ratings = []

        for _ in range(chunk_ratings):
            user_id = random.randint(1, 100000)
            event_id = random.randint(1, 50000)

            if i < 1500000:

                user_interests = read_user_interests(user_id, user_interests_data)
                event_types = read_event_types(event_id, event_types_data)
            else:
                user_interests = []
                event_types = []

            rating = generate_rating(user_interests, event_types)
            ratings.append([user_id, event_id, rating])

        df = pd.DataFrame(ratings, columns=["UserId", "eventId", "rating"])

        data.append(df)


    df = pd.concat(data, ignore_index=True)

    df.to_csv("ratings.csv", index=False, mode='w', encoding='utf-8')

    print("ratings.csv generated successfully!")


num_ratings = 1000000
chunk_size = 10000

generate_ratings(num_ratings, chunk_size)
