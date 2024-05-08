import pandas as pd
import numpy as np


num_courses = 500
courses_data = {
    'course_id': range(0, num_courses ),
    'course_name': [f'Course {i}' for i in range(1, num_courses + 1)]
}
courses_df = pd.DataFrame(courses_data)

num_users = 1000
users_data = {
    'student_id': range(0, num_users ),
    'cgpa': np.random.uniform(2.0, 4.0, size=num_users),
    'student_name': [f'Student {i}' for i in range(1, num_users + 1)]
}
users_df = pd.DataFrame(users_data)

num_interactions = 5000
interactions_data = {
    'student_id': np.random.choice(users_df['student_id'], size=num_interactions),
    'course_id': np.random.choice(courses_df['course_id'], size=num_interactions),
    'grade': np.random.choice(['A', 'B', 'C', 'D', 'F'], size=num_interactions)
}
interactions_df = pd.DataFrame(interactions_data)

courses_df.to_csv("courses.csv", index=False)
users_df.to_csv("users.csv", index=False)
interactions_df.to_csv("interactions.csv", index=False)
