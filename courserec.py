import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, f1_score
from torch_geometric.nn import GCNConv


users_df = pd.read_csv('students.csv')
courses_df = pd.read_csv('courses.csv')
grades_df = pd.read_csv('grades.csv')

user_encoder = LabelEncoder()
course_encoder = LabelEncoder()

grades_df['studentId'] = user_encoder.fit_transform(grades_df['studentId'])
grades_df['courseId'] = course_encoder.fit_transform(grades_df['courseId'])

grade_mapping = {'S': 5, 'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
grades_df['grade'] = grades_df['grade'].map(grade_mapping)

scaler = MinMaxScaler()
users_df['CGPA'] = scaler.fit_transform(users_df[['CGPA']])

def compute_metrics(pred_labels, true_labels):
    pred_labels_flat = pred_labels.flatten()
    true_labels_flat = true_labels.flatten()
    accuracy = accuracy_score(true_labels_flat, pred_labels_flat)
    precision = precision_score(true_labels_flat, pred_labels_flat, average='macro', zero_division=0)
    f1 = f1_score(true_labels_flat, pred_labels_flat, average='macro', zero_division=0)
    return accuracy, precision, f1

user_features = torch.tensor(users_df[['CGPA', 'year']].values, dtype=torch.float)
course_features = torch.randn(len(courses_df), 2)

edge_index = torch.tensor(grades_df[['studentId', 'courseId']].values.T, dtype=torch.long)

edge_weight = torch.ones(edge_index.shape[1])

data = Data(x=torch.cat((user_features, course_features), dim=0), edge_index=edge_index,edge_attr=edge_weight)

class ComplexGCNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super(ComplexGCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        self.conv3 = GCNConv(hidden_dim2, hidden_dim3)
        self.conv4 = GCNConv(hidden_dim3, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv4(x, edge_index)
        return x

input_dim = 2
hidden_dim1 = 128
hidden_dim2 = 256
hidden_dim3 = 512
output_dim = 2

model = ComplexGCNModel(input_dim=input_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, hidden_dim3=hidden_dim3, output_dim=output_dim)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

user_embeddings = model(data)
predictions = torch.sigmoid(torch.matmul(user_embeddings, user_embeddings.t()))

pred_labels = (predictions > 0.5).int()
accuracies, precisions, f1_scores = [], [], []
true_labels = torch.zeros(predictions.shape)

for _, row in grades_df.iterrows():
    true_labels[row['studentId'], row['courseId']] = row['grade']

for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    user_embeddings = model(data)
    predictions = torch.sigmoid(torch.matmul(user_embeddings, user_embeddings.t()))

    pred_labels = (predictions > 0.5).int()

    loss = criterion(predictions, true_labels.float())
    loss.backward()
    optimizer.step()

    accuracy, precision, f1 = compute_metrics(pred_labels.cpu().numpy(), true_labels.cpu().numpy())
    accuracies.append(accuracy)
    precisions.append(precision)
    f1_scores.append(f1)
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}, Accuracy: {accuracy}, Precision: {precision}, F1: {f1}')


plt.figure(figsize=(12, 8))

plt.plot(accuracies, label='Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))

plt.plot(precisions, label='Precision')
plt.title('Model Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.tight_layout()
plt.legend()
plt.show()


plt.figure(figsize=(12, 8))

plt.plot(f1_scores, label='F1 Score')
plt.title('Model F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')

plt.tight_layout()
plt.legend()
plt.show()


user_id = 62
user_current_year = users_df.loc[users_df['studentId'] == user_id, 'year'].values[0]

next_year_courses = courses_df[courses_df['year'] == (user_current_year + 1)]

user_embeddings = model(data)
similarity_scores = torch.matmul(user_embeddings[user_id], user_embeddings.t())
recommended_courses = torch.argsort(similarity_scores, descending=True)

recommended_courses = [int(course_id) for course_id in recommended_courses]


# print("Printing recommendations...")
# for course_id in recommended_courses:
#     print(f"Course ID: {course_id}")

filtered_recommendations = [int(course_id) for course_id in recommended_courses if course_id < 125 and courses_df.loc[int(course_id), 'year'] == user_current_year + 1]

print(f"Recommended courses for user {user_id} (next year):")
for course_id in filtered_recommendations:
    course_name = courses_df.loc[int(course_id), 'courseName']
    print(f"Course ID: {course_id}, Course Name: {course_name}")
