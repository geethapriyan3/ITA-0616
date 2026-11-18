import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder

# Create dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy',
                'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool',
                    'Cool', 'Cool', 'Mild', 'Cool', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal',
                 'Normal', 'Normal', 'High', 'Normal', 'Normal'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak',
             'Strong', 'Strong', 'Weak', 'Weak', 'Weak'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes',
             'No', 'Yes', 'No', 'Yes', 'Yes']
}

df = pd.DataFrame(data)

# Encode categorical values
encoders = {}

for column in df.columns:
    encoders[column] = LabelEncoder()
    df[column] = encoders[column].fit_transform(df[column])

# Separate features and target
X = df.drop('Play', axis=1)
y = df['Play']

# Build ID3 decision tree
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)

# Display decision tree
print("Decision Tree:")
print(export_text(model, feature_names=list(X.columns)))

# Classify a new sample
sample = pd.DataFrame([{
    'Outlook': 'Rainy',
    'Temperature': 'Cool',
    'Humidity': 'Normal',
    'Wind': 'Weak'
}])

# Encode the new sample using the same encoders
for column in sample.columns:
    sample[column] = encoders[column].transform(sample[column])

prediction = model.predict(sample)

result = encoders['Play'].inverse_transform(prediction)

print("New Sample:", {
    'Outlook': 'Rainy',
    'Temperature': 'Cool',
    'Humidity': 'Normal',
    'Wind': 'Weak'
})

print("Predicted Class:", result[0])