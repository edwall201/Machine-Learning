from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt

# OUTLOOK | TEMPERATURE | HUMIDITY | WINDY | CLASS  
# =====================================================   
# sunny   |      85     |    85    | false | Don't Play   
# sunny   |      80     |    90    | true  | Don't Play   
# overcast|      83     |    78    | false | Play   
# rainy   |      70     |    96    | false | Play   
# rainy   |      68     |    80    | false | Play   
# rainy   |      65     |    70    | true  | Don't Play   
# overcast|      64     |    65    | true  | Play   
# sunny   |      72     |    95    | false | Don't Play   
# sunny   |      69     |    70    | false | Play   
# rainy   |      75     |    80    | false | Play   
# sunny   |      75     |    70    | true  | Play   
# overcast|      72     |    90    | true  | Play   
# overcast|      81     |    75    | false | Play   
# rainy   |      71     |    80    | true  | Don't Play 

data = {
    'outlook': ['sunny', 'sunny', 'overcast', 'rainy', 'rainy', 'rainy', 'overcast', 'sunny', 'sunny', 'rainy', 'sunny', 'overcast', 'overcast', 'rainy'],
    'temperature': [85, 80, 83, 70, 68, 65, 64, 72, 69, 75, 75, 72, 81, 71],
    'humidity': [85, 90, 78, 96, 80, 70, 65, 95, 70, 80, 70, 90, 75, 80],
    'windy': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
    'class': ['Don\'t Play', 'Don\'t Play', 'Play', 'Play', 'Play', 'Don\'t Play', 'Play', 'Don\'t Play', 'Play', 'Play', 'Play', 'Play', 'Play', 'Don\'t Play']
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Encode categorical data
encoder = LabelEncoder()
df['outlook'] = encoder.fit_transform(df['outlook'])

# Separate features and target variable
X = df.drop(columns=['class'])
y = df['class']

# Create and fit the Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True)
plt.show()

# Predict for a new sample
new_sample = pd.DataFrame({'outlook': ['sunny'], 'temperature': [80], 'humidity': [75], 'windy': [False]})
new_sample['outlook'] = encoder.transform(new_sample['outlook'])
prediction = clf.predict(new_sample)
print(f"The decision for the new sample is: {prediction[0]}")
