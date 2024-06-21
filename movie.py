import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_csv('IMDb Movies India.csv', encoding='latin1')


df = df.dropna(subset=['Rating'])  # Drop rows where the target variable is missing


# Convert 'Duration' to integer and fill NaN values with the median
df['Duration'] = df['Duration'].str.replace(' min', '')
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
df['Duration'] = df['Duration'].fillna(df['Duration'].median())

# Convert 'Votes' to integer and fill NaN values with the median
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
df['Votes'] = df['Votes'].fillna(df['Votes'].median())



# Select relevant features for the model
X = df[['Duration', 'Genre', 'Director', 'Votes']]
y = df['Rating']

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Preprocess the data (OneHotEncoding for categorical features, StandardScaler for numerical features)
preprocessor = ColumnTransformer(
    transformers=[
        ('numerical', StandardScaler(), ['Duration', 'Votes']),
        ('categorical', OneHotEncoder(handle_unknown='ignore'), ['Genre', 'Director'])
    ]
)


#Create a pipeline with preprocessing and regression
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train the model
pipeline.fit(X_train, y_train)

# Predict the ratings on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Lets try an predict a missing rating of a movie
new_data = pd.DataFrame({
    'Duration': [90],
    'Genre': ['Drama Musical'],
    'Director': ['Soumyajit Majumdar'],
    'Votes': [8]
})

# Use the trained pipeline to predict the rating
predicted_rating = pipeline.predict(new_data)

print(f'Predicted Rating: {predicted_rating[0]}')

