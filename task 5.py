
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define accident data as a dictionary
data = {
    "Road_Conditions": ["Wet", "Dry", "Snowy", "Icy", "Wet", "Dry", "Snowy", "Icy"],
    "Weather": ["Rainy", "Sunny", "Cloudy", "Foggy", "Rainy", "Sunny", "Cloudy", "Foggy"],
    "Time_of_Day": ["Morning", "Afternoon", "Evening", "Night", "Morning", "Afternoon", "Evening", "Night"],
    "Accident_Severity": [1, 2, 3, 4, 1, 2, 3, 4]
}

# Convert dictionary to pandas DataFrame
df = pd.DataFrame(data)

# Analyze accident frequency by road conditions
road_conditions = df["Road_Conditions"].value_counts()
plt.bar(road_conditions.index, road_conditions.values)
plt.xlabel("Road Conditions")
plt.ylabel("Accident Frequency")
plt.title("Accident Frequency by Road Conditions")
plt.show()

# Analyze accident severity by weather
weather = df["Weather"].value_counts()
plt.bar(weather.index, weather.values)
plt.xlabel("Weather")
plt.ylabel("Accident Severity")
plt.title("Accident Severity by Weather")
plt.show()

# Identify accident hotspots using spatial analysis (not applicable without location data)

# Analyze contributing factors using statistical models
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop("Accident_Severity", axis=1), df["Accident_Severity"], test_size=0.2, random_state=42)

# Train a random forest classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Evaluate model performance
print("Model Accuracy:", rfc.score(X_test, y_test))
