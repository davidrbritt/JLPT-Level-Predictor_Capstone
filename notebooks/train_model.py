import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
#Parses Japanese text to create features
#Trains Random Forest Classifier(Predictive Method)
#Evaluates The accuracy of the model( Section D "Testing")

#Load Data
df = pd.read_csv('data/jlpt_vocab.csv')

#CLEANING DATA
#Drop the English column to keep the model focused only on the Japanese text
#And avoid mixing languages
initial_count = len(df)
if 'English' in df.columns:
    df =df.drop(columns=['English'])

#Drop rows missing critical data and remove duplicate words
df =df.dropna(subset=['Original', 'Furigana', 'JLPT Level'])
df =df.drop_duplicates(subset=['Original'])

#Clean text formatting
for col in ['Original', 'Furigana', 'JLPT Level']:
    df[col] = df[col].astype(str).str.strip()

print(f"Data Cleaning Complete: {len(df)} words remaining (Removed {initial_count - len(df)} rows.)")

#FEATURE ENGINEERING
def extract_features(row):
    word = str(row['Original'])
    reading = str(row['Furigana'])
    kanji_count = sum(1 for char in word if '\u4e00' <= char <= '\u9faf')
    hiragana_count = sum(1 for char in word if '\u3040' <= char <= '\u309f')
    katakana_count = sum(1 for char in word if '\u30a0' <= char <= '\u30ff')
    
    return [
        len(word),    #Written Length
        len(reading), #Phonetic length
        kanji_count,   #Visual complexity
        hiragana_count,
        katakana_count,
        kanji_count / len(word) if len(word) > 0 else 0 #Complexity ratio
    ]

X = np.array(df.apply(extract_features, axis=1).tolist())
y = df['JLPT Level']

#Training and Testing Split (Reserve 10% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

#Train model
model = RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

#Evaluate Results (SECTION D)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Total Samples: {len(df)}")
print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")
print(f"\nModel Accuracy on 10% Hold-Out: {accuracy:.2%}")
print("\nClassification Report (Section D):")
print(classification_report(y_test, y_pred))

#Save the model
joblib.dump(model, 'src/jlpt_model.pkl')
df.to_csv('data/jlpt_vocab_cleaned.csv', index=False)