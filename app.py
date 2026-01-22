import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from fugashi import Tagger

#Function for model to unpickle correctly
def get_manual_features(df):
    feats = []
    for _, row in df.iterrows():
        word = str(row['Original'])
        reading = str(row['Furigana'])
        kanji_count = sum(1 for char in word if '\u4e00' <= char <= '\u9faf')
        feats.append([
            len(word),
            len(reading),
            kanji_count, 
            kanji_count/len(word) if len(word) > 0 else 0
            ])
    return np.array(feats)

#Initialize NLP and Model
tagger = Tagger('-Owakati')

@st.cache_resource
def load_assets():
    model = joblib.load('src/jlpt_model.pkl')
    #Load the cleaned dataset globally for get_detailed_status()
    df = pd.read_csv('data/jlpt_vocab_cleaned.csv')
    return model, df

model, df = load_assets()

st.title("JLPT Paragraph Difficulty Analyzer")

#User Input
user_text = st.text_area("Paste Japanese text/paragraph here:", height=200)

if st.button("Analyze Text"):
    if user_text:
        #Tokenization and feature extraction
        words = []
        for word in tagger(user_text):
            if word.feature.pos1 in ['名詞', '動詞', '形容詞']:
                words.append({
                    'Original': word.surface,
                    'Furigana': word.feature.kana if word.feature.kana else word.surface
                })
        
        if words:
            #Bulk prediction
            current_batch_df = pd.DataFrame(words)
            
            #Logic to tag words as Known in the dataset or predicted
            def get_detailed_status(row):
                #Check for Kanji in the word
                has_kanji = any('\u4e00' <= char <= '\u9faf' for char in str(row['Original']))

                match = df[df['Original'] == row['Original']]
                if not match.empty:
                    return f"{match.iloc[0]['JLPT Level']} (Known)"
                else:
                    #Model predicts level for unseen words
                    pred = model.predict(pd.DataFrame([row]))[0]

                    #Heuristic: If model predicts a word as N1/N2 but it has no kanji
                    #override to N5 since there are almost no N1/N2 words without kanji and very few N3.
                    if not has_kanji and pred in ['N1', 'N2', 'N3']:
                        return "N5 (Predicted - Phonetic)"
                    return f"{pred} (Predicted)"
                
            current_batch_df['Status'] = current_batch_df.apply(get_detailed_status, axis=1)


            #Data for graphics
            status_counts = current_batch_df['Status'].value_counts().reset_index()
            status_counts.columns = ['Level Status', 'Count']

            #Predicted Graphic Pie Chart
            st.subheader("Predicted vs. Known Level Distribution")
            
            #Define consistent color map
            color_map = {
        'N1 (Known)': '#8B0000', 'N1 (Predicted)': '#FF4B4B',
        'N2 (Known)': '#E65100', 'N2 (Predicted)': '#FFA726',
        'N3 (Known)': '#FBC02D', 'N3 (Predicted)': '#FFF176',
        'N4 (Known)': '#2E7D32', 'N4 (Predicted)': '#81C784',
        'N5 (Known)': '#1565C0', 'N5 (Predicted)': '#64B5F6'
            }
            fig_pie = px.pie(
                status_counts,
                values='Count',
                names='Level Status',
                color='Level Status',
                color_discrete_map=color_map,
                hole = 0.5, #Donut Style for easier readability
                category_orders={"Level Status": [
                    "N1 (Known)", "N1 (Predicted)", "N2 (Known)", "N2 (Predicted)",
                    "N3 (Known)", "N3 (Predicted)", "N4 (Known)", "N4 (Predicted)",
                    "N5 (Known)", "N5 (Predicted)"
                ]}
            )

            #Add labels inside the slices
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

            #Descriptive graphic Bar Chart
            st.subheader("Word Frequency by Level")
            st.bar_chart(status_counts.set_index('Level Status').sort_index())

            #Final Metric - OLD (Often rated high level material as N5 due to common words still being used in difficult text)
            #dominant_level = status_counts.loc[status_counts['Count'].idxmax(), 'Level Status']
            #st.metric("Estimated Overall Difficulty", dominant_level)

            #Final Metric - NEW (Uses a threshold to determine difficulty of text as a whole)
            #Extract the base level Eg N1, N2, ... regardless of known/predicted
            current_batch_df['Base_Level'] = current_batch_df['Status'].str.extract(r'(N\d)')
            #Calculate percentages of each level
            level_distribution = current_batch_df['Base_Level'].value_counts(normalize=True) * 100
            #Apply Threshold Ladder Logic
            if level_distribution.get('N1', 0) >=8: 
                final_grade = "N1 (Advanced)"
                note = "Detected significant N1 vocabulary (over 8%)."
            elif level_distribution.get('N2', 0) >= 12:
                final_grade = "N2 (Upper Intermediate)"
                note = "Detected significant N2 markers."
            elif level_distribution.get('N3', 0) >= 20:
                final_grade = "N3 (Intermediate)"
                note = "Text consists of primarily intermediate vocabulary."
            elif level_distribution.get('N4', 0) >= 25:
                final_grade = "N4 (Elementary)"
                note = "Text shows signs of N4 difficulty level."
            else:
                final_grade = "N5 (Beginner)"
                note = "Text consists of almost entirely foundational vocabulary."

            #Display New Final Metric
            st.metric("Estimated Overall Difficulty", final_grade)
            st.info(f"**Analysis Note:** {note}")

            #Detailed Data Table
            st.divider()
            st.subheader("Detailed Vocabulary Analysis")
            st.write("This table shows the specific classification for every content word found.")
            st.dataframe(current_batch_df[['Original', 'Furigana', 'Status']], use_container_width=True)

        else:
            st.error("Please enter text containing valid Japanese nouns, verbs, or adjectives.")

#SECTION C/D Requirements
st.divider()
if st.checkbox("Show model Training Metadata"):
    st.write("Current Model Accuracy: 42.41%")
    st.image("assets/data_distribution.png", caption="Training Dataset Baseline")