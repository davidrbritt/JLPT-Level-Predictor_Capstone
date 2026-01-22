# JLPT Paragraph Difficulty Analyzer

This project is a machine learning powered Natural Language Processing (NLP) Dashboard meant to evaluate the difficulty level of Japanese text on the JLPT scale from N5 (Beginner) to N1 (Advanced). 

## Project Overview
This project addresses the issue of determining reading difficulty for Japanese language learners and educators. Unlike traditional dictionary-based lookups, this application uses a **Random Forest Classifier** to predict the difficulty of unknwn words which helps analyze a paragraph's difficulty as a whole.

### Key Features:
**Morphological Analysis:** Uses `Fugashi` (MeCab) to parse Japanese sentences into discrete parts of speech.

**Predictive Modeling:** Implements a TF-IDF character-level pipeline to classify unseen words.

**Advanced Visualizations:** Provides a "Known vs Predicted" pie chart, a bar chart of the word frequency by difficulty level, a threshold based difficulty estimation, and a detailed chart showing the known/predicted difficulty of each word detected.

**Heuristic Optimization:** Includes a Hiragana-veto system and threshold ladder to boost the accuracy of the final difficulty estimate.

---

## Installation and Setup

### 1. Prerequisites
* Python 3.9+
* **Git LFS:** This project uses Git Large File Storage for the 224 MB model file. Ensure Git LFS is installed before cloning.
    ```bash
    git lfs install
    ```
### 2. Clone the Repository

    git clone [https://github.com/davidrbritt/JLPT-Level-Predictor_Capstone.git](https://github.com/davidrbritt/JLPT-Level-Predictor_Capstone.git)
    cd JLPT-Level-Predictor_Capstone
    
### 3. Install Dependencies
    
    pip install -r requirements.txt
    
### 4. Run the Application
    
    streamlit run app.py

---
## Data Model and Performance

**Model Accuracy:** The Random Forest pipeline achieved an accuracy of ~42.41% on a balanced multi-class dataset.

**Descriptive Asset:** See assets/data_distribution.png for training data characteristics.

**Optimization:** The dashboard employs a layered approach to optimization. First the model runs then the Hiragana-veto heuristic (to handle unseen phonetic words), then finally the text as a whole is evaluated with the threshold ladder to refine the overall difficulty prediction.
---
## Project Structure

* app.py: The main streamlit dashboard application.

* requirements.txt: List of required Python libraries.

* src/: Contains the trained model jlpt_model.pkl (managed by Git LFS).

* data/: Contains the raw and cleaned JLPT vocabulary datasets.

* notebooks/: Contains source code for data analysis and model training (train_model.py, descriptive_analysis.py).

* docs/: Project documentation, including the Executive Summary and Transmittal Letter.

* screenshots/: Visual records of model iterations and the dashboard development.

* assets/: Images used within the application interface.

* .gitattributes: Configuration for Git LFS tracking.

## Development Progress

Visual records of the model's evolution, environment setup, and error resolution can be found in the `/screenshots` directory.