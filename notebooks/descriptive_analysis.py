import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

#####TASK C/D BAR CHART VISUAL

#Create an assets folder for report images if it does not exist
if not os.path.exists('assets'):
    os.makedirs('assets')

#Load data
df = pd.read_csv('data/jlpt_vocab.csv')

#Descriptive Analysis: Group by level
level_counts = df['JLPT Level'].value_counts().sort_index()

#Visualization: Bar Chart
plt.figure(figsize=(10, 6))
sns.barplot(x=level_counts.index, y=level_counts.values, hue = level_counts.index, palette='magma', legend = False)
plt.title('Vocabulary Distribution per JLPT Level')
plt.xlabel('JLPT Level (N5 = Easiest, N1 = Hardest)')
plt.ylabel('Number of Words')

#Save the Figure for written report
plt.savefig('assets/data_distribution.png')
print("Chart saved to assets/data_distribution.png")
plt.show()