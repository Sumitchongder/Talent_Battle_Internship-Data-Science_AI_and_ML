# <ins>Talent Battle Internship - Data-Science, AI, and ML</ins>

### --> Please refer <ins>**Twitter Sentiment Analysis.ipynb**</ins> file for the code [Jupyter Notebook File]

### --> Please refer <ins>**Literature Survey.md**</ins> file for the Literature Survey Documentation and Outcomes

## <ins>Steps to perform Sentiment Analysis on Twitter Sentiment Social Dataset:</ins>

### 1. Data Preprocessing:
(i) Loaded the Twitter Sentiment Social dataset using Pandas.

![1](https://github.com/Sumitchongder/Talent_Battle_Internship-Data-Science_AI_and_ML/assets/77958774/641345a2-68e8-4f5b-b8aa-c7b40db4a5f0)

(ii) Checked the shape and info of the dataset to understand its structure and information.

![2](https://github.com/Sumitchongder/Talent_Battle_Internship-Data-Science_AI_and_ML/assets/77958774/3ddf3853-f1e0-40b3-91c5-85252b44e138)

(iii) Visualized the distribution of sentiment labels ('positive' and 'negative') using Seaborn.

![3](https://github.com/Sumitchongder/Talent_Battle_Internship-Data-Science_AI_and_ML/assets/77958774/4541bd7d-d11d-4361-ba22-7c88688d6056)

(iv) Reviewed the first few reviews and their corresponding sentiments to get a sense of the data.

![7](https://github.com/Sumitchongder/Talent_Battle_Internship-Data-Science_AI_and_ML/assets/77958774/83fe4e7c-8f54-41cb-9820-d8d9fef996aa)

### 2. Text Preprocessing:
(i) Created a function to count the number of words in each review.

(ii) Cleaned and preprocessed the text data by converting it to lowercase, removing HTML tags, URLs, special characters, and stop words.

![8](https://github.com/Sumitchongder/Talent_Battle_Internship-Data-Science_AI_and_ML/assets/77958774/5fae7b93-6fdf-4fd7-8db1-c162ddd8331c)

(iii) Applied stemming to reduce words to their root form.

![4](https://github.com/Sumitchongder/Talent_Battle_Internship-Data-Science_AI_and_ML/assets/77958774/d343e6a9-9cff-4bcf-820c-6d5a6c85bf6e)

### 3. Exploratory Data Analysis (EDA):
(i) Analyzed the distribution of the number of words in reviews for both positive and negative sentiments.

![5](https://github.com/Sumitchongder/Talent_Battle_Internship-Data-Science_AI_and_ML/assets/77958774/1fed3f23-7d61-40fc-b8f4-c8d503a4cbe2)

![6](https://github.com/Sumitchongder/Talent_Battle_Internship-Data-Science_AI_and_ML/assets/77958774/288e7936-058f-492f-9622-aae18f116ece)

(ii) Visualized common words in positive and negative reviews using word clouds.

![9](https://github.com/Sumitchongder/Talent_Battle_Internship-Data-Science_AI_and_ML/assets/77958774/e833439d-2b75-4656-a8cf-06ea4e1d8aae)

![10](https://github.com/Sumitchongder/Talent_Battle_Internship-Data-Science_AI_and_ML/assets/77958774/ae4c5ed9-2643-436a-84aa-263bb427b8a3)

(iii) Extracted and visualized the most common words in both positive and negative reviews.

### 4. Feature Extraction:
(i) Used TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert text data into numerical form.

![11](https://github.com/Sumitchongder/Talent_Battle_Internship-Data-Science_AI_and_ML/assets/77958774/fa60ec5b-88f8-40e8-914f-20a67029dd1b)

### 5. Model Building:
(i) Split the dataset into training and testing sets.

(ii) Trained a Multinomial Naive Bayes (MNB) classifier as your baseline model and evaluated its accuracy.

![12](https://github.com/Sumitchongder/Talent_Battle_Internship-Data-Science_AI_and_ML/assets/77958774/16d9a1d0-a273-43b4-a25e-f5c6f7b925f5)

### 6. Deep Learning Models:
(i) Built Deep learning model: a Feedforward Neural Network (FFNN).

![13](https://github.com/Sumitchongder/Talent_Battle_Internship-Data-Science_AI_and_ML/assets/77958774/a4a49089-998e-408b-958a-84397bda4155)

(ii) Used the TF-IDF vectors as input features.

(iii) Compiled and trained each model with appropriate architecture and hyperparameters.

(iv) Evaluated the performance of each deep learning model on the test dataset.

![14](https://github.com/Sumitchongder/Talent_Battle_Internship-Data-Science_AI_and_ML/assets/77958774/e8a617f3-0875-4179-abd4-1cbc71346e46)

(v) Plotted the training loss and accuracy curves for the FNN model.

![15](https://github.com/Sumitchongder/Talent_Battle_Internship-Data-Science_AI_and_ML/assets/77958774/2c0e242e-7639-449c-b3a0-c787e41121f0)

![16](https://github.com/Sumitchongder/Talent_Battle_Internship-Data-Science_AI_and_ML/assets/77958774/32bf3fd6-69ed-41d5-9418-24201a3dc751)

### 7. Graphical User Interface Implementation (GUI):
(i) Enter a sentence to predict the sentiment.

![17](https://github.com/Sumitchongder/Talent_Battle_Internship-Data-Science_AI_and_ML/assets/77958774/db272288-0b88-401a-9fcd-450118aabd9f)

(ii) The output will be shown in the pop-up window stating if the sentiment is "positive" or "negative".

![18](https://github.com/Sumitchongder/Talent_Battle_Internship-Data-Science_AI_and_ML/assets/77958774/95a82fcd-c3ab-4dfa-81c7-1e1a36c3357e)

