# Project Description:
We would utilize data from GitHub README files to analyze and predict programming language based on the content.

# Project Goal:
The goal is to analyze and develop a model capable of predicting the primary programming language of a repository based on its content and structure. This will enable efficient categorization and organization of code repositories, aiding developers in discovering relevant projects and facilitating collaboration within the programming community.

# Data Dictionary:
![Alt text](https://github.com/Chellyandy/nlp-project/blob/main/data%20dictionary.png)

# Initial Thoughts/Questions:
1. Are there any significant differences in the word frequencies between README files of different programming languages?
2. Does the presence of specific libraries in the README file correlate with the programming language used?
3. What are the most common words throughout the data frame and  per each language?
4. What are the least common words throughout the data frame and  per each language?
   
# Project Plan:
- Acquire:
    - Acquired the data from github.com from the ["Most Forked Repositories"]([https://www.kaggle.com/datasets/meirnizri/covid19-dataset](https://github.com/search?o=desc&q=stars:%3E1&s=forks&type=Repositories))
    - Data was acquired as of June 27, 2023.
    - Data was scraped and a JSON file was created (data2.json).
    - 
- Prepare:
  - Removed non-ASCII characters & turned all characters into lowercase.
  - Removed stopwords, tokenized and lemmatized rows.
  - Created a new column with clean and lemmatized README content.
  - Created a bucket named 'other' to add all other languages that are not: JavaScript, Python, Java
  - Split the data into train, validate, and test for exploration
- Exploration:
    - Created visualizations and answered the following questions:
      1. Are there any significant differences in the word frequencies between README files of different programming languages?
      2. Does the presence of specific libraries in the README file correlate with the programming language used?
      3. What are the most common words throughout the data frame and  per each language?
      4. What are the least common words throughout the data frame and  per each language?
- Modeling:
- Deliverables:
# Steps to Reproduce:
# Summary/Takeaways:
# Key Findings, Recommendations, and Next Steps:
