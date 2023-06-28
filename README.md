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

# Steps on How to Reproduce Project:
1. Access the nlp-project repository on GitHub.
2. Click on the "Code" button and select "Download ZIP" to download the entire repository to your computer. Extract the downloaded ZIP file to a directory of your choice, or you can copy the SSH code to your terminal.
3. isit https://github.com/settings/tokens and generate a personal access token by clicking on the "Generate new token" button. Make sure to leave all checkboxes unchecked to avoid selecting any scopes.
4. Add env.py to the nlp-project file on your computer.
5. Copy the generated personal access token and paste it into your env.py file under the variable github_token.
6. Similarly, add your GitHub username to your env.py file under the variable github_username.
7. Once you have saved all the necessary information in your env.py file, you can run the final notebook.
   
# Project Plan:
- Acquire:
    - Acquired the data from github.com by extracting the ["Most     Forked Repositories"].([https://www.kaggle.com/datasets/meirnizri/covid19-dataset](https://github.com/search?o=desc&q=stars:%3E1&s=forks&type=Repositories))
    - Data was collected as of June 27, 2023.
    - Data was scraped and a JSON file was created (data2.json).
    - Data Contains 180 repositories.
- Prepare:
  - Removed non-ASCII characters and converted all characters to lowercase.
  - Removed stopwords, tokenized, and lemmatized rows.
  - Created a new column with cleaned and lemmatized README content.
  - Created a bucket named 'other' to include all other languages that are not JavaScript, Python,Java, TypeScript, or     HTML.
  - Deleted extra words that were not relevant to the project
  - Split the data into train, validation, and test sets for exploration.
- Exploration:
    - Created visualizations and answered the following questions:
      1. Are there any significant differences in the word frequencies between README files of different programming             languages?
      2. Does the presence of specific libraries in the README file correlate with the programming language used?
      3. What are the most common words throughout the data frame and  per each language?
      4. What are the least common words throughout the data frame and  per each language?
- Modeling:
   - After vectorizing the words, we will use accuracy as our evaluation metric.
   - The baseline accuracy is 47.2%.
   - We employed Decision Tree Classifier, Random Forest, and K-Nearest Neighbor as our models for predicting                programming languages based on README content.

- Deliverables:
   - A five minutes slide presentation summarizing findings in exploration and results of modeling.
   
# Summary/Takeaways:


# Key Findings, Recommendations, and Next Steps:
