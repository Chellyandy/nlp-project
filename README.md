# Project Description:
We would be utilizing data from Github README files to analyze and predict programming language based on the content.

# Project Goal:
The goal is to analyze and develop an model capable of predicting the primary programming language of a repository based on its content and structure. This will enable efficient categorization and organization of code repositories, aiding developers in discovering relevant projects and facilitating collaboration within the programming community.

# Data Dictionary:
![Alt text](https://github.com/Chellyandy/nlp-project/blob/main/data%20dictionary.numbers)

# Initial Thoughts/Questions:
# Project Plan:
- Acquire:
    - Acquired the data from github.com from the ["Most Forked Repositories"]([https://www.kaggle.com/datasets/meirnizri/covid19-dataset](https://github.com/search?o=desc&q=stars:%3E1&s=forks&type=Repositories))
    - Data was acquired as of June 27,2023.
    - Data was scraped and json file was created (data2.json).
    - 
- Prepare:
  - Removed non-ascii characters & turned all characters into lower case.
  - Removed stopwords
  - Created a new column with clean and lemmatized README content.
  - Created a bucket named 'other' to add all other languages that are not: JavaScript, Python, Java
- Exploration
- Modeling
- Deliverables:
# Steps to Reproduce:
# Summary/Takeaways:
# Key Findings, Recommendations and Next Steps:
