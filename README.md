# Classification-using-OpenAI-embedding
Embedding is a more regular concept in NLP, where every text in the input are converted into the numerical format. To be more specific the embedding model maps the text into multi diamensional vector space. The numbers iutputed by the model are the text's location in the space.

More similar words will be placed closer together and Dissimilar words appears further away in the space. For example, word teacher and student will be placed closer together. and Student and Rice are placed further away in the space.


This ability of grouping of words means that embedding can be used to extract the semantic meaning. For example,
1. Which way is it to the supermarket ?
2. Could I have the direction to the shop?

both of the above sentence are semantically similar but structrely different. An embedding model will analyse the semantic similarity and return the similar output in both scenario.

How are the embedding model differe from the traditional model ?

Traditional model uses the Keyword matching pattern. for example if we query "comfortable running shoes", it analyses the keyword in the query and it may return the following:
1. Comfortable running shorts.
2. Running shoes for kids.
3. Top 10 running routes in New York city

if we see the above example then we can see that the query return entire return results. In the first output instance it used the keyword "comfortable running" from query but it doesn't meets our criteria beacuse we searched for shoes.


I have followed the following point while doing this experiment.

1. Choose an OpenAI Model for embedding.
  There are seveal choices among the OpenAI models like GPT, GPT-2, GPT-3.5, GPT-4, ADA etc.
  Here I have chosen ADA model for embedding purposes.

2. Initialize the OpenAI Model
  Use the OpenAI API to initialize the chosen model. Here I am using the Azure Open AI services to intitialize the service.
  Please refer the code for more details.

3. Tokenize and Embed the Text Data
  Use the initialized model to generate embeddings for each tokenized text. Concatenate or average the embeddings for each text to obtain a single vector representation for the entire text.

4. Train a Classifier
  Split your dataset into training and testing sets. Choose a classifier suitable for your task, such as K-Nearest Neighbors (KNN), Support Vector Machines (SVM), or Neural Networks. I have chosen Random forest classifer. It was supposed give a     good accuracy.Train the classifier using the embedded representations of your training data.

5. Evaluate the Classifier
  Test the classifier using the embedded representations of your testing data. Evaluate the performance of the classifier using metrics such as accuracy, precision, recall, and F1-score.

6. Make Predictions
  Once your classifier is trained and evaluated, you can use it to make predictions on new, unseen text data. Embed the new text data using the same process described above. Feed the embedded representations into your trained classifier to classify the new data based on similarity. 
