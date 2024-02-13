import os
from openai import AzureOpenAI
import pandas as pd
import numpy as np
import data_extract
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from ast import literal_eval

client = AzureOpenAI(api_key=os.getenv("AZURE_OPENAI_KEY"),
                api_version="2023",
                    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
                    )

# Reading the input files
final_df =  pd.read_csv("input.csv")
unique_labels = final_df['label'].unique()  
print(f"shape of dataframe is {final_df.shape}")

#Details of the openai models and creating embedding for the inputs
#Model name keep on changing, so use the latest one to get the best result
deployment_name = "text-embedding-ada-002"
def get_embedding(text, model=deployment_name):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

#Finding the total number of tokens involved

#Applying embeeding to the text column and saving it in csv format.
final_df['ada_embedding'] = final_df['text'].apply(lambda x: get_embedding(x, model=deployment_name))
final_df.to_csv('/mnt/nfs/research/projects/manun/Git_Experiemnt/experiments/projects/openai_embedding/output/output.csv', index=False)

# reading transcript text & labels
embedded_df = pd.read_csv("./output/output.csv")

#**************************************************
########## Classification #########################
###################################################

clf = RandomForestClassifier(n_estimators=100)
embedded_df["ada_embedding"] = embedded_df.ada_embedding.apply(literal_eval).apply(np.array)
X = np.stack(embedded_df["ada_embedding"].values)
#converting labels to integer values
le = LabelEncoder()
encoded_labels = le.fit_transform(embedded_df['label'])
#Training the model
clf.fit(X, encoded_labels)


#Testing the perfromance of the model with a random output
test_input = "And are you happy for me to run your whatsapp chat? Yeah, it's fine."
embedded_value = get_embedding(test_input, deployment_name)
embedded_value_array = np.array(embedded_value)
embedded_value_2d = embedded_value_array.reshape(1, -1)

preds = clf.predict(embedded_value_2d)
predicted_labels = le.inverse_transform(preds)
probas = clf.predict_proba(embedded_value_2d)

