import csv
import pandas as pd 
import numpy as np
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore') # Never print matching warnings
df = pd.read_excel('N:/sem4/machine learning/project/data_set.xlsx')
print(df)

df.head()
data = df.fillna(method='ffill')
data.head()
def process_data(data):
    data_list = []
    data_name = data.replace('^', '_').split('_')
    n=1
    for names in data_name:
        if (n%2 == 0):
            data_list.append(names)
        n+=1
    return data_list
disease_list = []
disease_symptom_dict = defaultdict(list)
disease_symptom_count = {}
count = 0

for idx, row in data.iterrows():
    
    # Get the Disease Names
    if (row['Disease'] !="\xc2\xa0") and (row['Disease'] != ""):
        disease = row['Disease']
        disease_list = process_data(data=disease)
        count = row['Count of Disease Occurrence']

    # Get the Symptoms Corresponding to Diseases
    if (row['Symptom'] !="\xc2\xa0") and (row['Symptom'] != ""):
        symptom = row['Symptom']
        symptom_list = process_data(data=symptom)
        for d in disease_list:
            for s in symptom_list:
                disease_symptom_dict[d].append(s)
            disease_symptom_count[d] = count
# Save cleaned data as CSV
f = open('N:/sem4/machine learning/project/cleaned_data.csv', 'w')

with f:
    writer = csv.writer(f)
    for key, val in disease_symptom_dict.items():
        for i in range(len(val)):
            writer.writerow([key, val[i], disease_symptom_count[key]])
# Read Cleaned Data as DF
df = pd.read_csv('N:/sem4/machine learning/project/cleaned_data.csv')
df.columns = ['disease', 'symptom', 'occurence_count']
df.head()

df.replace(float('nan'), np.nan, inplace=True)
df.dropna(inplace=True)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(df['symptom'])
print(integer_encoded)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)

cols = np.asarray(df['symptom'].unique())
df_ohe = pd.DataFrame(columns = cols)
for i in range(len(onehot_encoded)):
    df_ohe.loc[i] = onehot_encoded[i]
df_disease = df['disease']
df_concat = pd.concat([df_disease,df_ohe], axis=1)
df_concat.drop_duplicates(keep='first',inplace=True)
cols = df_concat.columns
cols = cols[1:]
# Since, every disease has multiple symptoms, combine all symptoms per disease per row
df_concat = df_concat.groupby('disease').sum()
df_concat = df_concat.reset_index()

df_concat.to_csv("N:/sem4/machine learning/project/training_dataset.csv", index=False)
# One Hot Encoded Features
X = df_concat[cols]

# Labels
y = df_concat['disease']    

from sklearn.metrics import classification_report, plot_confusion_matrix
def performance_of_classifier(classifier,clf,X_test,y_test,y_pred):
    print("---------------PERFORMANCE ANALYSIS FOR {} CLASSIFIER----------------\n".format(clf))

    print("Real Test dataset labels: \n{}\n".format(y_test))
    print("Predicted Test dataset labels: \n{}\n".format(y_pred))

    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(classifier, X_test, y_test,cmap=plt.cm.Blues)  
    plt.show()
    
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
dt = DecisionTreeClassifier()
clf_dt=dt.fit(X, y)
clf_dt.score(X, y)
y_pred = clf_dt.predict(X_test)
performance_of_classifier(dt,"DT",X_test,y_test,y_pred)


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb = gnb.fit(X,y)
gnb.score(X,y)
y_pred = gnb.predict(X_test)
performance_of_classifier(gnb,"NB",X_test,y_test,y_pred)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
clf_knn = knn.fit(X,y)
clf_knn.score(X,y)
y_pred = clf_knn.predict(X_test)
performance_of_classifier(knn,"KNN",X_test,y_test,y_pred)

from sklearn import svm
svm_c =svm.SVC()
clf_svm = svm_c.fit(X,y)
clf_svm.score(X,y)
y_pred = clf_svm.predict(X_test)
performance_of_classifier(svm_c,"SVM",X_test,y_test,y_pred)

disease_pred = clf_dt.predict(X)
disease_real = y.values
for i in range(0, len(disease_real)):
    if disease_pred[i]!=disease_real[i]:
        print ('Pred: {0}\nActual: {1}\n'.format(disease_pred[i], disease_real[i]))
        
        
        