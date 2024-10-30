# Hybrid Intrusion Detection System Using Machine Learning for Adaptive Cloud and DDoS Security

## Overview

This project is focused on developing a **Hybrid Intrusion Detection System (IDS)** that utilizes machine learning algorithms to detect various network intrusions, including Distributed Denial of Service (DDoS) attacks. The IDS combines signature-based detection with anomaly-based detection to improve detection rates of both known and zero-day attacks.

The system will utilize the CICIDS 2017 dataset, which is publicly available and widely used for evaluating intrusion detection systems. This dataset contains real-world attack scenarios, such as Distributed Denial of Service (DDoS), brute force attacks, and infiltration, simulating the behavior of both normal and malicious network traffic across various protocols. The dataset is organized into eight separate files, each corresponding to a different day of observation, capturing diverse attack vectors and benign traffic. Both PCAP and CSV formats are provided, enabling comprehensive traffic analysis and feature extraction.
The dataset is publicly accessible via the following  Canadian Institute for Cybersecurity- dataset description page https://www.unb.ca/cic/datasets/ids-2017.html, and the archive used for training can be downloaded here -  Dataset Record Form.
Key insights from the CICIDS 2017 dataset show that it contains diverse attack traces, including DoS, Web Attacks, Infiltration, and DDoS, covering both benign and malicious traffic. The dataset, with its various protocols like HTTP, FTP, SSH, and email, provides a comprehensive foundation for detecting multi-protocol attacks in cloud environments.

### 2. Dataset Preparation

Ensure that the dataset files are located in the project directory. The code expects files such as:

- `Monday-WorkingHours.pcap_ISCX.csv`
- `Tuesday-WorkingHours.pcap_ISCX.csv`
- ... and other related traffic data files.

The code combines all these datasets into a single DataFrame for easier processing.

### 3. Running the Code

#### Data Loading and Cleaning

The first step is to load and clean the data. You can run the following section of the code to read and combine the datasets in chunks (for memory efficiency) and save it as `combined_dataset.csv`:

```python
csv_files = ["Monday-WorkingHours.pcap_ISCX.csv", ...]  # List of CSVs
chunk_size = 50000
combined_chunks = []

for file in csv_files:
    for chunk in pd.read_csv(file, engine='python', encoding='ISO-8859-1', chunksize=chunk_size):
        combined_chunks.append(chunk)

combined_df = pd.concat(combined_chunks, ignore_index=True)
combined_df.to_csv('combined_dataset.csv', index=False)
```

#### Data Preprocessing

Next, the data is cleaned by removing unnecessary columns, handling missing values, and encoding categorical features:

```python
combined_df = combined_df.drop(columns=['Fwd Header Length.1'])  # Dropping irrelevant columns
combined_df.replace([np.inf, -np.inf, np.nan], -1, inplace=True)  # Handling missing/infinite values

# Label Encoding for categorical features
le = preprocessing.LabelEncoder()
combined_df[string_features] = combined_df[string_features].apply(lambda col: le.fit_transform(col))
```

#### Address Class Imbalance

To address class imbalance between benign and malicious traffic, undersampling is applied:

```python
benign_included_max = attack_total / 30 * 70  # 70% benign, 30% attacks
benign_inc_probability = (benign_included_max / benign_total) * enlargement

indexes = []
for index, row in combined_df.iterrows():
    if row['Label'] != "BENIGN":
        indexes.append(index)
    else:
        if random.random() > benign_inc_probability: continue
        if benign_included_count > benign_included_max: continue
        benign_included_count += 1
        indexes.append(index)

df_balanced = combined_df.loc[indexes]
df_balanced.to_csv("web_attacks_balanced.csv", index=False)
```

#### Model Training

You can train a **Decision Tree** and **Random Forest** classifier with the following code:

```python
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(max_leaf_nodes=5, random_state=0)
decision_tree.fit(X_train, y_train)

# For Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=250, random_state=42)
rf.fit(X_train, y_train)
```

#### Hyperparameter Tuning

Hyperparameter tuning can be performed using `GridSearchCV` or `RandomizedSearchCV`:

```python
from sklearn.model_selection import GridSearchCV

parameters = {
    'n_estimators': [10],
    'min_samples_leaf': [3],
    'max_features': [3],
    'max_depth': [3, 5, 10, 20]
}

gcv = GridSearchCV(rfc, parameters, scoring='f1', cv=5, return_train_score=True, n_jobs=-1)
gcv.fit(X_train, y_train)
```

#### Model Evaluation

Evaluate the model's performance using confusion matrices, accuracy, precision, recall, and F1-score:

```python
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
```

#### Visualizing the Decision Tree

Visualize the decision tree structure:

```python
from graphviz import Source
from sklearn import tree

dot_data = tree.export_graphviz(decision_tree, out_file=None, feature_names=X_train.columns, filled=True)
graph = Source(dot_data)
graph.render("decision_tree")  # Save the graph
```

#### Feature Importance

To identify the most important features contributing to intrusion detection:

```python
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

for index, i in enumerate(indices[:10]):
    print(f'{index + 1}.\tFeature {i}\tImportance {importances[i]:.3f}')
```

#### Model Persistence

To save the trained model for future use:

```python
import pickle
with open('webattack_detection_rf_model.pkl', 'wb') as f:
    pickle.dump(rf, f)
```

---


## Future Work

- **Expand Feature Set**: Additional features like flow-based statistics can be added to improve detection accuracy.
- **Real-Time Detection**: Extend the system to detect intrusions in real-time cloud environments.
- **Model Deployment**: The trained model can be deployed as part of a microservice to monitor network traffic in real time.

---

## Conclusion

This project provides a robust foundation for building a hybrid intrusion detection system using machine learning. The system is designed to detect various types of web attacks, including zero-day vulnerabilities, in cloud environments while balancing accuracy and resource efficiency.
