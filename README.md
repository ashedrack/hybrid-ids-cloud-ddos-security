# Hybrid Intrusion Detection System Using Machine Learning for Adaptive Cloud and DDoS Security

## Overview

This project is focused on developing a **Hybrid Intrusion Detection System (IDS)** that utilizes machine learning algorithms to detect various network intrusions, including Distributed Denial of Service (DDoS) attacks. The IDS combines signature-based detection with anomaly-based detection to improve detection rates of both known and zero-day attacks.

The code includes several key stages: data preprocessing, feature selection, model training (with Decision Trees and Random Forest), hyperparameter tuning using GridSearchCV, and model evaluation. 


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
