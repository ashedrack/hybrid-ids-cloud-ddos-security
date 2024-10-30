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
