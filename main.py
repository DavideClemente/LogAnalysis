import pandas as pd
from tensorflow.keras.layers import TextVectorization, Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential

# Load the data
traces = pd.read_csv('Raw_Logs/HDFS_v1/preprocessed/Event_traces.csv')
labels = pd.read_csv('Raw_Logs/HDFS_v1/preprocessed/anomaly_label.csv')

log_templates = pd.read_csv('Raw_Logs/HDFS_v1/preprocessed/HDFS.log_templates.csv')

data = traces.merge(labels, on='BlockId')

data = data[['Features', 'Label_x']]
data.rename(columns={'Label_x': 'Label'}, inplace=True)

data['Label'] = data['Label'].apply(lambda x: 1 if x == 'Fail' else 0)

print(data.head())

MAX_LEN = 50
VOCAB_SIZE = len(log_templates['EventId'].unique()) + 1

vectorizer = TextVectorization(
    max_tokens=VOCAB_SIZE,  # Set your desired vocabulary size
    output_mode='int',
    output_sequence_length=MAX_LEN  # Set your desired sequence length
)

vectorizer.adapt(data['Features'])

X = vectorizer(data['Features'])

print("Sample Encoded Sequence: ", X[0])

