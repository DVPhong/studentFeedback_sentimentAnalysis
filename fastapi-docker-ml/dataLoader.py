import pandas as pd
from datasets import load_dataset
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import CountVectorizer

def loadDataset(dataset):
  data = {}
  data['train'] = pd.DataFrame(dataset['train'])
  data['train'].rename(columns={'sentiment':'label'}, inplace=True)

  data['val'] = pd.DataFrame(dataset['validation'])
  data['val'].rename(columns={'sentiment':'label'}, inplace=True)

  data['test'] = pd.DataFrame(dataset['test'])
  data['test'].rename(columns={'sentiment':'label'}, inplace=True)

  return data['train'], data['val'], data['test']

def remove_less_freq(df:pd.DataFrame, freq:int):
  freq_data = pd.Series(' '.join(df['sentence']).split()).value_counts()
  words_less_freq = freq_data[freq_data <= freq].index

  def filter_sentence(sentence:str):
    return ' '.join([word for word in sentence.split() if word not in words_less_freq])

  df['sentence'] = df['sentence'].apply(filter_sentence)
  return df

dataset = load_dataset("uitnlp/vietnamese_students_feedback")
dataset.remove_columns(['topic'])

train_data, val_data, test_data = loadDataset(dataset)
train_data = remove_less_freq(train_data, 5)
val_data = remove_less_freq(val_data, 5)
test_data = remove_less_freq(test_data, 5)

ros = RandomOverSampler(random_state=1)
X_resampled, y_resampled = ros.fit_resample(train_data['sentence'].values.reshape(-1, 1), train_data['label'])
resampled_data = pd.DataFrame({'sentence': X_resampled.flatten(), 'label': y_resampled})

vectorizer = CountVectorizer(max_df = 0.85,ngram_range=(1,2))
bow_train_features = vectorizer.fit_transform(resampled_data['sentence'])
bow_val_features = vectorizer.transform(val_data['sentence'])
bow_test_features = vectorizer.transform(test_data['sentence'])