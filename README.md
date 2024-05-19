# PTB-XL-Binary-Classification-with-CNN-DeepLearning
## Binary classification of "IMI"  Inferior Myocardial Infarction

### 1. Download the PTB-XL DataSet
[PTB-XL DataSet](https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip) - Download link
### 2.Import Libraries 
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wfdb
import os
from scipy.signal import find_peaks, medfilt, butter, sosfilt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,roc_curve, roc_auc_score, classification_report, f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import he_normal
```
### 3. Load data CSV file
```
data=pd.read_csv(r"C:\Users\shada\Desktop\project trying my best part 2\data\ptbxl_database.csv") #add path to csv file
data.head()
```

<div>
<img src="https://github.com/shadagamal/ptb_xl-binary-classification-with-cnn-deeplearning/blob/main/output%20photos/Screenshot%202024-05-19%20070346.png?raw=true"width=600 heigth=600>   
</div>


### 4. Making a separate DataFrame that contains specific columns "desired columns" and extract the first key of the 'scp_codes' column 
```
desired_columns = ['patient_id', 'scp_codes', 'filename_hr','baseline_drift','static_noise','burst_noise','electrodes_problems','extra_beats']  # Replace these with the actual column names you want
separate_data = data[desired_columns].copy()
df = pd.DataFrame(separate_data)

#making a function to extract first key of the 'scp_codes' column as it contains the diagnose 
def extract_first_key(scp_codes):
    scp_dict = eval(scp_codes)
    first_key = next(iter(scp_dict.keys()), None)
    return first_key

# Apply function to create new column with first key
df['scp_codes'] = df['scp_codes'].apply(extract_first_key)

print(df)
```


### 5. Only take 'IMI' disease 
#### We group each disease then take all records with 'IMI' and equal parts of 'NORM'
```
grouped = df.groupby(df['scp_codes'])
X_data = []
for scp_codes in ['NORM', 'IMI']:
    X_data.append(grouped.get_group(scp_codes).sample(n=2317)) 
result = pd.concat(X_data) 
print(result)
print(len(result))
```

### 6. encode the target value so it can be processed using CNN model

```
label_map = {'IMI': 0, 'NORM': 1}

# Applying the map function to the 'scp_codes' column
result_filtered['scp_codes'] = result_filtered['scp_codes'].map(label_map)

# Printing the modified DataFrame
print(result_filtered)
```

### 7. Preprocessing (normalization, r_peak detection, filters applying, beat splitting, and saving functions)
```
def detect_r_peaks(ecg_signal):
    r_peaks, _ = find_peaks(ecg_signal, height=0, distance=50)
    return r_peaks

def segment_beats(ecg_signal, r_peaks, max_length):
    beats = []
    for i in range(len(r_peaks) - 1):
        beat_start = r_peaks[i]
        beat_end = r_peaks[i + 1]
        beat = ecg_signal[beat_start:beat_end]
        if len(beat) > max_length:
            beat = beat[:max_length]
        elif len(beat) < max_length:
            beat = np.pad(beat, (0, max_length - len(beat)), 'constant')
        beats.append(beat)
    return beats

def z_score_normalize(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    normalized_signal = (signal - mean) / std
    return normalized_signal

def process_ecg_record(record_path, max_length):
    record = wfdb.rdrecord(record_path)
    num_leads = record.p_signal.shape[1]
    
    # Use the first lead as the reference lead to detect R-peaks
    lead_signal_ref = record.p_signal[:, 0]

    # Apply median filter
    filtered_signal_median_ref = medfilt(lead_signal_ref, kernel_size=3)

    # Apply high-pass filter to correct baseline drift
    sos_ref = butter(4, 0.5, 'highpass', fs=record.fs, output='sos')
    filtered_signal_ref = sosfilt(sos_ref, filtered_signal_median_ref)

    # Apply Z-score normalization
    normalized_signal_ref = z_score_normalize(filtered_signal_ref)

    # Detect R-peaks on the normalized signal of the reference lead
    r_peaks_indices = detect_r_peaks(normalized_signal_ref)

    all_beats = [[] for _ in range(num_leads)]

    for i in range(num_leads):
        lead_signal = record.p_signal[:, i]

        # Apply median filter
        filtered_signal_median = medfilt(lead_signal, kernel_size=3)

        # Apply high-pass filter to correct baseline drift
        sos = butter(4, 0.5, 'highpass', fs=record.fs, output='sos')
        filtered_signal = sosfilt(sos, filtered_signal_median)

        # Apply Z-score normalization
        normalized_signal = z_score_normalize(filtered_signal)

        # Segment the ECG signal into beats using the R-peaks from the reference lead
        beats = segment_beats(normalized_signal, r_peaks_indices, max_length)

        all_beats[i].extend(beats)

    # Combine beats from all leads into a single array with shape (num_beats, max_length, num_leads)
    num_beats = len(all_beats[0])
    combined_beats = np.zeros((num_beats, max_length, num_leads), dtype=np.float32)
    for i in range(num_leads):
        combined_beats[:, :, i] = all_beats[i][:num_beats]

    return combined_beats
#a code to itrate over each record and applying the above functions then saving them in array
def process_ecg_records_from_dataframe(dataframe, max_length):
    all_records = []
    all_labels = []
    for idx, row in dataframe.iterrows():
        record_path = row['filename_hr']
        label = row['scp_codes']  # Assuming 'scp_codes' is the label column
        print(f"Processing record: {record_path}")
        all_records.append(process_ecg_record(record_path, max_length))
        all_labels.append(label)
    return all_records, all_labels
# creat a dataset to make it easier to split into train, validation and test
def create_dataset(ecg_records, labels):
    X = []
    y = []
    for record, label in zip(ecg_records, labels):
        X.extend(record)  # No need to stack, since each beat is already processed
        y.extend([label] * record.shape[0])  # Assign the label to each beat
    X = np.array(X)
    y = np.array(y)
    return X, y
```


<div>
<img src="https://github.com/shadagamal/ptb_xl-binary-classification-with-cnn-deeplearning/blob/main/output%20photos/Screenshot%202024-05-19%20072906.png?raw=true"width=600 heigth=600>   
</div>


### 8. Applying all the above functions 
```
df = result_filtered  
# Define the maximum length for beats
max_length = 500

# Split the DataFrame into train(70%), validation(15%), and test(15%) sets
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Process all ECG records listed in each subset
print("Processing Training Set")
train_records, train_labels = process_ecg_records_from_dataframe(train_df, max_length)

print("Processing Validation Set")
val_records, val_labels = process_ecg_records_from_dataframe(val_df, max_length)

print("Processing Test Set")
test_records, test_labels = process_ecg_records_from_dataframe(test_df, max_length)

# Create datasets for training, validation, and testing
X_train, y_train = create_dataset(train_records, train_labels)
X_val, y_val = create_dataset(val_records, val_labels)
X_test, y_test = create_dataset(test_records, test_labels)

# Reshape the data to fit the CNN input shape
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")
```
### 9. Creating CNN model
```

num_classes=1 # binary classification 
# Define the CNN model
model = Sequential([
    Conv1D(filters=32, kernel_size=3, input_shape=(500, 12),kernel_initializer=he_normal()),
    LeakyReLU(alpha=0.01), 
    MaxPooling1D(pool_size=2),
    Conv1D(filters=64, kernel_size=3),
    LeakyReLU(alpha=0.01),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=128, kernel_size=3),
    LeakyReLU(alpha=0.01),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=256, kernel_size=3),
    LeakyReLU(alpha=0.01),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=512, kernel_size=3),
    LeakyReLU(alpha=0.01),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128),
    LeakyReLU(alpha=0.01),
    Dropout(0.5),
    Dense(num_classes, activation='sigmoid')  # Binary classification
])

model.summary()
```
<div>
<img src="https://github.com/shadagamal/ptb_xl-binary-classification-with-cnn-deeplearning/blob/main/output%20photos/Screenshot%202024-05-19%20072102.png?raw=true"width=500 heigth=500>   
</div>

### 10. Train the model
```
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Define the callbacks
callbacks = [
    ModelCheckpoint('best_model.h4', monitor='val_loss', save_best_only=True, mode='min', verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
]

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=128, validation_data=(X_val, y_val),
                    callbacks=callbacks)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
```
### 11. Predict the labels for the test set and computing the confusion matrix, sensitivity, specificity, F1 score, precision, recall, plotting the ROC curve with AUC, and printing the classification report.

```
# Predict the labels for the test set
y_pred_probs = model.predict(X_test)  # Probabilities for the positive class
y_pred_classes = (y_pred_probs > 0.5).astype("int32")  # Binary predicted class labels

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)


# Plot the confusion matrix
disp = ConfusionMatrixDisplay(conf_matrix)
disp.plot(cmap=plt.cm.Blues) 
print('Confusion Matrix:')
print(conf_matrix)

# Extracting true positives, false positives, true negatives, and false negatives
tn, fp, fn, tp = conf_matrix.ravel()

# Calculate sensitivity (recall)
sensitivity = tp / (tp + fn)
print(f'Sensitivity: {sensitivity}')

# Calculate specificity
specificity = tn / (tn + fp)
print(f'Specificity: {specificity}')

# Calculate F1 score, precision, and recall
f1 = f1_score(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes)
recall = recall_score(y_test, y_pred_classes)

# Print the F1 score, precision, and recall
print(f'F1 Score: {f1}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
# ROC and AUC for binary classification
fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
roc_auc = roc_auc_score(y_test, y_pred_probs)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Print classification report
print('Classification Report:')
print(classification_report(y_test, y_pred_classes))
```
<div>
<img src="https://github.com/shadagamal/ptb_xl-binary-classification-with-cnn-deeplearning/blob/main/output%20photos/Screenshot%202024-05-19%20082422.png?raw=true"width=500 heigth=500>   
<img src="https://github.com/shadagamal/ptb_xl-binary-classification-with-cnn-deeplearning/blob/main/output%20photos/Screenshot%202024-05-19%20082703.png?raw=true"width=500 heigth=500>
<img src="https://github.com/shadagamal/ptb_xl-binary-classification-with-cnn-deeplearning/blob/main/output%20photos/Screenshot%202024-05-19%20082445.png?raw=true"width=500 heigth=500>
</div>
