import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Embedding, Conv1D, LSTM, Dense, Flatten

from keras_self_attention import SeqSelfAttention

from tensorflow.keras.layers import Dropout



# prep data for modelling

#Read the data
df = pd.read_csv('Peptide_glycan_data.csv')



# Combine peptide and glycan sequences

df['combined'] = df['peptide'] + df['glycan']



# Tokenization

tokenizer = Tokenizer(char_level=True, oov_token='UNK')

tokenizer.fit_on_texts(df['combined'])



# Convert sequences into integer sequences

sequences = tokenizer.texts_to_sequences(df['combined'])



# Padding

padded_sequences = pad_sequences(sequences, padding='post')



# Train-validation split

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, df['rt'].values, test_size=0.2, random_state=42)



#data = pd.read_csv('GlycanComp.csv')

#label = data.pop('RT')



# Train-Test split

#X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)



def build_model(seq_length):

    model = Sequential()

    input_dim = len(tokenizer.word_index) + 1

    #input_dim = 32



    # Embedding layer

    model.add(Embedding(input_dim=input_dim, output_dim=128, input_length=seq_length))



    # Convolutional layers

    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))

    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))



    # Recurrent layer

    model.add(LSTM(128, return_sequences=True))  # Set return_sequences to True for attention

    

     # Dropout layer (added after LSTM)

    #model.add(Dropout(0.1))  # 10% dropout rate



    # Attention mechanism

    model.add(SeqSelfAttention(attention_activation='softmax'))



    # Dense layers

    model.add(Flatten())  # Flatten the output for the Dense layer

    model.add(Dense(64, activation='relu'))

    model.add(Dense(1))



    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])

    return model



#build model

model = build_model(seq_length=padded_sequences.shape[1])

#model = build_model(seq_length = 32)

# train

history = model.fit(X_train, y_train, validation_split=0.2, epochs=200, batch_size=20, verbose=1)



# evaluate

loss, mae, mse = model.evaluate(X_test, y_test, verbose=0)

print(f'Validation Mean Absolute Error (MAE): {mae}')

print(f'Validation Mean Squared Error (MSE): {mse}')



# visualise

plt.figure(figsize=(14,5))



# Plotting loss

plt.subplot(1, 2, 1)

plt.plot(history.history['loss'], label='Train Loss')

plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title('Loss Evolution')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.savefig('glycoRT_model2_eval_lossPlot.svg', dpi=300)



# Plotting Mean Absolute Error

plt.subplot(1, 2, 2)

plt.plot(history.history['mae'], label='Train MAE')

plt.plot(history.history['val_mae'], label='Validation MAE')

plt.title('Mean Absolute Error Evolution')

plt.xlabel('Epochs')

plt.ylabel('MAE')

plt.legend()

plt.tight_layout()

#plt.show()

plt.savefig('glycoRT_model2_eval_maePlot.svg', dpi=300)



# Predict 

y_pred = model.predict(X_test)



# evaluate:

from sklearn.metrics import r2_score, mean_squared_error



# Calculate R^2 score

r2 = r2_score(y_test, y_pred)

print(f"R^2 Score: {r2:.4f}")



# Calculate MSE

mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")



# Calculate error

errors = y_test - y_pred.squeeze()  # squeeze to ensure the dimensions match





#visulaise 

plt.figure(figsize=(10,6))

plt.scatter(y_test, y_pred, alpha=0.5)

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')

plt.xlabel('True Retention Time')

plt.ylabel('Predicted Retention Time')

plt.title('True vs Predicted Retention Time')

plt.grid(True)

#plt.show()

plt.savefig('glycoRT_model2_eval_r2Plot.svg', dpi=300)



plt.figure(figsize=(10, 6))

plt.hist(errors, bins=100, alpha=0.7, color='blue')

plt.xlabel('Prediction Error')

plt.ylabel('Frequency')

plt.title('Histogram of Prediction Errors')

plt.grid(True)

#plt.show()

plt.savefig('glycoRT_model2_eval_errorHist.svg', dpi=300)





# save model

model.save('glycoRT_model2.h5')