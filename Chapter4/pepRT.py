import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout
from kerastuner.tuners import RandomSearch
import seaborn as sns
import tensorflow_addons as tfa
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter 


# Dictionary for amino acids to gravy
aa_to_gravy = {
               "A": 1.800,
               "R": -4.500,
               "N": -3.500,
               "D": -3.500,
               "C": 2.500,
               "E": -3.500,
               "Q": -3.500,
               "G": -0.400,
               "H": -3.200,
               "I": 4.500,
               "L": 3.800,
               "K": -3.900,
               "M": 1.900,
               "F": 2.800,
               "P": -1.600,
               "S": -0.800,
               "T": -0.700,
               "W": -0.900,
               "Y": -1.300,
               "V": 4.200}


# Function to get gravy for a sequence
def get_gravy(s):
    if len(s) == 0:
        return 0
    return sum([aa_to_gravy.get(si, 0) for si in s]) / len(s)

#Function to preprocess data for amino acid, diamino acid, gravy, pep length

def preprocess_data(df):
    if "Sequence" not in df.columns:
        raise ValueError("Data frame missing sequence column")
    if "Retention time" not in df.columns:
        raise ValueError("Data frame missing RT column")
    df_= pd.DataFrame([Counter(list(x)) for x in df['Sequence']], index=df.index)
    df_f = df.join(df_)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1,2))
    df_feat_ = pd.DataFrame(vectorizer.fit_transform(df_f["Sequence"]).toarray(), index=df_f.index, columns=vectorizer.get_feature_names())
    df_feat = df_f.join(df_feat_)
    df_feat["length"] = df_feat["Sequence"].apply(len)
    df_feat["gravy"] = df_feat["Sequence"].apply(get_gravy)
    # create feature and  label datasets by dropping sequence column and poping retention column
    df_feat = df_feat.fillna(0)
    df_feat.drop('Sequence', axis=1, inplace=True)
    df_label = df_feat.pop('Retention time')
    return df_feat, df_label

# load train data
df1 = pd.read_csv("train_dataset.csv")

#load test data
df2 = pd.read_csv('test_dataset.csv')

# create train dataset and train label dataset
train_data, train_labels = preprocess_data(df1)
#train_data.to_csv('train_data.csv')
#train_labels.to_csv('train_labels.csv')

# create test dataset and test label dataset
test_data, test_labels = preprocess_data(df1)
#test_data.to_csv('test_data.csv')
#test_labels.to_csv('test_labels.csv')


# Split the Training Data into Training & Validation Sets
X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)


# 1. Tune the Model Structure
def build_model(hp):
    model = keras.Sequential()
    model.add(Input(shape=(10,)))

    # Tune the number of units in the first dense layer
    for i in range(hp.Int('num_layers', 2, 5)):
        model.add(Dense(units=hp.Int('units_' + str(i),
                                        min_value=30,
                                        max_value=600,
                                        step=30),
                       activation='relu'))
        model.add(Dropout(rate=hp.Float('dropout_' + str(i),
                                          min_value=0.0,
                                          max_value=0.5,
                                          step=0.1)))
    model.add(Dense(1))

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mse', 'mae'])
    return model

tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,
    executions_per_trial=3)

tuner.search(train_data, train_labels,
             epochs=5,
             validation_data=(val_data, val_labels))

# Get the optimal hyperparameters
best_hps_structure = tuner.get_best_hyperparameters(num_trials=1)[0]
print('Best parameteres:', best_hps_structure)

# 2. Construct the Optimal Model Structure
model = tuner.hypermodel.build(best_hps_structure)

# 3. Tune Learning Rate & Batch Size
def build_model_with_tuned_structure(hp):
    model = keras.Sequential()
    model.add(Input(shape=(60,)))
    
    for i in range(best_hps_structure.get('num_layers')):
        model.add(Dense(units=best_hps_structure.get('units_' + str(i)),
                        activation='relu'))
        model.add(Dropout(rate=best_hps_structure.get('dropout_' + str(i))))

    model.add(Dense(1))

    lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss='mean_squared_error',
                  metrics=['mse', 'mae'])

    return model

tuner_lr_bs = RandomSearch(
    build_model_with_tuned_structure,
    objective='val_loss',
    max_trials=5,
    executions_per_trial=3)

batch_size = hp.Choice('batch_size', values=[50, 200, 500, 1000])

tuner_lr_bs.search(train_data, train_labels,
                   epochs=5,
                   batch_size=batch_size,
                   validation_data=(val_data, val_labels))

best_hps_lr_bs = tuner_lr_bs.get_best_hyperparameters(num_trials=1)[0]
print('Best learning parameteres:', best_hps_structure)

# 4. Train the Final Model with Early Stopping
final_model = tuner_lr_bs.hypermodel.build(best_hps_lr_bs)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

history = final_model.fit(X_train, y_train,
                          epochs=200,
                          batch_size=best_hps_lr_bs.get('batch_size'),
                          validation_data=(X_val, y_val),
                          callbacks=[early_stopping])

def evaluate_on_test(model, test_data, test_labels):
    test_loss, test_mse, test_mae = model.evaluate(test_data, test_labels)
    print(f"Test Loss (MSE): {test_loss}")
    print(f"Test MSE: {test_mse}")
    print(f"Test MAE: {test_mae}")

# 5. Visualize Training and Validation Loss/MSE

def plot_loss_mse(history, save_path="loss_mse_plot.png"):
    plt.figure(figsize=(14, 5))

    # Plotting Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting MSE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mse'], label='Training MSE')
    plt.plot(history.history['val_mse'], label='Validation MSE')
    plt.title('Training and Validation MSE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    #plt.show()
    plt.savefig('DNNmodel_eval_loss_mse_Plot.svg', dpi-300)


def calculate_error_metrics(true_labels, predictions):
    mse = mean_squared_error(true_labels, predictions)
    r2 = r2_score(true_labels, predictions)

    print(f"Mean Squared Error on Test Set: {mse}")
    print(f"R^2 on Test Set: {r2}")

def plot_true_vs_predicted_and_histogram(true_labels, predictions, save_path="true_vs_predicted_histogram.png"):
    plt.figure(figsize=(14, 5))

    # True vs. Predicted Scatterplot
    plt.subplot(1, 2, 1)
    plt.scatter(true_labels, predictions, color='black', alpha=0.5)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True vs Predicted Values')
    plt.plot([min(true_labels), max(true_labels)], [min(true_labels), max(true_labels)], color='blue')  # x=y line

    # Histogram of Errors
    plt.subplot(1, 2, 2)
    error = predictions - true_labels
    plt.hist(error, bins=100)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Histogram of Prediction Errors')

    plt.tight_layout()
    plt.savefig(save_path)
    #plt.show()
    plt.savefig('DNNmodel_eval_error_Plot_Hist.svg', dpi-300)


evaluate_on_test(final_model, test_data, test_labels)
calculate_error_metrics(test_labels, predictions)
plot_loss_mse(history, "loss_mse_plot.png")
plot_true_vs_predicted_and_histogram(test_labels, predictions, "true_vs_predicted_histogram.png")

# save model
model.save('pepRT.h5')

# visulaise

def plot_tuner_results(tuner):
    results = tuner.get_best_hyperparameters(num_trials=tuner.oracle.trials)
    trial_scores = [trial.score for trial in tuner.oracle.trials.values()]

    plt.figure(figsize=(14, 5))
    plt.bar(range(len(trial_scores)), trial_scores)
    plt.xlabel('Trial')
    plt.ylabel('Validation Loss')
    plt.title('Scores of all Trials')
    plt.show()
    
# For the structure tuner
plot_tuner_results(tuner)

# For the learning rate & batch size tuner
plot_tuner_results(tuner_lr_bs)

