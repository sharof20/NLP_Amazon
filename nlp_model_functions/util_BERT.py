# BERT
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, log_loss


def split_data(df):
    """
    Splits data into training and test sets.

    Parameters:
    - df (DataFrame): Input DataFrame with 'Reviews' and 'Rating' columns.

    Returns:
    - train_texts (Series): Training set for text reviews.
    - test_texts (Series): Test set for text reviews.
    - train_labels (Series): Training set for ratings.
    - test_labels (Series): Test set for ratings.
    """

    # Split data into training and test sets using a 70-30 split ratio.
    return train_test_split(df['Reviews'], df['Rating'], test_size=0.3)

# Load the uncased BERT tokenizer, converting all tokens to lowercase.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def tokenize_texts(texts):
    """
    Tokenize input texts using the BERT tokenizer.

    Parameters:
    - texts (iterable): A list or series of text strings to tokenize.

    Returns:
    - input_ids (tensor): Tensor of token ids for each input text.
    - attention_masks (tensor): Tensor of attention masks for each input text.
    """

    input_ids = []
    attention_masks = []

    # Iterate through each text and encode it
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
                            text,
                            add_special_tokens=True,  # BERT-specific tokens
                            max_length=160,
                            truncation=True,
                            padding='max_length',
                            return_attention_mask=True,
                            return_tensors='tf',
                       )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert list of encoded values into tensors
    input_ids = tf.concat(input_ids, axis=0)
    attention_masks = tf.concat(attention_masks, axis=0)

    return input_ids, attention_masks

def get_dataset(input_ids, attention_masks, labels):
    """
    Convert tokenized input and labels into a TensorFlow dataset.

    Parameters:
    - input_ids (tensor): Token ids for each input text.
    - attention_masks (tensor): Attention masks for each input text.
    - labels (iterable): Corresponding labels for each input text.

    Returns:
    - data (tf.data.Dataset): A TensorFlow dataset containing tokenized input and labels.
    """

    # Create a TensorFlow dataset from tokenized inputs and labels
    data = tf.data.Dataset.from_tensor_slices(({"input_ids": input_ids, "attention_mask": attention_masks}, labels))

    return data

def get_model():
    """
    Load the BERT model, freeze specified layers, and modify the classifier with regularization and dropout.

    Returns:
    - model (TFBertForSequenceClassification): The modified BERT model for sequence classification.
    """

    # Load the BERT model for sequence classification
    model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")

    # Make the first 9 layers of the BERT model non-trainable
    for layer in model.bert.encoder.layer[:9]:
        layer.trainable = False

    # Define a new classifier layer with dropout and L2 regularization
    classifier_input = tf.keras.Input(shape=(model.config.hidden_size,), dtype=tf.float32, name="inputs")
    x = tf.keras.layers.Dropout(0.5)(classifier_input)
    classifier_output = tf.keras.layers.Dense(
        model.config.num_labels,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=model.config.initializer_range),
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    )(x)

    # Construct the new classifier model
    regularized_classifier_model = tf.keras.Model(inputs=classifier_input, outputs=classifier_output)

    # Integrate the new classifier into the BERT model
    model.classifier = regularized_classifier_model

    return model

def compile_and_train(model, train_data, test_data):
    """
    Compile, train, and evaluate the BERT model using given data.

    Parameters:
    - model (TFBertForSequenceClassification): The BERT model for sequence classification.
    - train_data (tf.data.Dataset): The training dataset.
    - test_data (tf.data.Dataset): The validation/test dataset.

    Returns:
    - model (TFBertForSequenceClassification): The trained BERT model.
    """

    # Define learning rate and decay for optimizer
    initial_learning_rate = 3e-6
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=10000,  # Decrease learning rate every 10,000 steps
        decay_rate=0.9,
        staircase=True)

    # Use the Adam optimizer with learning rate decay
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Compile the model with sparse categorical crossentropy and accuracy metric
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

    # Set early stopping criteria to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitor validation loss for early stopping
        patience=2,  # Number of epochs with no improvement to trigger early stopping
        verbose=1,
        restore_best_weights=True)  # Restore model weights from the best epoch

    # Train the model using the training data and validate using test data
    history = model.fit(train_data, epochs=20, verbose=1, validation_data=test_data, callbacks=[early_stopping])

    return model, history



def evaluate_metrics(model, test_data, test_labels):
    # 1. Obtain model's predictions for the test set
    predictions = model.predict(test_data)
    predicted_logits = predictions.logits
    predicted_probs = tf.nn.softmax(predicted_logits, axis=-1)
    predicted_labels = tf.argmax(predicted_probs, axis=1)

    # Convert test_labels to a numpy array for metric calculations
    test_labels_np = np.array(test_labels)

    # 2. Calculate metrics using sklearn
    accuracy = accuracy_score(test_labels_np, predicted_labels)
    f1 = f1_score(test_labels_np, predicted_labels, average='macro')  # Assuming a multi-class problem
    roc_auc = roc_auc_score(test_labels_np, predicted_probs.numpy()[:, 1])
    precision = precision_score(test_labels_np, predicted_labels, average='macro', zero_division=1)
    recall = recall_score(test_labels_np, predicted_labels, average='macro')
    cross_entropy = log_loss(test_labels_np, predicted_probs.numpy())

    # Sensitivity and Specificity for binary classification
    if len(set(test_labels_np)) == 2:  # Assuming labels are 0 and 1 for binary classification
        tp = tf.math.count_nonzero(predicted_labels * test_labels_np)
        tn = tf.math.count_nonzero((predicted_labels - 1) * (test_labels_np - 1))
        fp = tf.math.count_nonzero(predicted_labels * (test_labels_np - 1))
        fn = tf.math.count_nonzero((predicted_labels - 1) * test_labels_np)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
    else:
        sensitivity = "Not applicable for multi-class"
        specificity = "Not applicable for multi-class"

    # Print metrics
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC Score: {roc_auc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    print(f"Cross-Entropy Loss: {cross_entropy}")


def predict_with_bert(model, input_ids, attention_masks):
    predictions = model.predict({"input_ids": input_ids, "attention_mask": attention_masks})[0]
    # Convert logits to probabilities
    probs = tf.nn.softmax(predictions, axis=-1)
    # Get the predicted class
    predicted_class = tf.argmax(probs, axis=-1)
    return predicted_class.numpy()

def plot_confusion_matrix(model, test_input_ids, test_attention_masks, test_labels, save_path=None):
    """
    Plots a confusion matrix using predictions from a BERT model on test data.

    Parameters:
    - loaded_model (model object): Pre-trained BERT model
    - test_input_ids (np.array): Input IDs for testing
    - test_attention_masks (np.array): Attention masks for testing
    - test_labels (np.array): True labels for test data
    - save_path (str, optional): Path to save the confusion matrix plot. If None, the plot is not saved.

    Returns:
    - None: Displays the confusion matrix plot
    """

    # Get predictions on test data
    predictions = predict_with_bert(model, test_input_ids, test_attention_masks)
    true_labels = test_labels.to_numpy()

    # Compute the confusion matrix
    cm = confusion_matrix(true_labels, predictions)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=set(true_labels), yticklabels=set(true_labels))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_and_save_loss(history, save_path):
    plt.figure(figsize=(10,6))

    # Plot training loss
    plt.plot(history.history['loss'], label='Training Loss')

    # Plot validation loss
    plt.plot(history.history['val_loss'], label='Validation Loss')

    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Save the plot to a file
    plt.savefig(save_path)
    plt.show()
