## Project Name: Sentiment Analysis of Customer Reviews using Deep Learning and Natural Language Processing

### Description:
This project aims to develop a sentiment analysis system that utilizes deep learning techniques and natural language processing (NLP) to analyze customer reviews. The system will be capable of automatically determining the sentiment expressed in the reviews, whether positive or negative. The data for this analysis have been obtained through web scraping, where relevant customer reviews will be extracted from Amazon.
Two state-of-the-art approaches for sentiment analysis on Amazon product reviews are evaluated: a Bidirectional LSTM model utilizing pre-trained GloVe word embeddings, and a BERT-based model.

### The repository is structured as follows:

1. **project_notebook**: The main notebook where the entire project can be executed. The notebook contains detailed explanation each step of the model constructions.
2. **data**: Contains all datasets essential for model building.
3. **data_collection**: Houses the file `util_collect.py`, which includes functions and imports for data collection.
4. **data_management**: This folder focuses on data management tasks such as merging, cleaning, and preparing datasets for sentiment analysis of Amazon reviews.
5. **model_results**: Stores saved NLP models and relevant plots for model evaluation.
6. **nlp_model_functions**: Contains utility functions for models. These are imported and used in the main project notebook for model creation.
7. **sentiment_analysis_of_Amazon_product_reviews**: It explains how models performed.
## How to Get Started:

To get started, follow these steps:

1. Clone the repository to have a local copy of the project on your machine.

1. Create a Conda environment specific to this project and activate it using the
   following commands:

   ```bash
   $ conda env create -f environment.yml
   $ conda activate amazon_reviews
   ```

   This will help you manage dependencies and ensure that you have the necessary
   packages installed which are written down in the file `environment.yml`.

1. Install Chromedriver: This is not required to run the project by default and you can
   already proceed to the next step. We skip the task of data collection through
   webscraping in Chrome as it takes time (about 10 hours) and the rest of the tasks
   will use the data that we have already collected. Therefore, you can install
   chromedriver only if you want to collect data from scratch. Go
   [here](https://chromedriver.chromium.org/getting-started) to download and install the
   chromedriver on your machine. Make sure you have a version that is compatible with
   your Chrome browser.

   Futhermore, if you want to run the webscraping part, then uncomment the data_collection function in the project notebook.

## Usage:

### Running the project

Once you've set up the environment and installed the necessary packages, you can execute the project by running the `project_notebook` located in the `home` folder of the project.

The project results will be saved in the **model_results** folder. The project is organized into the following modules, each consisting of specific tasks:

- `data_collection` - Data collection via web scraping (skipped by default).
- `data_management` - Cleaning of raw data.
- `model` - Execution of the NLP model on the data.
- `paper` - A concise summary of the project and its outcomes.

### 1. List five different tasks that belong to the field of natural language processing.

1. **Text classification**: The process of categorizing text into predefined groups or classes.
2. **Named Entity Recognition**: The process of identifying and classifying named entities (e.g., persons, organizations, locations) in text.
3. **Question answering**: A task where the system provides an answer to a user's question based on a given dataset or knowledge base.
4. **Summarization**: The process of creating a concise and coherent summary of a larger text.
5. **Translation**: Translating text from one language to another.

### 2. What is the fundamental difference between econometrics/statistics and suprevised machine learning?
- **Econometrics**: 
  - Goal: Estimate fundamentally unobservable parameters and test hypotheses about them.
  - Focus: Justifying assumptions.
  - Post-Estimation: Cannot test how well it worked.
  
- **Supervised Machine Learning**: 
  - Goal: Predict observable things.
  - Focus: Experimentation, evaluation, and finding out what works.
  - Post-Training: Can check how well it works.

### 3. Can you use stochastic gradient descent to tune the hyperparameters of a random forrest. If not, why?
No, stochastic gradient descent (SGD) cannot be used to tune the hyperparameters of a random forest. The primary reason is that random forests are non-differentiable models. SGD requires the gradient of the loss function with respect to model parameters for optimization. Given that random forests are constructed from decision trees that make discrete splits on features, there's no continuous gradient available for the optimization process.

### 4. What is imbalanced data and why can it be a problem in machine learning?

Imbalanced data refers to scenarios where certain outcomes or classes dominate over others in a dataset. Imbalanced data in the context of Amazon review scores often occurs when there are significantly more positive reviews (e.g., 4 or 5 stars) compared to negative reviews (e.g., 1 or 2 stars) in the dataset. This imbalance can create challenges when training machine learning models for sentiment analysis or product recommendation. Machine learning models might be misled by always predicting the more frequent class, resulting in deceptively high accuracy. However, the model might not truly understand the data's nuances. This skewed prediction can be problematic, especially when the minority class is crucial.

### 5. Why are samples split into training and test data in machine learning?

In machine learning, it's standard practice to divide samples into training and test sets. The primary reason is to ensure a reliable model evaluation. The training set is where the model learns and adjusts its parameters. Meanwhile, the test set acts as new, unseen data to assess how well the model generalizes to unfamiliar data. This separation helps in identifying and preventing overfitting, where a model might excel on training data but fail on new data. It's vital to ensure that no information from the test set influences the training to maintain a genuine evaluation.

### 6. Describe the pros and cons of word and character level tokenization.

**Word Tokenization**:
- **Pros**:
  - Retains the structure of words.
  - Straightforward to implement.
  - Typically results in a moderate vocabulary size.
- **Cons**:
  - Struggles with variations like "awsoooooome".
  - Can misinterpret typos such as "hepl".
  - Might not handle morphological changes like "helped".
  - Vocabulary might not cover all words, leading to incompleteness.

**Character Tokenization**:
- **Pros**:
  - Simple to implement.
  - Results in a very small vocabulary size.
  - Eliminates the issue of unknown words.
- **Cons**:
  - Doesn't retain the structure of entire words.
  - Tokenized texts can be lengthy.

### 7. Why does fine-tuning usually give you a better performing model than feature extraction?

Fine-tuning usually results in a better-performing model than feature extraction because it optimizes the entire model, including the hidden states, to the specific task at hand. In contrast, feature extraction relies on the pre-trained model's hidden states, which might not be perfectly aligned with the new task.

### 8. What are advantages of feature extraction over fine-tuning?

**Advantages of Feature Extraction over fine-tuning are**:
1. **Simplicity**: Feature extraction involves training the parameters of a classification model without adjusting the entire pre-trained model.
2. **Flexibility**: The classifier in feature extraction can be diverse, allowing for various models like random forests or linear classifiers.
3. **Efficiency**: Feature extraction typically requires less computational power. It can be performed with just a CPU, making it more accessible for those without powerful GPUs.
4. **Speed**: Since only the top layers are trained, and not the entire network, feature extraction can be faster than fine-tuning, especially when using large pre-trained models.

### 9. Why are neural networks trained on GPUs or other specialized hardware?

Neural networks are often trained on GPUs or specialized hardware due to:
1. **Parallel Processing**: GPUs excel at handling multiple tasks at once, making them perfect for the parallel computations in neural network training.
2. **Floating-Point Units**: With a higher count of floating-point units, GPUs can conduct numerous mathematical operations simultaneously.
3. **Efficiency**: Given the computational demands of training neural networks, GPUs, optimized for matrix operations and high computational tasks, offer a substantial speed advantage.

### 10. How can you write pytorch code that uses a GPU if it is available but also runs on a laptop that does not have a GPU?

To ensure your PyTorch code is adaptable to different hardware setups, you can define the device based on the availability of a GPU:

~~~python
import torch

# Determine the device: GPU (cuda) if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a tensor
a = torch.ones(180, 180)

# Transfer the tensor to the chosen device
a_gpu = a.to(device)
~~~
### 11. How many trainable parameters would the neural network in this video have if we remove the second hidden layer but leave it otherwise unchanged.

**12,730** = 784×16+16×10+(16+10)

### 12. Why are nonlinearities used in neural networks? Name at least three different nonlinearities!

**Why are nonlinearities essential in neural networks?**
Nonlinearities introduce complexity into neural networks, allowing them to learn intricate patterns in the data. Without them, even multi-layered networks would act as a single-layer linear model, limiting their capability to understand complex relationships.

**Three common nonlinearities**:
1. ReLU (Rectified Linear Unit)
2. Sigmoid
3. Tanh (Hyperbolic Tangent)

### 13. Some would say that softmax is a bad name. What would be a better name and why?

A more intuitive name for "softmax" might be "normalized_exponential_scaling" or "exp_normalizer". The rationale behind this is that the softmax function scales its input values using exponentiation and subsequently normalizes them, ensuring the output values form a valid probability distribution.

### 14. What is the purpose of DataLoaders in PyTorch?

DataLoaders in PyTorch are designed to facilitate the process of feeding data to the model. Their main purposes include:
1. Simplifying the iteration over data batches.
2. Abstracting the mechanics of data shuffling and batching.
3. Providing options to enable or disable data shuffling.
4. Managing datasets that aren't perfect multiples of the batch size with the `drop_last` parameter.
5. Optionally loading data in parallel for enhanced efficiency.

### 15. Name a few different optimizers that are used to train deep neural networks?

Several optimizers used in training deep neural networks are:
1. **SGD (Stochastic Gradient Descent)**
2. **Adam (Adaptive Moment Estimation)**
3. **RMSprop (Root Mean Square Propagation)**
4. **Adagrad (Adaptive Gradient Algorithm)**
5. **Momentum**
6. **Adadelta (Adaptive Delta)**

### 16. What happens when the batch size during the optimization is set too small?

When the batch size is set too small during optimization: update becomes erratic
1. **Erratic Updates**: Weight updates can become unstable due to high variance in gradient estimates.
2. **Slower Convergence**: The model might take longer to converge or might not converge at all.

### 17. What happens when the batch size during the optimization is set too large?

When the batch size is set too large during optimization:
1. **Slower Computations**: Gradient calculations on large batches can be time-consuming.
2. **Memory Limitations**: Larger batches might exceed GPU or system memory.
3. **Less Frequent Updates**: Weight updates become less frequent, potentially affecting convergence speed.
4. **Risk of Overfitting**: Training with large batches for many epochs can lead to overfitting.
5. **Sharper Minima**: The model might converge to sharper minima, affecting generalization on new data.

### 18. Why can the feed-forward neural network we implemented for image classification not be used for language modelling?

Feed-forward neural networks tailored for image classification are not apt for language modeling due to:
1. **Fixed Input Size**: These networks need a consistent input size, whereas language data can differ in length.
2. **Absence of Memory**: They don't maintain memory or state between inputs, essential for processing sequential data like language.
3. **Task Specificity**: The model's architecture might be optimized for image classification and not be adaptable for language modeling.

### 19. Why is an encoder-decoder architecture used for machine translation (instead of the simpler encoder only architecture used for language modelling)?
  Answer: The encoder-decoder architecture is favored for machine translation because input and target sentences might have different lengths and word orders. While language models can be used for translation, they aren't as efficient for this task. The encoder captures the entire source sentence's essence, and the decoder generates the corresponding translation, ensuring accurate alignment between source and target meanings.
### 20. Is it a good idea to base your final project on a paper or blogpost from 2015? Why or why not?
LSTM: While LSTM networks were introduced in the late 1990s, they gained substantial popularity in the 2010s. By 2015, they were well-established. So, referencing LSTM-related materials from 2015 might still be relevant.
BERT: BERT (Bidirectional Encoder Representations from Transformers) was introduced by Google in 2018. Any content from 2015 wouldn't cover BERT. Thus, the model would miss out on the methodologies, techniques, and best practices related to BERT.
### 21. Do you agree with the following sentence: To get the best model performance, you should train a model from scratch in Pytorch so you can influence every step of the process?

While training a model from scratch in PyTorch provides deep understanding and the ability to influence every step, it doesn't always ensure the best performance. Here are some considerations:

1. **Transfer Learning**: Using pre-trained models and fine-tuning them for specific tasks can often yield superior results, especially with limited data.
2. **Computational Cost**: Training from scratch can be resource-intensive and time-consuming.
3. **Risk of Overfitting**: Without a sufficiently diverse dataset, there's a risk of the model overfitting when trained from scratch.
4. **Expertise Required**: Comprehensive understanding of neural networks, optimization techniques, and other intricacies is needed to train from scratch.

In essence, while training from scratch offers control and deep insights, it's not always the best approach for optimal performance. The strategy should be tailored to the problem at hand, the data available, and the desired outcomes.

### 22. What is an example of an encoder-only model?

An example of an encoder-only model is BERT. BERT (Bidirectional Encoder Representations from Transformers) is a natural language processing model that uses a transformer-based architecture to pre-train on large amounts of text data. Unlike traditional models that read text sequentially, BERT can understand the context of words by considering both their left and right surrounding words, making it bidirectional. It's widely used for various NLP tasks such as text classification, sentiment analysis, and question answering.

### 23. What is the vanishing gradient problem and how does it affect training?

The vanishing gradient problem refers to the situation where the gradients of RNNs can easily vanish, meaning they become zero. This issue, along with the exploding gradient problem, poses challenges during training. Both scenarios are problematic, and while LSTMs improve this situation, they don't completely eliminate the problem. Transformers, too, can face these gradient issues but use residual connections and layer normalization to counteract them.

### 24. Which model has a longer memory: RNN or Transformer?
Transformers have a longer memory compared to RNNs. While RNNs struggle to model relationships between distant words due to their recurrent structure, transformers overcome this limitation.

### 25. What is the fundamental component of the transformer architecture?
Answer: The fundamental component of the transformer architecture is the Attention Mechanism, specifically the "Self-Attention" mechanism. It allows the model to focus on different parts of the input with varying degrees of attention, capturing long-range dependencies in the data.


