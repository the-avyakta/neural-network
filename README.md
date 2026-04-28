# Neural Network from Scratch - Titanic Survival

A simple implementation of a neural network built **from scratch using NumPy**, trained on the Titanic dataset to predict survival.



## 🚀 What this project does

* Loads Titanic dataset using fetch_openml
* Preprocesses data (encoding + scaling)
* Implements a **1 hidden layer neural network** manually
* Uses:

  * ReLU activation (hidden layer)
  * Sigmoid activation (output layer)
  * Binary Cross Entropy loss
* Trains using **backpropagation + gradient descent**



## 🧠 Model Architecture

```text
Input (3 features)
      ↓
Hidden Layer (4 neurons, ReLU)
      ↓
Output Layer (1 neuron, Sigmoid)
```



## ⚙️ Features used

* Passenger Class (`pclass`)
* Gender (`sex`)
* Age (`age`)



## 📊 Results

| Model                  | Accuracy |
| - | -- |
| Scratch Neural Network | ~74%     |
| MLPClassifier          | ~67%     |



## 🔑 Key Concepts Covered

* Forward propagation
* Backpropagation
* Gradient descent
* Weight initialization (He init)
* Feature scaling
* Train/test split (stratified)



## 🛠 Tech Stack

* Python
* NumPy
* scikit-learn (for comparison only)



## ▶️ How to run

```bash
python simple_nn_scratch.py
```



## 📌 Notes

* This project focuses on **understanding neural networks**, not just using libraries
* Everything (forward + backward pass) is implemented manually



## 🚀 Future Improvements

* Add more features (fare, family size, etc.)
* Improve model architecture
* Add evaluation metrics (precision, recall, ROC)
* Turn into a reusable class



