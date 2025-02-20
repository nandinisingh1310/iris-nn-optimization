# Iris Dataset - Neural Network Optimization and Regularization

## Overview
This project demonstrates the implementation of various optimization and regularization techniques on a Neural Network using the Iris dataset. The primary objectives include:

- Implementing Momentum, RMSProp, and Adam optimizers from scratch.
- Demonstrating overfitting and implementing L1, L2 regularization, and Dropout.
- Visualizing the effect of these techniques on model performance.

## Dataset
The Iris dataset consists of 150 samples with four features each, representing different flower species:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

Each sample belongs to one of three species: Setosa, Versicolor, or Virginica.

### Dataset Sources:
- Scikit-Learn Iris Dataset
- Kaggle UCI Iris Dataset

## Steps and Implementation
### 1. Data Preprocessing
- Load the dataset using `load_iris()`.
- Slice into independent (X) and dependent (y) features.
- Convert categorical labels to one-hot encoding.
- Standardize features using `StandardScaler()`.
- Split into training and testing sets.

### 2. Building a Neural Network
Implemented using Sequential API from Keras.

#### Architecture:
- **Input Layer**: 8 neurons, tanh activation.
- **Hidden Layer 1**: 16 neurons, tanh activation.
- **Hidden Layer 2**: 16 neurons, tanh activation.
- **Hidden Layer 3**: 8 neurons, ReLU activation.
- **Output Layer**: 3 neurons, softmax activation.

#### Compilation:
- **Optimizer**: Adam
- **Loss function**: categorical_crossentropy
- **Training**: 200 epochs with batch size 32.

### 3. Evaluating the Model
- Loss and accuracy are printed after evaluation.
- **Visualization**: Loss vs. Epochs for training and testing sets.

### 4. Implementing Regularization Techniques
- **Without Regularization**
- **L1 Regularization**: Applied `kernel_regularizer=l1(0.01)` on all input and hidden layers.
- **L2 Regularization**: Applied `kernel_regularizer=l2(0.01)`.
- **Dropout**: Dropped 20% of nodes from all hidden layers.
- **Plot loss curves** for each case.

### 5. Implementing Optimization Algorithms from Scratch
- **Loss function**: 
- **Gradient Calculation**: 
- **Optimizers Implemented**:
  - Momentum
  - RMSProp
  - Adam
- **Convergence Criteria**: When `loss < threshold`.
- **Visualization**: Loss vs. Epochs for each optimizer.

### 6. Conclusion
- Compared the number of iterations required for each optimizer to converge.
- Identified the best optimizer based on loss and convergence speed.

## Results
| Optimizer | Convergence Iterations |
|-----------|----------------------|
| Momentum  | 198                  |
| RMSProp   | 25                   |
| Adam      | 195                 |

**Adam proved to be the best optimizer in terms of speed and stability!**

## Installation and Running the Code
### Prerequisites
Ensure you have the following libraries installed:
```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn
```

### Running the Code
Clone this repository and execute the script:
```bash
git clone <repository-url>
cd <repository-folder>
python main.py
```

## Visualization
Loss plots for different cases are available in the results folder.

## Future Improvements
- Implement additional optimizers such as Nadam and SGD with Nesterov Momentum.
- Test on other datasets to compare optimizer performance.
- Fine-tune hyperparameters for better generalization.

📌 Follow for more machine learning projects! 🚀

