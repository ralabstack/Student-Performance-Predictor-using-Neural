# Student Performance Predictor using a Neural Network from Scratch
The project is a complete machine learning project that predicts whether a student will pass or fail based on study habits, attendance, previous grades, and lifestyle factors. The neural network is implemented from scratch using only NumPy, covering forward propagation, backpropagation, and gradient descent. Built as part of a Handshake AI Skills Showcase.
**A Handshake AI Skills Showcase Project by Vamshi Kiran Rathod Bhukya**

## Overview
This project predicts whether a student will **pass or fail** using a feedforward neural network built entirely from scratch with NumPy and no deep learning frameworks used.

## Project Structure
| File | Description |
|---|---|
| `Student_Performance_Predictor.ipynb` | Main Jupyter notebook (full walkthrough) |
| `neural_network.py` | Neural network class (NumPy only) |
| `train_evaluate.py` | Standalone training & evaluation script |
| `generate_dataset.py` | Synthetic dataset generator |
| `student_data.csv` | Generated dataset (1,000 students) |
| `results.png` | Loss curve + confusion matrix |

## How to Run
```bash
# 1. Install dependencies
pip install numpy pandas matplotlib scikit-learn jupyter
# 2. Generate dataset (already included)
python generate_dataset.py
# 3. Run the notebook
jupyter notebook Student_Performance_Predictor.ipynb
# OR run the training script directly
python train_evaluate.py
```

## Model Results
- **Train Accuracy=** ~85%
- **Test Accuracy=** ~83%
- **Architecture=** 8 → 16 → 8 → 1 (ReLU + Sigmoid)
- **Optimiser=** Gradient Descent with Backpropagation

## Skills Demonstrated
- Neural network architecture & implementation
- Backpropagation & gradient descent
- Data preprocessing & feature engineering
- Model evaluation (accuracy, precision, recall, F1)
- Data visualisation with Matplotlib
