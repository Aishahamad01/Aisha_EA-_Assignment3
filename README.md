# Aisha_EA-_Assignment3
PINNS and Neural ODEs


This repository contains two experiments on modeling and learning dynamics from sparse data:

## Question 1: Cardiac Activation Time
Estimate cardiac activation times `T(x, y)` in a 2D domain using:
- A data-driven neural network (MSE loss)
- A Physics-Informed Neural Network (PINN) that enforces the Eikonal equation:
  

### ✅ Workflow
1. **Synthetic Data Generation**  
 - Define true activation map `T(x, y)`  
 - Define conduction velocity `V(x, y)`  
 - Sample sparse points using Latin Hypercube Sampling (LHS)

2. **Model Training**  
 - `Model 1`: Standard Neural Network using only data  
 - `Model 2`: PINN trained using residual of Eikonal equation

3. **Evaluation & Visualization**  
 - Compute RMSE for both models  
 - Visualize predictions, ground truth, and error heatmaps  
 - Colorbar and training points shown for interpretation

---

## Question  2: Neural ODEs vs. Standard NN for Classification
Compare a traditional 1-hidden-layer neural network with a Neural ODE model for binary classification on 2D synthetic datasets (e.g., Moons).

### ✅ Workflow
1. **Dataset**
 - Load 2D moons dataset from `sklearn.datasets`

2. **Modeling**
 - `Baseline`: Fully connected network with 1 hidden layer (ReLU)
 - `Neural ODE`: Uses `torchdiffeq.odeint` with `rk4` solver

3. **Training & Evaluation**
 - Use `CrossEntropyLoss` and `Adam` optimizer  
 - Compare accuracy, loss, and decision boundaries

4. **Discussion**
 - Compare discrete vs. continuous-depth architectures  
 - Optional: Euler method vs. residual connection analogy

---
### Assumptions
 - Activation function and velocity maps are synthetic 
 - Moons dataset is used with noise
## 📦 Dependencies

Install the following Python packages:

```bash
pip install numpy matplotlib torch scikit-learn pyDOE torchdiffeq
