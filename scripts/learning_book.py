# scripts/learning_book.py
import numpy as np
import pickle
import os
def relu(x):
    return np.maximum(0, x)
def d_relu(x):
    return (x > 0).astype(float)
def softmax(u):
    # 오버플로 방지
    shift_u = u - np.max(u)
    exp_u = np.exp(shift_u)
    return exp_u / np.sum(exp_u)
def cross_entropy(y, t):
    return -np.sum(t * np.log(y + 1e-12))
def main():
    # 1. Load MNIST data (/255.0 적용됨)
    dataset_path = os.path.join(os.path.dirname(__file__), '../dataset')
    with open(os.path.join(dataset_path, 'trainimages.bin'),'rb') as fi:
        X = pickle.load(fi)  # shape (60000, 784)
    with open(os.path.join(dataset_path, 'trainlabels.bin'),'rb') as fl:
        T = pickle.load(fl)  # shape (60000,)
    # 2. Hyperparams (from scikit best params)
    input_node = 784
    hidden_node = 100
    output_node = 10
    eta = 0.001
    MaxIter = 100
    np.random.seed(10)
    w = 0.1 * (2*np.random.random((hidden_node, input_node+1)) - 1)  # hidden
    v = 0.1 * (2*np.random.random((output_node, hidden_node+1)) - 1) # output
    # Setup
    xl = np.ones((input_node+1,1))       # input + bias
    z = np.ones((hidden_node+1,1))       # hidden + bias
    E1 = 0.0
    # 3. Initial error
    for i in range(len(X)):
        xl[1:,0] = X[i]
        uh = np.dot(w, xl)
        z[1:,0] = relu(uh).reshape(-1)
        uo = np.dot(v, z)
        y  = softmax(uo.flatten())
        t = np.zeros(output_node)
        t[T[i]] = 1.0
        E1 += cross_entropy(y, t)
    E1 /= len(X)
    print("Initial Error:", E1)
    # 4. Training loop
    for epoch in range(1, MaxIter+1):
        E = 0.0
        for i in range(len(X)):
            # Forward
            xl[1:,0] = X[i]
            uh = np.dot(w, xl)
            z[1:,0] = relu(uh).reshape(-1)
            uo = np.dot(v, z)
            y  = softmax(uo.flatten())
            t = np.zeros(output_node)
            t[T[i]] = 1.0
            E += cross_entropy(y, t)
            # Backprop
            del_k = y.reshape(-1,1) - t.reshape(-1,1)  # (10,1)
            dEdv = np.dot(del_k, z.T)                  # (10, hidden_node+1)
            del_j = d_relu(uh).reshape(-1,1) * np.dot(v[:,1:].T, del_k)
            dEdw = np.dot(del_j, xl.T)
            # Update
            v -= eta * dEdv
            w -= eta * dEdw
        E /= len(X)
        print(f"Epoch {epoch}/{MaxIter}, Error={E:.6f}")
    print("Training finished. Final error:", E)
    # 5. Save model
    model_path = os.path.join(os.path.dirname(__file__), '../model/learningdata.npz')
    np.savez(model_path, w=w, v=v)
    print("Saved model to:", model_path)
if __name__ == "__main__":
    main()