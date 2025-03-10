# scripts/predict_custom_1.py
import numpy as np
import matplotlib.pyplot as plt
import pickle

def relu(x):
    return np.maximum(0, x)

def softmax(u):
    shift_u = u - np.max(u)
    exp_u = np.exp(shift_u)
    return exp_u / np.sum(exp_u)

def main():
    # 1) 모델 가중치 로드
    data = np.load('model/learningdata.npz')
    w = data['w']
    v = data['v']

    hidden_node = w.shape[0]
    print(f"Hidden layer size: {hidden_node}")

    # 2) 특정 인덱스의 이미지 로드
    index = 11  # 예시
    with open('dataset/testimages.bin','rb') as f_imgs:
        X_test = pickle.load(f_imgs)
    with open('dataset/testlabels.bin','rb') as f_lbl:
        T_test = pickle.load(f_lbl)

    # 3) 시각화
    image_data = X_test[index].reshape(28,28)
    label = T_test[index]
    plt.title(f"Actual Label: {label}")
    plt.imshow(image_data, cmap='gray')
    plt.show()

    # 4) Forward (ReLU + Softmax)
    input_node = 784
    newX = np.ones((input_node+1,1))
    newX[1:, 0] = X_test[index]

    z = np.ones((hidden_node+1,1))
    uh = np.dot(w, newX)
    z[1:, 0] = relu(uh).reshape(-1)

    uo = np.dot(v, z)
    y  = softmax(uo.flatten())

    predict_value = np.argmax(y)
    print(f"Model Prediction: {predict_value}")

if __name__ == "__main__":
    main()