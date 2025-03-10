# scripts/accuracy_check.py
import numpy as np
import pickle
import os

def relu(x):
    return np.maximum(0, x)

def softmax(u):
    # 오버플로 방지
    shift_u = u - np.max(u)
    exp_u = np.exp(shift_u)
    return exp_u / np.sum(exp_u)

def main():
    # 1) 가중치 로드
    model_path = os.path.join(os.path.dirname(__file__), '../model/learningdata.npz')
    data = np.load(model_path)
    w = data['w']
    v = data['v']
    print("✅ Loaded model from:", model_path)

    # 은닉 노드 수 자동 감지
    hidden_node = w.shape[0]
    print(f"Hidden layer size detected: {hidden_node}")

    # 2) 테스트 데이터 로드
    with open('dataset/testimages.bin','rb') as f_imgs:
        X_test = pickle.load(f_imgs)  # 이미 0~1 범위
    with open('dataset/testlabels.bin','rb') as f_labels:
        T_test = pickle.load(f_labels)

    correct_count = 0
    N = len(X_test)
    input_node = 784

    # forward 시 사용하는 임시 배열
    xl = np.ones((input_node+1, 1))        # (785,1)
    z  = np.ones((hidden_node+1, 1))       # (hidden_node+1, 1)

    # 3) 모든 테스트 이미지 예측
    for i in range(N):
        xl[1:, 0] = X_test[i]             # (784,)

        # 은닉층: ReLU
        uh = np.dot(w, xl)                # shape=(hidden_node,1)
        z[1:, 0] = relu(uh).reshape(-1)   # shape=(hidden_node,)

        # 출력층: Softmax
        uo = np.dot(v, z)                 # shape=(10,1)
        y  = softmax(uo.flatten())        # (10,)

        predict_value = np.argmax(y)
        if predict_value == T_test[i]:
            correct_count += 1

    accuracy = correct_count / N * 100
    print(f"\nAccuracy on test dataset: {accuracy:.2f}%")

    # 4) 일부 예측 확인
    print("\nSample predictions (first 10):")
    for i in range(10):
        xl[1:, 0] = X_test[i]
        uh = np.dot(w, xl)
        z[1:, 0] = relu(uh).reshape(-1)

        uo = np.dot(v, z)
        y  = softmax(uo.flatten())

        predict_value = np.argmax(y)
        print(f"Index={i}, Actual: {T_test[i]}, Predicted: {predict_value}")

if __name__ == "__main__":
    main()