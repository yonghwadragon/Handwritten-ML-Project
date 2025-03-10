# scripts/predict_book.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import os

def relu(x):
    return np.maximum(0, x)

def softmax(u):
    shift_u = u - np.max(u)
    exp_u = np.exp(shift_u)
    return exp_u / np.sum(exp_u)

def main():
    # 1) 모델 가중치 로드
    model_path = os.path.join(os.path.dirname(__file__), '../model/learningdata.npz')
    try:
        data = np.load(model_path)
        w = data['w']
        v = data['v']
        print(f"✅ Model weights loaded from: {model_path}")
    except FileNotFoundError:
        print("❌ Error: learningdata.npz not found in model folder.")
        return

    # 은닉 노드 수 자동 검출
    hidden_node = w.shape[0]
    print(f"Hidden layer size: {hidden_node}")

    # 2) test 이미지 로드
    test_image_path = os.path.join(os.path.dirname(__file__), '../dataset/test.png')
    try:
        newdata = img.imread(test_image_path)
        print(f"✅ Test image loaded from: {test_image_path}")
    except FileNotFoundError:
        print("❌ Error: test.png not found in dataset folder.")
        return

    # 3) 이미지 정규화 & 색상 반전 (MNIST는 검은 배경, 흰 숫자)
    if newdata.max() > 1:
        newdata = newdata / 255.0
    if len(newdata.shape) == 3:
        newdata = newdata[:, :, 0]
    newdata = 1 - newdata

    # 4) Forward Propagation
    #    input_node=784 → newX=(785,1)
    input_node = 784
    newX = np.ones((input_node+1, 1))
    newX[1:, 0] = newdata.flatten()

    # 은닉층
    z = np.ones((hidden_node+1, 1))
    uh = np.dot(w, newX)                   # (hidden_node,1)
    z[1:, 0] = relu(uh).reshape(-1)        # ReLU

    # 출력층
    uo = np.dot(v, z)                      # (10,1)
    y  = softmax(uo.flatten())            # Softmax

    # 예측 결과
    predict_value = np.argmax(y)
    print('✅ Predicted digit:', predict_value)

    # 5) 시각화
    plt.figure(figsize=(6, 3))
    # 입력 이미지
    plt.subplot(1, 2, 1)
    plt.imshow(newdata, cmap="gray")
    plt.title("Processed Image")
    plt.axis("off")

    # 예측 확률
    plt.subplot(1, 2, 2)
    plt.bar(range(10), y, color="blue")
    plt.xticks(range(10))
    plt.xlabel("Digit")
    plt.ylabel("Probability")
    plt.title("Prediction Probabilities")

    plt.show()

if __name__ == "__main__":
    main()