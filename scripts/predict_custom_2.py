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
    data = np.load('model/learningdata.npz')
    w = data['w']
    v = data['v']
    hidden_node = w.shape[0]
    print(f"Hidden layer size: {hidden_node}")

    # 2) 사용자 손글씨 0_1.png
    custom_image_path = "dataset/sample_images/2/2_4.png"
    if not os.path.exists(custom_image_path):
        print("❌ Image not found:", custom_image_path)
        return

    newdata = img.imread(custom_image_path)
    if newdata.max() > 1:
        newdata = newdata / 255.0
    if len(newdata.shape) == 3:
        newdata = newdata[:,:,0]
    newdata = 1 - newdata  # MNIST는 검은 배경, 흰 글자

    plt.title("My Handwritten Digit")
    plt.imshow(newdata, cmap='gray')
    plt.show()

    # 3) Forward (ReLU + Softmax)
    input_node = 784
    newX = np.ones((input_node+1,1))
    # Flatten
    for i in range(28):
        newX[i*28+1:(i+1)*28+1,0] = newdata[i,:]

    z = np.ones((hidden_node+1,1))
    uh = np.dot(w, newX)
    z[1:, 0] = relu(uh).reshape(-1)

    uo = np.dot(v, z)
    y  = softmax(uo.flatten())

    predict_value = np.argmax(y)
    print(f"Prediction for custom image: {predict_value}")

if __name__ == "__main__":
    main()