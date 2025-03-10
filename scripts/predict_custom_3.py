# scripts/predict_custom_3.py
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
    # 1) 모델 로드
    model_path = os.path.join(os.path.dirname(__file__), '../model/learningdata.npz')
    data = np.load(model_path)
    w = data['w']
    v = data['v']
    hidden_node = w.shape[0]
    print(f"[predict_custom_3] Loaded model from: {model_path}")
    print(f"Hidden layer size: {hidden_node}")

    # 2) 반복할 파일 목록
    #   digit=0..9, image idx=1..4 (ex: 0_1.png.. 0_4.png)
    #   만약 파일이 실제로 없으면 skip
    input_node = 784

    for digit in range(10):           # 0..9
        for idx in range(1, 5):       # 1..4
            # 예: dataset/sample_images/3/3_2.png
            custom_image_path = os.path.join(
                os.path.dirname(__file__),
                f'../dataset/sample_images/{digit}/{digit}_{idx}.png'
            )

            if not os.path.exists(custom_image_path):
                # 파일 없으면 건너뛰기
                print(f"Skip: File not found: {custom_image_path}")
                continue

            # 3) 이미지 로드 & 전처리
            newdata = img.imread(custom_image_path)
            if newdata.max() > 1:
                newdata = newdata / 255.0

            # RGB→Grayscale
            if len(newdata.shape) == 3:
                newdata = newdata[:,:,0]

            # MNIST는 검은 배경, 흰 숫자
            newdata = 1 - newdata

            # Forward
            newX = np.ones((input_node+1,1))
            newX[1:,0] = newdata.flatten()

            z = np.ones((hidden_node+1,1))
            uh = np.dot(w, newX)
            z[1:, 0] = relu(uh).reshape(-1)

            uo = np.dot(v, z)
            y  = softmax(uo.flatten())

            predict_value = np.argmax(y)

            # 4) 결과 표시
            print(f"\n[File: {digit}_{idx}.png] => Predicted: {predict_value}")

            # 시각화(원본 이미지 & 확률 분포)
            plt.figure(figsize=(7,3))
            # 왼쪽: 입력 이미지
            plt.subplot(1,2,1)
            plt.title(f"Input: {digit}_{idx}.png")
            plt.imshow(newdata, cmap='gray')
            plt.axis("off")

            # 오른쪽: 예측 확률 막대 그래프
            plt.subplot(1,2,2)
            plt.bar(range(10), y, color="blue")
            plt.xticks(range(10))
            plt.xlabel("Digit")
            plt.ylabel("Probability")
            plt.title(f"Prediction: {predict_value}")

            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    main()