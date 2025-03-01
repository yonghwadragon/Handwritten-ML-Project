# predict_custom_1.py
import numpy as np
import matplotlib.pyplot as plt
import pickle

def sigmoid(x):
    return 1/(1+np.exp(-x))

# 학습된 가중치 로드
data = np.load('model/learningdata.npz')
w = data['w']
v = data['v']

# 특정 인덱스의 이미지를 살펴볼 수 있도록
index = 11 # 예: 10번째 이미지 확인

# 테스트 이미지/라벨 로드
with open('dataset/testimages.bin','rb') as f_imgs:
    X_test = pickle.load(f_imgs)/255.0
with open('dataset/testlabels.bin','rb') as f_labels:
    T_test = pickle.load(f_labels)

# 이미지 데이터 불러오기
image_data = X_test[index].reshape(28,28)
label = T_test[index]

# 시각화 (흑백으로 보이게 cmap='gray' 사용)
plt.title(f"Actual Label: {label}")
plt.imshow(image_data, cmap='gray')
plt.show()

# 예측
newX = np.ones([785,1])
newX[1:, 0] = X_test[index]
z = np.ones([31,1])
z[1:, :] = sigmoid(np.dot(w, newX))
y = sigmoid(np.dot(v, z))

predict_value = np.argmax(y)
print(f"Model Prediction: {predict_value}")
