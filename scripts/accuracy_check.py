# accuracy_check.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import pickle

def sigmoid(x):
    return 1/(1+np.exp(-x))

# 1) 학습된 가중치 로드
data = np.load('model/learningdata.npz')
w = data['w']
v = data['v']

# 2) testimages.bin, testlabels.bin 불러오기
with open('dataset/testimages.bin','rb') as f_imgs:
    X_test = pickle.load(f_imgs)/255.0
with open('dataset/testlabels.bin','rb') as f_labels:
    T_test = pickle.load(f_labels)

correct_count = 0

# 3) 반복문으로 모든 테스트 이미지 예측
for i in range(len(X_test)):
    newX = np.ones([785,1])
    newX[1:,0] = X_test[i]

    z = np.ones([31,1])
    z[1:,:] = sigmoid(np.dot(w,newX))
    y = sigmoid(np.dot(v,z))

    predict_value = np.argmax(y)

    if predict_value == T_test[i]:
        correct_count += 1

accuracy = correct_count / len(X_test) * 100
print(f"Accuracy on test dataset: {accuracy:.2f}%")
