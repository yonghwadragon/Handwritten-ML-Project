# predict_custom_2.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

def sigmoid(x):
    return 1/(1+np.exp(-x))

data = np.load('model/learningdata.npz')
w = data['w']
v = data['v']

# 예: 0_1.png (손글씨 숫자 0) 예측
custom_image_path = "dataset/sample_images/0/0_1.png"
newdata = img.imread(custom_image_path)

# 흑백으로 처리 (이미지 포맷에 따라 조정)
# 예시: 그림판 흑백 PNG라면, newdata[:,:,0] 형태가 아닐 수도 있음 -> 확인 필요
newdata = 1 - newdata[:,:,0]

plt.title("My Handwritten Digit")
plt.imshow(newdata, cmap='gray')
plt.show()

newX = np.ones([785,1])
for i in range(28):
    newX[i*28+1:(i+1)*28+1,0] = newdata[i,:]

z = np.ones([31,1])
z[1:, :] = sigmoid(np.dot(w,newX))
y = sigmoid(np.dot(v,z))

predict_value = np.argmax(y)
print(f"Prediction for custom image: {predict_value}")