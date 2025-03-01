#predict_book.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
def sigmoid(x):
	return 1/(1+np.exp(-x))
# data = np.load('learningdata.npz') -> data = np.load('model/learningdata.npz')
# 학습된 가중치를 'model/learningdata.npz'에서 로드
data = np.load('model/learningdata.npz')
w = data['w'];v = data['v']
# newdata = img.imread("test.png") -> newdata = img.imread("dataset/test.png")
# 테스트 이미지를 'dataset/test.png' 경로로 변경
newdata = img.imread("dataset/test.png")
newdata = 1-newdata[:,:,0]
plt.imshow(newdata)
newX = np.ones([785,1])
for i in range(0,28):
    newX[i*28+1:(i+1)*28+1,0] = newdata[i,:]
z = np.ones([31,1])
z[1:,:] = sigmoid(np.dot(w,newX))
y = sigmoid(np.dot(v,z))
predict_value=np.argmax(y)
print('New image is : %d'%(predict_value))