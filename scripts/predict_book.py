#predict_book.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
def sigmoid(x):
	return 1/(1+np.exp(-x))
data = np.load('learningdata.npz')
w = data['w'];v = data['v'];
newdata = img.imread("test.png")
newdata = 1-newdata[:,:,0]
plt.imshow(newdata)
newX = np.ones([785,1]);
for i in range(0,28):
    newX[i*28+1:(i+1)*28+1,0] = newdata[i,:]
z = np.ones([31,1])
z[1:,:] = sigmoid(np.dot(w,newX))
y = sigmoid(np.dot(v,z))
predict_value=np.argmax(y)
print('New image is : %d'%(predict_value))