#learning_book.py
import numpy as np
import pickle 
def mnist_load(filename):
    Images = open(filename+'images.bin','rb')
    Labels = open(filename+'labels.bin','rb')
    X = pickle.load(Images)/255.0
    T = pickle.load(Labels)
    Images.close()
    Labels.close()
    return (X,T)
def sigmoid(x):
	return 1.0/(1.0+np.exp(-x))
def d_sigmoid(u):
    return np.exp(-u)/(1.0+np.exp(-u))**2
# (X,T) = mnist_load('train') -> (X, T) = mnist_load('dataset/train')
# 경로 수정: 'dataset/train'으로 바꿔서 dataset 폴더 안에 있는 trainimages.bin, trainlabels.bin 사용
(X, T) = mnist_load('dataset/train')
input_node = 28*28
hidden_node = 30
output_node = 10
np.random.seed(10)
w = 0.1*(2*np.random.random((hidden_node,input_node+1))-1)
v = 0.1*(2*np.random.random((output_node,hidden_node+1))-1)
xl = np.ones((input_node+1,1))
z = np.ones((hidden_node+1,1))
y = np.ones((output_node,1))
eta = 0.05;MaxIter = 50;Tol=1.0e-10;Resid=Tol*2
Iter=1;E1 = 0
for m in range(len(X)):
    xl[1:,0] = X[m]
    uh = np.dot(w,xl)
    z[1:,:] = sigmoid(uh)
    uo = np.dot(v,z)
    y = sigmoid(uo)
    t=np.zeros([output_node,1])				
    t[T[m]] = 1.0
    E1 = E1+np.sum((y-t)**2)
E1=E1/len(T)
print('%i-th update and error is %f'%(Iter-1,E1))
while Resid>=Tol and Iter<=MaxIter:
    for m in range(len(X)):
        xl[1:,0] = X[m]
        uh = np.dot(w,xl)
        z[1:,:] = sigmoid(uh)
        uo = np.dot(v,z)
        y = sigmoid(uo)
        t=np.zeros([output_node,1])		
        t[T[m]] = 1.0
        del_k = (y-t)*d_sigmoid(uo)
        dEdv = np.dot(del_k,z.T)
        del_j = d_sigmoid(uh)*np.dot(v[:,1:].T,del_k)
        dEdw = np.dot(del_j,xl.T)
        v = v-eta*dEdv
        w = w-eta*dEdw
    E2 = 0
    for m in range(len(T)):
        xl[1:,0] = X[m]
        uh = np.dot(w,xl)
        z[1:,:] = sigmoid(uh)
        uo = np.dot(v,z)
        y = sigmoid(uo)
        t=np.zeros([output_node,1])		
        t[T[m]] = 1.0
        E2 = E2+np.sum((y-t)**2)
    E2 = E2/len(T)
    Resid = abs(E2-E1)
    E1=E2
    print("%i-th update and error is %f" %(Iter,E1))
    Iter+=1
print("The learning is finished")
# np.savez('learningdata', w=w,v=v) -> np.savez('model/learningdata', w=w, v=v)
# 학습된 가중치를 'model/learningdata.npz'로 저장 (model 폴더)
np.savez('model/learningdata', w=w, v=v)