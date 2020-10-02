from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import svm

# mnist=fetch_openml('mnist_784',data_home='custom_data_home')
# X,y=mnist['data'],mnist['target']
# np.save('mnist-file_X', X)
# np.save('mnist-file_y', y)
X=np.load('mnist-file_X.npy',allow_pickle=True)
y=np.load('mnist-file_y.npy',allow_pickle=True)



x_train,x_test=X[:6000],X[6000:7000]
y_train,y_test=y[:6000],y[6000:7000]

# shuffle_index=np.random.permutation(6000)
# x_train,y_train=x_train[shuffle_index],y_trian[shuffle_index]

# 7:
random_digit=x_train[4000]
# digit_reshaped=random_digit.reshape(28,28)
# plt.imshow(digit_reshaped,cmap=matplotlib.cm.binary,interpolation="nearest")
# plt.show()

y_train_for_7=(y_train.astype(np.int8)==7)
y_test_for_7=(y_test.astype(np.int8)==7)

# print(y_train_for_7[4000])

# clf=LogisticRegression(tol=0.1,max_iter=2000) #LogisticRegression
clf=svm.SVC() #Support Vector Machine
clf.fit(x_train,y_train_for_7)
y_pred=clf.predict([random_digit])

cross_validation=cross_val_score(clf,x_train,y_train_for_7,cv=3,scoring="accuracy")

print(cross_validation.mean())

cross_validation=cross_val_score(clf,x_test,y_test_for_7,cv=3,scoring="accuracy")

print(cross_validation.mean())


# print(l.__class__)
