from data import get_training_data
from perceptron import Perceptron
from sklearn.metrics import accuracy_score

data = get_training_data()
X = data['dataset']
y = data['targets']

model = Perceptron(input_size=4)

model.train(X,y,10000)

preds = model.predict(X)
print(preds)

preds_test = model.predict([[1,0,1,0]])
print(preds_test)

print("Accuracy train data:", accuracy_score(y, preds))
print("Accuracy test data:", accuracy_score([0], preds_test))