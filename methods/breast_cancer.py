from sklearn.tree import DecisionTreeClassifier
import pickle
with open(r'methods\breast_cancer_model.pickle','rb') as fp:
    model = pickle.load(fp)


def predict_breast_cancer(lst):
    y_pred = model.predict([lst])
    if y_pred[0]=="M":
        ans = "Malignant"
    else:
        ans = "Benign"
    return ans