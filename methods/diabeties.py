
import pickle
with open(r'methods\diabetesmodel.pickle','rb') as fp:
    model = pickle.load(fp)


def predict_diabeties(lst):
    y_pred = model.predict([lst])
    if y_pred[0]==1:
        ans = "Diabetic"
    else:
        ans = "Non-Diabetic"
    return ans