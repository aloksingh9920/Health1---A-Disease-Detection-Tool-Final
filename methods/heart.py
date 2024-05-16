import pickle
with open(r'methods\heartmodel.pickle','rb') as fp:
    model = pickle.load(fp)

with open(r'methods\heartscaler.pickle','rb') as fp:
    scaler = pickle.load(fp)
    
def predict_heart(lst):
    print(1)
    lst = scaler.transform([lst])
    print(2)
    y_pred = model.predict(lst)
    print(3)
    if y_pred[0]==1:
        ans = "Presence of Heart Disease"
    else:
        ans = "Absence of Heart Disease"
    print(4)
    return ans