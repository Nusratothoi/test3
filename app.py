from flask import Flask, request, render_template
import numpy as np
import pandas
import sklearn
import pickle

model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
mx = pickle.load(open('minmaxscaler.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")
@app.route("/predict",methods=['POST'])
def predict():
    ILTES = request.form['ILTES']
    TOEFL = request.form['TOEFL']
    SAT = request.form['SAT']
    GRE = request.form['GRE']
    WRITING = request.form['WRITING']
    VERBAL = request.form['VERBAL']

    feature_list = [ILTES, TOEFL, SAT, GRE, WRITING, VERBAL]
    single_pred = np.array(feature_list).reshape(1, -1)

    mx_features = mx.transform(single_pred)
    sc_mx_features = sc.transform(mx_features)
    prediction = model.predict(sc_mx_features)

    UniversityData_dict = {1: "oxford university", 2: "University of Cambridge",
                      3: "Imperial College London", 4: "Kings College London",
                      5: "London School lof Economics and Political Science",
                      6: "University of Manchester", 7: "University of Glasgow",
                      8: "University of Birmingham", 9: "University of Sheffield",
                      10: "University of Warwick", 11: "University of Bristol", 12: "UCL", 13: "University of Southampton",
                      14: "University of Leeds", 15: "University of Liverpool", 16: "Lancaster University", 17: "Durham University", 18: "University of Sussex",
                      19: "Aston University", 20: "City University of London", 21: "North umbria University"}

    if prediction[0] in UniversityData_dict:
       UniversityData = UniversityData_dict[prediction[0]]
       result = "{} Recommended".format(UniversityData)
    else:
       result = "Sorry."
    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)