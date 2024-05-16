from flask import Flask, request, render_template, flash
from PIL import Image
import io
import numpy as np
from werkzeug.utils import secure_filename
import os

# importing methods
from methods.tumor import predict_tumor
from methods.CovidPneumonia import predict_CovidPneumonia
from methods.Alzheimer import predict_Alzheimer
from methods.breast_cancer import predict_breast_cancer
from methods.diabeties import predict_diabeties
from methods.heart import predict_heart

UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_image(image):
    # Your image processing code here
    # For demonstration, let's just return the dimensions of the image
    width, height = image.size
    return {"width": width, "height": height}


@app.route('/', methods = ['GET'])
def index():
    return render_template('index.html')



@app.route('/brain_tumor_detection', methods=['GET', 'POST'])
def brain_tumor_detection():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return 'error'
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return 'error no selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            result = predict_tumor(filename)
            tumor_img = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            return render_template('tumor2.html',result=result,tumor_image=tumor_img)
    return render_template('tumor.html')
# Define routes for other detection tasks similarly


@app.route('/CovidPneumonia_detection', methods=['GET', 'POST'])
def CovidPneumonia_detection():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return 'error'
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return 'error no selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            results = predict_CovidPneumonia(filename)
            CovidPneumonia_img = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            return render_template('CovidPneumonia2.html',result=results,CovidPneumonia_image=CovidPneumonia_img)
    return render_template('CovidPneumonia.html')



@app.route('/Alzheimer_detection', methods=['GET', 'POST'])
def Alzheimer_detection():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return 'error'
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return 'error no selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            results = predict_Alzheimer(filename)
            Alzheimer_img = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            return render_template('Alzheimer2.html',result=results,Alzheimer_image=Alzheimer_img)
    return render_template('Alzheimer.html')





@app.route('/Breast_Cancer__detection', methods=['GET', 'POST'])
def Breast_Cancer_detection():
    if request.method == 'POST':
        # check if the post request has the file part
        try:
            radius_mean=request.form['radius_mean']
            texture_mean=request.form['texture_mean']
            perimeter_mean=request.form['perimeter_mean']
            area_mean=request.form['area_mean']
            smoothness_mean=request.form['smoothness_mean']
            compactness_mean=request.form['compactness_mean']
            concavity_mean=request.form['concavity_mean']
            concave_points_mean=request.form['concave_points_mean']
            symmetry_mean=request.form['symmetry_mean']
            fractal_dimension_mean=request.form['fractal_dimension_mean']
            radius_se=request.form['radius_se']
            texture_se=request.form['texture_se']
            perimeter_se=request.form['perimeter_se']
            area_se=request.form['area_se']
            smoothness_se=request.form['smoothness_se']
            compactness_se=request.form['compactness_se']
            concavity_se=request.form['concavity_se']
            concave_points_se=request.form['concave_points_se']
            symmetry_se=request.form['symmetry_se']
            fractal_dimension_se=request.form['fractal_dimension_se']
            radius_worst=request.form['radius_worst']
            texture_worst=request.form['texture_worst']
            perimeter_worst=request.form['perimeter_worst']
            area_worst=request.form['area_worst']
            smoothness_worst=request.form['smoothness_worst']
            compactness_worst=request.form['compactness_worst']
            concavity_worst=request.form['concavity_worst']
            concave_points_worst=request.form['concave_points_worst']
            symmetry_worst=request.form['symmetry_worst']
            fractal_dimension_worst=request.form['fractal_dimension_worst']
            lst = [radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst]
            results = predict_breast_cancer(lst)
            return render_template('breast_cancer2.html',result=results)
        except Exception as e:
            return e
    return render_template('breast_cancer.html')


@app.route('/diabeties_detection', methods=['GET', 'POST'])
def diabeties_detection():
    if request.method == 'POST':
        # check if the post request has the file part
        try:
            Pregnancies=request.form['Pregnancies']
            Glucose=request.form['Glucose']
            BloodPressure=request.form['BloodPressure']
            SkinThickness=request.form['SkinThickness']
            Insulin=request.form['Insulin']
            BMI=request.form['BMI']
            lst = [Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI]
            results = predict_diabeties(lst)
            return render_template('diabeties2.html',result=results)
        except Exception as e:
            return e
    return render_template('diabeties.html')


@app.route('/heart_detection', methods=['GET', 'POST'])
def heart_detection():
    if request.method == 'POST':
        # check if the post request has the file part
        try:
            Age=request.form['Age']
            Sex=request.form['Sex']
            Chest_pain_type=request.form['Chest_pain_type']
            BP=request.form['BP']
            Cholesterol=request.form['Cholesterol']
            FBS_over_120=request.form['FBS_over_120']
            EKG_results=request.form['EKG_results']
            Exercise_angina=request.form['Exercise_angina']
            ST_depression=request.form['ST_depression']
            Slope_of_ST=request.form['Slope_of_ST']
            Number_of_vessels_fluro=request.form['Number_of_vessels_fluro']
            Thallium=request.form['Thallium']
            lst = [Age,Sex,Chest_pain_type,BP,Cholesterol,FBS_over_120,EKG_results,Exercise_angina,ST_depression,Slope_of_ST,Number_of_vessels_fluro,Thallium]
            results = predict_heart(lst)
            return render_template('heart2.html',result=results)
        except Exception as e:
            return e
    return render_template('heart.html')


if __name__ == '__main__':
    app.run(debug=True)








# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file:
#             try:
#                 contents = file.read()
#                 image = Image.open(io.BytesIO(contents))
#                 result = process_image(image)
#                 return render_template('result.html', result=result)
#             except Exception as e:
#                 return str(e)
#     return render_template('index.html')


# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file:
#             try:
#                 contents = file.read()
#                 image = Image.open(io.BytesIO(contents))
#                 result = process_image(image)
#                 return render_template('result.html', result=result)
#             except Exception as e:
#                 return str(e)
#     return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
