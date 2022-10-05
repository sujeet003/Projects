from flask import Flask, request, render_template
import pickle
import numpy as np
app=Flask(__name__,template_folder='./templates',static_folder='./templates/static')#
# app = Flask(__name__,template_folder='./frontend/templates',static_folder='./frontend/static')

model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():
    int_features = [float(x)for x in request.form.values()]# if x else 0 
#if above line does not fetch expected output, try code given below:
#   int_features = [float(x) for x in list(request.form.values())]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    if prediction == 0:
        output = "This song is not popular."
    else :
        output = "This song is poplar."
    return render_template('index.html', result=output)
    
if __name__ =="__main__":
    app.run(debug=True)





# from flask import Flask, request, render_template
# import pickle
# import numpy as np

# app=Flask(__name__,template_folder='./templates')

# model = pickle.load(open("model.pkl", "rb"))

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict',methods=['POST'])
# def predict():
#     # if request.method=='POST':
#         # int_features = [float(x) for x in request.form.values()]
#     #if above line does not fetch expected output, try code given below:
#         int_features = [float(x) for x in list(request.form.values())]
#         # print(request.__dict__)
#         # final_features = [np.array(int_features)]
#         # print(final_features)
#         # print("int_features: ",int_features)

#         final_features = [{'song_duration_ms':int_features[0], 'acousticness':int_features[1], 'danceability':int_features[2], 'energy':int_features[3], 'instrumentalness':int_features[4], 'key':int_features[5], 'liveness':int_features[6], 'loudness':int_features[7], 'audio_mode':int_features[8], 'speechiness':int_features[9], 'tempo':int_features[10], 'time_signature':int_features[11], 'audio_valence':int_features[12]}]
#         # final_features= np.array(final_features).reshape(-1,1)#error
#         # final_features.shape
#         prediction = model.predict(final_features[0])
#         if prediction == 0:
#             output = "This song is not popular."
#         else :
#             output = "This song is popular."
#         return render_template('index.html', result=output)
# if __name__ =="__main__":
#     # app.debug = True
#     app.run(debug=True)