from flask import Flask ,render_template,url_for,request
import numpy as np 
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

# Home Route
@app.route('/')
def home():
	return render_template('home.html')

# prediction
@app.route('/predict',methods=['POST'])
def predict():
	int_feature = [x for x in request.form.values()]
	print(int_feature)
	int_feature = [float(i) for i in int_feature]
	final_features = [np.array(int_feature)]
	prediction = model.predict(final_features)

	output = prediction
	print(output)

	return render_template('home.html',prediction_text= output)


if __name__ == "__main__":
	app.run(debug=True)
