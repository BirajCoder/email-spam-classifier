from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
cv = pickle.load(open("model/cv.pkl", 'rb'))
clf = pickle.load(open("model/clf.pkl", 'rb'))


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['post'])
def predict():
    email = request.form.get('email')
    # predict email
    print(email)
    X = cv.transform([email])
    prediction = clf.predict(X)
    prediction = 1 if prediction == 1 else -1
    return render_template('index.html', response=prediction)


if __name__ == "__main__":
    app.run(debug=True)
