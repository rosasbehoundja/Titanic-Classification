from flask import Flask, redirect, render_template, request, url_for, session
import pickle

app = Flask(__name__)
app.secret_key = "titanic"

with open("models/rf.pkl", "rb") as file:
    rf_model = pickle.load(file)

@app.route("/", methods = ["POST", "GET"])
def index():
    if request.method == "POST":
        # validate inputs
        pclass = int(request.form["pclass"])
        age = float(request.form["age"])
        sibsp = int(request.form["sibsp"])
        parch = int(request.form["parch"])
        fare = float(request.form["fare"])
        sex_male = int(request.form["sex_male"])
        embarked = request.form["embarked"]

        embarked_Q = 1 if embarked == "Q" else 0
        embarked_S = 1 if embarked == "S" else 0

        session["features"] = [pclass, age, sibsp, parch, fare, sex_male, embarked_Q, embarked_S]
        return redirect(url_for("predict"))
    return render_template("index.html")

@app.route("/predict")
def predict():
    features = session.get("features")

    # prediction
    pred = rf_model.predict([features])[0] 
    result = "survive" if pred == 1 else "die"
    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)