from flask import Flask, render_template, request
import pickle
import pandas as pd
import joblib as jb

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/literature")
def literature():
    return render_template("literature.html")


@app.route("/material")
def material():
    return render_template("material.html")


@app.route("/algorithms")
def performance():
    return render_template("algorithms.html")


@app.route("/test")
def test():
    return render_template("test.html")


@app.route("/test", methods=["POST"])
def prd():
    music = dict()
    music["Popularity"] = request.form["Popularity"]
    music["danceability"] = request.form["danceability"]
    music["energy"] = request.form["energy"]
    music["loudness"] = request.form["loudness"]
    music["speechiness"] = request.form["speechiness"]
    music["acousticness"] = request.form["acousticness"]
    music["instrumentalness"] = request.form["instrumentalness"]
    music["liveness"] = request.form["liveness"]
    music["valence"] = request.form["valence"]
    music["tempo"] = request.form["tempo"]
    music["duration_in min/ms"] = request.form["duration_ms"]
    music["key"] = request.form["key"]
    music["mode"] = request.form["mode"]
    music["time_signature"] = request.form["time_signature"]
    

    Input = pd.DataFrame(
        data=[[music["Popularity"], music["danceability"], music["energy"], music["loudness"], music["speechiness"], music["acousticness"],
               music["instrumentalness"], music["liveness"],
               music["valence"], music["tempo"], music["duration_in min/ms"], music["key"], music["mode"],
               music["time_signature"]]],
        columns=['Popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                 'valence', 'tempo', 'duration_in min/ms', 'key', 'mode', 'time_signature'])
    output = Input.shape

    Ans = predict(Input)

    return render_template('test.html', output=Ans) 


def predict(Input):
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    model = pickle.load(open('finalized_model.pickle', 'rb'))

    Input[
        ['Popularity','danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence',
         'tempo', 'duration_in min/ms']] = scaler.fit_transform(Input[['Popularity', 'danceability', 'energy', 'loudness', 'speechiness',
                                                                'acousticness', 'instrumentalness', 'liveness',
                                                                'valence', 'tempo', 'duration_in min/ms']])

    Prediction = model.predict(Input)
    result = ""

    if (Prediction == 1):
        result = "Alternative Music"
    elif (Prediction == 0):
        result = "Acoustic Music"
    elif (Prediction == 7):
        result = "Instrumental Music"
    elif (Prediction == 9):
        result = "Pop Music"
    
    print(result)

    return (result)


if __name__ == "__main__":
    app.run(debug=True)
