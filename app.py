from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# 创建一个主页
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata",methods= ["GET","POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        pass


if __name__ == "__main__":
    app.run(debug=True)