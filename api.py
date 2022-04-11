from flask import Flask, jsonify, render_template
from flask import request
from models import *
model = Models()
app = Flask(__name__)

@app.route('/api')
def index(methods=['GET']):
    q = request.args.get('q')
    res = jsonify({'answer': model.predict(q)})
    return res

@app.route('/')
def home(methods=['GET']):
    return render_template("index.html")

@app.route("/get")
def get_bot_response():    
    userText = request.args.get('q')    
    return model.predict(userText) 

if __name__ == '__main__':
    app.run(host = '127.0.0.1', debug = True )
