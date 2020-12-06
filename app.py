from flask import Flask
from flask import render_template
from flask import request
from predictions import getUserPreds
from predictions import getCommentPreds
import json

app = Flask(__name__)

'''
Run using: python -m flask run
'''

@app.route('/', methods = ['GET'])
def start():
    return(render_template('index.html'))

@app.route('/username', methods = ['POST'])
def username():
    return(getUserPreds(request.form['username']))

@app.route('/comment', methods = ['POST'])
def comment():
    return(getUserPreds(request.form['comment']))