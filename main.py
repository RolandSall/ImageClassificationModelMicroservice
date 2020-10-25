import flask

from flask import Flask
app = Flask(__name__)

@app.route('/')
def home():
    return 'Model Entry Point'


if __name__ == '__main__':
    app.run()