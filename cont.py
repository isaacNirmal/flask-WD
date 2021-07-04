from flask import Flask
from flask import current_app

app = Flask(__name__)

class Test:
    
    def __init__(self) -> None:
        print("isaac")

c = Test()

@app.route('/')
def index():
    return 'Welcome!'


@app.route('/i')
def index1():
    return 'Welcome! -----------------'

if __name__ == '__main__':
    app.run()