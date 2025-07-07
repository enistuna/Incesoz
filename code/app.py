from flask import Flask
from network import network

app = Flask(__name__)
app.register_blueprint(network, url_prefix="/")


if __name__ == '__main__':
    app.run()
