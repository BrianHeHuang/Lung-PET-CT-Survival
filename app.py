from flask import Flask

from config import config

app = Flask(__name__)
app.config['DEBUG'] = config.DEBUG
app.config['SECRET_KEY'] = config.SECRET
app.config["SQLALCHEMY_DATABASE_URI"] = config.DB_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config['SQLALCHEMY_ECHO'] = config.PRINT_SQL
