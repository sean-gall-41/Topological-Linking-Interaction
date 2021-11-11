from flask import Flask

app = Flask(__name__)

# for now set the configuration to debug.
# app.config['DEBUG'] = True

from app import routes
