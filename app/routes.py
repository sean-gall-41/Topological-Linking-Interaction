import io

import numpy as np
from flask import Response, jsonify, render_template, request
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from app import app

from .generate_chain import generate_closed_chain
from .monte_carlo import basic_monte_carlo_sim
from .private.utilities import chain_to_JSON

N  = 100
NUM_CHAINS = 200

from .projection import find_reg_project, rot_saw_xy

# from .tests.intersect_unit_test import intersect_unit_test
# from .tests.populate_alexander_matrix_unit_test import \
#     populate_alexander_matrix_unit_test


@app.route("/data_helper", methods=["GET", "POST"])
def data_helper():
    """return json representation of SAW for GET request, return OK, 200 else.
    
    This function handles GET and POST requests from our web app. So far the 
    POST requests are quite simple, but in the future, we may handle more
    complex tasks here. The GET request is likewise very simply, only handing
    over the serialized SAW information to the web app."""
    if request.method == "POST": # POST request
        print("Incoming..")
        print(request.get_json())  # parse as JSON
        return "OK", 200

    else: # GET request
        chain = generate_closed_chain(N, shift=True)[0]
        payload = chain_to_JSON(chain)
        return jsonify(payload)


# @app.route("/plot.png", methods=["GET"])
# def plot_png():
#     fig = create_figure()
#     output = io.BytesIO()
#     FigureCanvas(fig).print_png(output)
#     return Response(output.getvalue(), mimetype='image/png')


# def create_figure():
#     raw_data = basic_monte_carlo_sim(N, NUM_CHAINS, table=True, shift=True)
#     hist, bin_edges = np.histogram(raw_data[:,1], 50, density=True)
#     fig = Figure()
#     axis = fig.add_subplot(111)
#     axis.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor="black",
#              align="edge")
#     return fig


@app.route("/")
@app.route("/index")
def index():
    # look inside 'templates' and serve 'index.html'
    return render_template("index.html")

# @app.after_request
# def add_header(response):
#     response.cache_control.max_age = 300
#     return response
