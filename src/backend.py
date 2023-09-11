# Import Flask and queue modules
import random

from flask import Flask, request, jsonify
from main import evaluate_texts

# Create a Flask app instance
app = Flask(__name__)


@app.route("/evaluate", methods=["POST"])
def producer():
    # Get the data from the request body as JSON
    data = request.get_json()
    # Return a success message
    return jsonify(evaluate_texts(data["text"])[0]), 200
    #f = random.uniform(0, 1)
    #return jsonify({"isGPT": f > 0.5, "f":f}), 200


# Run the app on port 5000
if __name__ == "__main__":
    app.run(port=5000)
