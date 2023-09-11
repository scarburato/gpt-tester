# Import Flask and queue modules
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
   # return jsonify({"isGPT": False, "lp":0.5}), 200


# Run the app on port 5000
if __name__ == "__main__":
    app.run(port=5000)
