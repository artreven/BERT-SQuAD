import os

from flask import Flask,request,jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from bert import QA

app = Flask(__name__)
CORS(app)

load_dotenv()
model = QA(os.getenv("OUTPUT_DIR"))


@app.route("/predict", methods=['POST'])
def predict():
    doc = request.json["document"]
    q = request.json["question"]
    try:
        out = model.predict(doc,q)
        return jsonify({"result":out})
    except Exception as e:
        app.logger.warning(e)
        return jsonify({"result":"Model Failed"})


if __name__ == "__main__":
    app.run('0.0.0.0', port=8000, debug=True)