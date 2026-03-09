from flask import Flask, render_template, request, jsonify
from src.rag_pipeline import answer_query

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")

    if not user_message:
        return jsonify({"reply": "Please enter a question."})

    answer = answer_query(user_message)

    return jsonify({"reply": answer})


if __name__ == "__main__":
    app.run(debug=True)