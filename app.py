# from crypt import methods
from flask import Flask, render_template, request
from process_chatbot import preparation, generate_response
from model_rs import recommendation
import pickle

# download nltk
preparation()

#Start Chatbot
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/rekomedasi_System")
def rs():
    return render_template("rs.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    Nama = request.form['wisata']
    rekomendasi = recommendation(Nama)
    
    return render_template('rs.html', rekomen=rekomendasi)

@app.route("/chatbot")
def cb():
    return render_template('chatbot.html')

@app.route("/get")
def get_bot_response():
    user_input = str(request.args.get('msg'))
    result = generate_response(user_input)
    return result

if __name__ == "__main__":
    app.run(debug=True)