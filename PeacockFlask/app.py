from flask import Flask, render_template, url_for
app = Flask(__name__)


@app.route("/")
@app.route("/home")
def home():
    return render_template('index.html')

@app.route("/visMode")
def visMode():
    return render_template('visMode.html')


if __name__ == '__main__':
    app.run(debug=True)