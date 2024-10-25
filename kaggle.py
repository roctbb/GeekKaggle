import json
import uuid
from functools import wraps

from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import pickle
import os
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from celery import Celery
from sklearn import metrics

load_dotenv()

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL',
                                                  'postgresql+psycopg2://username:password@localhost:5432/mydatabase')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your_secret_key')

db = SQLAlchemy(app)
migrate = Migrate(app, db)
celery = Celery(app.name, broker=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'))
celery.conf.update(app.config)


def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def __repr__(self):
        return f'<User {self.username}>'


class Competition(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(256), unique=True, nullable=False)
    description = db.Column(db.Text(), nullable=False)
    tests_path = db.Column(db.String(256), nullable=False)
    answers_path = db.Column(db.String(256), nullable=False)
    metric = db.Column(db.String(50), nullable=False, default='accuracy')

    def __repr__(self):
        return f'<Competition {self.name}>'


class Solution(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user_model.id'), nullable=False)
    competition_id = db.Column(db.Integer, db.ForeignKey('competition_model.id'), nullable=False)
    file_path = db.Column(db.String(120), nullable=False)
    result = db.Column(db.Text(), nullable=True)

    user = db.relationship('User', backref=db.backref('solutions', lazy=True))
    competition = db.relationship('Competition', backref=db.backref('solutions', lazy=True))

    def __repr__(self):
        return f'<Solution {self.id}>'


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if not username or not password:
            return render_template('register.html', error='Username and password are required')

        if User.query.filter_by(username=username).first():
            return render_template('register.html', error='Username already exists')

        hashed_password = generate_password_hash(password, method='sha256')
        new_user = User(username=username, password_hash=hashed_password)

        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if not username or not password:
            return render_template('login.html', error='Username and password are required')

        user = User.query.filter_by(username=username).first()
        if user is None or not check_password_hash(user.password_hash, password):
            return render_template('login.html', error='Invalid username or password')

        session['user_id'] = user.id
        return redirect(url_for('competitions'))
    return render_template('login.html')


@app.route('/competitions', methods=['GET'])
@require_auth
def competitions():
    competition_list = Competition.query.all()
    return render_template('competitions.html', competitions=competition_list)


@app.route('/competition/<int:competition_id>', methods=['GET'])
@require_auth
def competition(competition_id):
    user_id = session['user_id']
    competition = Competition.query.get_or_404(competition_id)
    solutions = Solution.query.filter_by(competition_id=competition.id).all()
    return render_template('competition.html', competition=competition, solutions=solutions, user_id=user_id)


@app.route('/competition/<int:competition_id>/upload', methods=['POST'])
@require_auth
def upload_model(competition_id):
    file = request.files['file']

    if not file:
        return render_template('upload.html', error='File is required')

    if not file.filename.endswith('.pkl'):
        return render_template('upload.html', error='Only pickle files are allowed')

    filename = f"{uuid.uuid4()}_{file.filename}"
    model_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(model_path)

    user = User.query.filter_by(id=session['user_id']).first()
    solution = Solution(user_id=user.id,
                        competition_id=competition_id,
                        file_path=model_path)

    db.session.add(solution)
    db.session.commit()

    validate_model_task.apply_async(args=[solution.id])

    return redirect('/competition/' + str(competition_id))


@celery.task
def validate_model_task(solution_id):
    with app.app_context():
        solution = Solution.query.get(solution_id)
        model_path = solution.file_path

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        with open(solution.competition.tests_path, 'r') as f:
            tests = json.load(f)

        with open(solution.competition.tests_path, 'r') as f:
            answers = json.load(f)

        predictions = model.predict(answers)

        # Calculate the metric
        if solution.competition.metric == 'r2':
            result = metrics.r2_score(answers, predictions)
        elif solution.competition.metric == 'rmse':
            result = metrics.mean_squared_error(answers, predictions, squared=False)
        elif solution.competition.metric == 'accuracy':
            result = metrics.accuracy_score(answers, predictions)
        else:
            result = 'Invalid metric'

        solution.result = result
        db.session.commit()


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
