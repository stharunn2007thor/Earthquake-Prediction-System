from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()


class Prediction(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    latitude = db.Column(db.Float)

    longitude = db.Column(db.Float)



    risk = db.Column(db.String(100))

    confidence = db.Column(db.Float)

    timestamp = db.Column(db.String(100))


class User(UserMixin, db.Model):

    id = db.Column(db.Integer, primary_key=True)

    username = db.Column(
        db.String(100),
        unique=True,
        nullable=False
    )

    email = db.Column(
        db.String(100),
        unique=True,
        nullable=False
    )

    password = db.Column(
        db.String(200),
        nullable=False
    )