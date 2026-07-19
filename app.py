import folium
import joblib
import requests
import numpy as np
import pandas as pd
import os

from datetime import datetime

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    send_file
)

from flask_bcrypt import Bcrypt

from flask_login import (
    LoginManager,
    login_user,
    logout_user,
    login_required,
    current_user
)

from flask_mail import Mail, Message

from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle
)

from reportlab.lib import colors

from models import db, Prediction, User


# ================= CREATE FLASK APP =================

app = Flask(__name__)

app.config['SECRET_KEY'] = "earthquake_secret"

app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///earthquake.db"

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


# ================= EMAIL CONFIGURATION =================



app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True

app.config['MAIL_USERNAME'] = 'tharunvivacious@gmail.com'
app.config['MAIL_PASSWORD'] = 'mhoytipoufssgxdw'

mail = Mail(app)


# ================= INITIALIZE DATABASE =================

db.init_app(app)


# ================= BCRYPT =================

bcrypt = Bcrypt(app)


# ================= LOGIN MANAGER =================

login_manager = LoginManager()

login_manager.init_app(app)

login_manager.login_view = "login"


@login_manager.user_loader
def load_user(user_id):

    return User.query.get(
        int(user_id)
    )


# ================= LOAD MODEL =================

model = joblib.load(
    "earthquake_model.pkl"
)
print("Model expects:", model.n_features_in_)

# ================= HOME PAGE =================

@app.route('/')
def home():

    return render_template(
        'index.html'
    )



# ================= PREDICTION =================

@app.route('/predict', methods=['POST'])
def predict():

    try:

        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])

        current = datetime.now()
        year = current.year
        month = current.month
        day = current.day
        hour = current.hour
        minute = current.minute

        features = np.array([[

      

            latitude,
            longitude,
            year,
            month,
            day,
            hour,
            minute
        ]])

        print(features)
        print("Shape:", features.shape)
        print("Model expects:", model.n_features_in_)

        prediction = model.predict(features)[0]

        probability = model.predict_proba(features)

        high_probability = probability[0][1] * 100

        low_probability = probability[0][0] * 100

        if prediction == 1:

            prediction_text = "⚠ HIGH EARTHQUAKE RISK"

            confidence = high_probability

            color = "danger"

        else:

            prediction_text = "✅ LOW EARTHQUAKE RISK"

            confidence = low_probability

            color = "success"

        current_time = datetime.now().strftime(
            "%d-%m-%Y %H:%M:%S"
        )

        new_prediction = Prediction(

        


             latitude=latitude,
             longitude=longitude,
             risk=prediction_text,
             confidence=round(confidence, 2),
             timestamp=current_time
        )






        

        db.session.add(new_prediction)

        db.session.commit()

        # ================= EMAIL ALERT =================

        if prediction == 1:

            msg = Message(

                "⚠ HIGH EARTHQUAKE RISK DETECTED",

                sender=app.config['MAIL_USERNAME'],

                recipients=[current_user.email]

            )

            msg.body = f"""

HIGH EARTHQUAKE RISK DETECTED

Latitude : {latitude}

Longitude : {longitude}



Confidence : {round(confidence,2)}%

Timestamp : {current_time}

"""

            mail.send(msg)

        return render_template(

            'index.html',

            prediction_text=prediction_text,

            confidence=round(confidence, 2),

            high_probability=round(high_probability, 2),

            low_probability=round(low_probability, 2),

            color=color

        )

    except Exception as e:

        return render_template(

            'index.html',

            prediction_text=f"Error : {str(e)}"

        )





print(model.n_features_in_)
print(os.path.abspath("earthquake_model.pkl"))


# ================= USER REGISTER =================

@app.route('/register', methods=['GET', 'POST'])
def register():

    if request.method == "POST":

        username = request.form['username']

        email = request.form['email']

        password = request.form['password']

        hashed_password = bcrypt.generate_password_hash(
            password
        ).decode('utf-8')

        user = User(

            username=username,

            email=email,

            password=hashed_password

        )

        db.session.add(user)

        db.session.commit()

        return redirect('/login')

    return render_template(
        'register.html'
    )


# ================= USER LOGIN =================

@app.route('/login', methods=['GET', 'POST'])
def login():

    if request.method == "POST":

        email = request.form['email']

        password = request.form['password']

        user = User.query.filter_by(
            email=email
        ).first()

        if user and bcrypt.check_password_hash(
                user.password,
                password):

            login_user(user)

            return redirect('/')

    return render_template(
        'login.html'
    )


# ================= LOGOUT =================

@app.route('/logout')
@login_required
def logout():

    logout_user()

    return redirect('/login')


# ================= HISTORY =================

@app.route('/history')
@login_required
def history():

    predictions = Prediction.query.all()

    return render_template(

        'history.html',

        predictions=predictions

    )


# ================= SEARCH =================

@app.route('/search', methods=['GET', 'POST'])
@login_required
def search():

    predictions = []

    if request.method == 'POST':

        risk = request.form['risk']

        confidence = request.form['confidence']

        date = request.form['date']

        query = Prediction.query

        # Risk Filter
        if risk != "ALL":

            query = query.filter(

                Prediction.risk.like(

                    f"%{risk}%"

                )

            )

        # Confidence Filter
        if confidence:

            query = query.filter(

                Prediction.confidence >=

                float(confidence)

            )

        # Date Filter
        if date:

            query = query.filter(

                Prediction.timestamp.like(

                    f"%{date}%"

                )

            )

        predictions = query.all()

    return render_template(

        'search.html',

        predictions=predictions

    )


# ================= MAP =================

@app.route('/map')
@login_required
def map_view():

    predictions = Prediction.query.all()

    if predictions:

        center_lat = predictions[0].latitude

        center_lon = predictions[0].longitude

    else:

        center_lat = 20

        center_lon = 0

    m = folium.Map(

        location=[center_lat, center_lon],

        zoom_start=4,

        tiles="CartoDB positron"

    )

    for row in predictions:

        if row.confidence >= 80:

            color = "red"

        elif row.confidence >= 50:

            color = "orange"

        else:

            color = "green"

        folium.CircleMarker(

            location=[

                row.latitude,

                row.longitude

            ],

            radius=max(

                5,

                row.confidence / 10

            ),

            color=color,

            weight=3,

            fill=True,

            fill_color=color,

            fill_opacity=0.8,

            tooltip=row.risk,

            popup=folium.Popup(

                f"""
                <h4>🌍 Earthquake Information</h4>

                <b>Risk Level:</b> {row.risk}<br>

                <b>Latitude:</b> {row.latitude}<br>

                <b>Longitude:</b> {row.longitude}<br>

                <b>Depth:</b> {row.depth} km<br>

                <b>Confidence:</b> {row.confidence}%<br>

                <b>Timestamp:</b> {row.timestamp}
                """,

                max_width=300

            )

        ).add_to(m)

    map_html = m._repr_html_()

    return render_template(

        "map_page.html",

        map_html=map_html

    )


# ================= LIVE EARTHQUAKE DATA =================

@app.route('/live')
@login_required
def live():

    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"

    response = requests.get(url)

    data = response.json()

    earthquakes = []

    for quake in data['features']:

        earthquakes.append({

            "place": quake['properties']['place'],

            "magnitude": quake['properties']['mag'],

            "latitude": quake['geometry']['coordinates'][1],

            "longitude": quake['geometry']['coordinates'][0],

            "depth": quake['geometry']['coordinates'][2]

        })

    return render_template(

        "live.html",

        earthquakes=earthquakes

    )



# ================= DASHBOARD =================

@app.route('/dashboard')
@login_required
def dashboard():

    predictions = Prediction.query.all()

    total_predictions = len(predictions)

    high_risk = len(
        [p for p in predictions
         if "HIGH" in p.risk]
    )

    low_risk = len(
        [p for p in predictions
         if "LOW" in p.risk]
    )

    confidence_values = [
        p.confidence
        for p in predictions
    ]

    prediction_numbers = list(
        range(
            1,
            total_predictions + 1
        )
    )

    latitudes = [
        p.latitude
        for p in predictions
    ]

    longitudes = [
        p.longitude
        for p in predictions
    ]

    average_confidence = round(

        sum(confidence_values) /
        len(confidence_values),

        2

    ) if confidence_values else 0

    return render_template(

        "dashboard.html",

        total_predictions=total_predictions,

        high_risk=high_risk,

        low_risk=low_risk,

        confidence_values=confidence_values,

        prediction_numbers=prediction_numbers,

        average_confidence=average_confidence,

        latitudes=latitudes,

        longitudes=longitudes

    )


# ================= EXPORT CSV =================

@app.route('/export_csv')
@login_required
def export_csv():

    predictions = Prediction.query.all()

    data = []

    for row in predictions:

        data.append({

            "Latitude": row.latitude,

            "Longitude": row.longitude,

            "Depth": row.depth,

            "Risk": row.risk,

            "Confidence": row.confidence,

            "Timestamp": row.timestamp

        })

    df = pd.DataFrame(data)

    filename = "earthquake_predictions.csv"

    df.to_csv(

        filename,

        index=False

    )

    return send_file(

        filename,

        as_attachment=True

    )


# ================= EXPORT EXCEL =================

@app.route('/export_excel')
@login_required
def export_excel():

    predictions = Prediction.query.all()

    data = []

    for row in predictions:

        data.append({

            "Latitude": row.latitude,

            "Longitude": row.longitude,

           

            "Risk": row.risk,

            "Confidence": row.confidence,

            "Timestamp": row.timestamp

        })

    df = pd.DataFrame(data)

    filename = "earthquake_predictions.xlsx"

    df.to_excel(

        filename,

        index=False

    )

    return send_file(

        filename,

        as_attachment=True

    )


# ================= EXPORT PDF =================

@app.route('/export_pdf')
@login_required
def export_pdf():

    predictions = Prediction.query.all()

    pdf = SimpleDocTemplate(

        "earthquake_predictions.pdf"

    )

    data = [[

        "Latitude",

        "Longitude",

      

        "Risk",

        "Confidence",

        "Timestamp"

    ]]

    for row in predictions:

        data.append([

            row.latitude,

            row.longitude,

            

            row.risk,

            row.confidence,

            row.timestamp

        ])

    table = Table(data)

    table.setStyle(

        TableStyle([

            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),

            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),

            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),

            ('GRID', (0, 0), (-1, -1), 1, colors.black)

        ])

    )

    pdf.build([table])

    return send_file(

        "earthquake_predictions.pdf",

        as_attachment=True

    )
# ================= ADMIN PANEL =================

@app.route('/admin')
@login_required
def admin():

    users = User.query.all()

    predictions = Prediction.query.all()

    return render_template(

        'admin.html',

        users=users,

        predictions=predictions

    )


# ================= DELETE USER =================

@app.route('/delete_user/<int:id>')
@login_required
def delete_user(id):

    user = User.query.get_or_404(id)

    db.session.delete(user)

    db.session.commit()

    return redirect('/admin')


# ================= DELETE PREDICTION =================

@app.route('/delete_prediction/<int:id>')
@login_required
def delete_prediction(id):

    prediction = Prediction.query.get_or_404(id)

    db.session.delete(prediction)

    db.session.commit()

    return redirect('/admin')



# ================= SEND TEST EMAIL =================

@app.route('/send_email')
@login_required
def send_email():

    msg = Message(

        "Earthquake Prediction System",

        sender=app.config['MAIL_USERNAME'],

        recipients=[current_user.email]

    )

    msg.body = """

Earthquake Prediction System

Email Service Working Successfully.

"""

    mail.send(msg)

    return "Email Sent Successfully"


# ================= CREATE DATABASE =================

with app.app_context():

    db.create_all()


# ================= RUN APP =================

if __name__ == "__main__":

    app.run(

        debug=True

    )

