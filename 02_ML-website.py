from flask import Flask, request, render_template_string
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer

app = Flask(__name__)

# Load the trained pipeline and feature list
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

# HTML Template (futuristic premium UI)
html_page = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>🏠 Dr Strokey - House Price Predictor</title>
  <style>
    /* ====== GLOBAL ====== */
    body {
      font-family: 'Poppins', sans-serif;
      margin: 0;
      padding: 0;
      height: 100vh;
      display: flex;
      justify-content: center;
      align-items: center;
      background: radial-gradient(circle at top left, #1e90ff, #7f37ff, #ff0080, #ff8c00);
      background-size: 400% 400%;
      animation: gradientShift 15s ease infinite;
      overflow: hidden;
    }

    @keyframes gradientShift {
      0% { background-position: 0% 50%; }
      50% { background-position: 100% 50%; }
      100% { background-position: 0% 50%; }
    }

    /* Floating glowing spheres */
    .orb {
      position: absolute;
      width: 200px;
      height: 200px;
      border-radius: 50%;
      filter: blur(150px);
      opacity: 0.6;
      animation: floaty 20s infinite ease-in-out alternate;
      z-index: 0;
    }
    .orb:nth-child(1) { background: #00ffe7; top: 10%; left: 10%; animation-duration: 25s; }
    .orb:nth-child(2) { background: #ff0080; bottom: 20%; right: 10%; animation-duration: 30s; }
    .orb:nth-child(3) { background: #ffe600; top: 50%; left: 70%; animation-duration: 22s; }

    @keyframes floaty {
      from { transform: translateY(0px) translateX(0px); }
      to { transform: translateY(-60px) translateX(40px); }
    }

    /* ====== CONTAINER ====== */
    .container {
      position: relative;
      z-index: 1;
      background: rgba(255,255,255,0.08);
      backdrop-filter: blur(25px);
      border-radius: 25px;
      padding: 45px 55px;
      max-width: 650px;
      width: 90%;
      box-shadow: 0 12px 50px rgba(0,0,0,0.5);
      animation: dropIn 1.5s cubic-bezier(.19,1,.22,1);
    }

    @keyframes dropIn {
      from { opacity:0; transform: translateY(-100px) scale(0.9); }
      to { opacity:1; transform: translateY(0) scale(1); }
    }

    h1 {
      text-align: center;
      font-size: 2.8rem;
      margin-bottom: 25px;
      background: linear-gradient(to right, #00ffe7, #ff0080, #ffe600);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      text-shadow: 0 0 25px rgba(255,255,255,0.4);
      animation: glowPulse 2s infinite alternate;
    }

    @keyframes glowPulse {
      from { text-shadow: 0 0 10px #fff, 0 0 25px #ff0080; }
      to { text-shadow: 0 0 25px #00ffe7, 0 0 45px #ffe600; }
    }

    /* ====== INPUTS ====== */
    input[type=text] {
      width: 100%;
      padding: 15px;
      margin-top: 15px;
      border: none;
      border-radius: 12px;
      outline: none;
      background: rgba(255,255,255,0.85);
      font-size: 1rem;
      transition: all 0.35s ease;
    }
    input[type=text]:focus {
      background: #fff;
      transform: scale(1.05);
      box-shadow: 0 0 15px rgba(30,144,255,0.9);
    }

    /* ====== BUTTON ====== */
    button {
      background: linear-gradient(135deg, #ff0080, #1e90ff, #7f37ff);
      color: white;
      padding: 16px;
      border: none;
      width: 100%;
      margin-top: 25px;
      border-radius: 15px;
      cursor: pointer;
      font-size: 1.2rem;
      font-weight: 700;
      letter-spacing: 1px;
      transition: all 0.4s ease;
      position: relative;
      overflow: hidden;
    }
    button::before {
      content:"";
      position:absolute;
      top:0; left:-100%;
      width:100%; height:100%;
      background: rgba(255,255,255,0.3);
      transform:skewX(-25deg);
      transition: all 0.7s ease;
    }
    button:hover::before { left:100%; }
    button:hover {
      transform: translateY(-3px) scale(1.05);
      box-shadow: 0 8px 25px rgba(0,0,0,0.45);
    }

    /* ====== RESULT ====== */
    .result {
      margin-top: 30px;
      padding: 20px;
      border-radius: 15px;
      font-size: 1.4rem;
      font-weight: bold;
      text-align: center;
      animation: fadePop 1s ease-out;
    }
    .healthy {
      background: rgba(0,255,150,0.15);
      border: 2px solid #00e676;
      color: #00e676;
      text-shadow: 0 0 10px #00e676;
    }
    .danger {
      background: rgba(255,0,80,0.15);
      border: 2px solid #ff1744;
      color: #ff1744;
      text-shadow: 0 0 10px #ff1744;
    }

    @keyframes fadePop {
      from { opacity:0; transform:scale(0.7); }
      to { opacity:1; transform:scale(1); }
    }
  </style>
</head>
<body>
  <!-- glowing background orbs -->
  <div class="orb"></div>
  <div class="orb"></div>
  <div class="orb"></div>

  <div class="container">
    <h1>🏠 Dr Strokey</h1>
    <form method="POST" enctype="multipart/form-data">
      <input type="text" placeholder="Longitude" name="a"><br>
      <input type="text" placeholder="Latitude" name="b"><br>
      <input type="text" placeholder="Housing Median Age" name="c"><br>
      <input type="text" placeholder="Total Rooms" name="d"><br>
      <input type="text" placeholder="Total Bedrooms" name="e"><br>
      <input type="text" placeholder="Population" name="f"><br>
      <input type="text" placeholder="Households" name="g"><br>
      <input type="text" placeholder="Median Income" name="h"><br>
      <input type="text" placeholder="Ocean Proximity" name="i"><br>
      <button type="submit">⚡ Predict Now</button>
    </form>

    {% if result %}
       <div class="result {% if 'not' in result %}healthy{% else %}danger{% endif %}">{{ result }}</div>
    {% endif %}
  </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        a = float(request.form["a"])
        b = float(request.form["b"])
        c = float(request.form["c"])
        d = float(request.form["d"])
        e = float(request.form["e"])
        f = float(request.form["f"])
        g = float(request.form["g"])
        h = float(request.form["h"])
        i = float(request.form['i'])
        
        input_feature = [[a,b,c,d,e,f,g,h,i]]
        dataFrame_feature = pd.DataFrame(input_feature,columns=['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','ocean_proximity'])
        final_feature_upgrade = pipeline.transform(dataFrame_feature)
        prediction = model.predict(final_feature_upgrade)
        result = f"💰 Predicted House Price: ${prediction[0]:,.2f}"
    
    return render_template_string(html_page, result=result)

if __name__ == "__main__":
    app.run(debug=True)
