from flask import Flask, request, render_template_string
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained pipeline + model
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

# Modern HTML Template with images/icons
html_page = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>House Price Predictor AI</title>
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap" rel="stylesheet">
  <style>
    body {
      margin:0;
      font-family: 'Poppins', sans-serif;
      background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
      color: #fff;
      display:flex; justify-content:center; align-items:center;
      min-height:100vh;
      overflow-x:hidden;
    }
    .container {
      background: rgba(255,255,255,0.06);
      backdrop-filter: blur(20px);
      border-radius: 20px;
      padding: 40px 50px;
      width: 90%;
      max-width: 700px;
      box-shadow: 0 15px 45px rgba(0,0,0,0.5);
      animation: slideUp 1.2s ease;
    }
    @keyframes slideUp {
      from { transform: translateY(60px); opacity:0; }
      to { transform: translateY(0); opacity:1; }
    }
    h1 {
      text-align:center;
      font-size:2.8rem;
      font-weight:800;
      margin-bottom:10px;
      background: linear-gradient(90deg, #00c6ff, #0072ff);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    h2 {
      text-align:center;
      font-size:1.2rem;
      font-weight:400;
      opacity:0.8;
      margin-bottom:30px;
    }

    /* Image banner */
    .banner {
      display:flex;
      justify-content:center;
      margin-bottom: 20px;
    }
    .banner img {
      width: 160px;
      filter: drop-shadow(0 0 20px #00c6ff);
      animation: pulse 3s infinite alternate;
    }
    @keyframes pulse {
      from { transform: scale(1); }
      to { transform: scale(1.1); }
    }

    /* Input Group */
    .input-group {
      display:flex;
      align-items:center;
      margin-bottom:18px;
      background: rgba(255,255,255,0.1);
      border-radius: 12px;
      padding: 12px;
      transition:0.3s;
    }
    .input-group:hover { background: rgba(255,255,255,0.2); }
    .input-group img {
      width:28px; height:28px; margin-right:12px;
      filter: invert(1);
    }
    input[type=text] {
      width:100%;
      padding:10px;
      border:none;
      background:transparent;
      color:#fff;
      font-size:1rem;
      outline:none;
    }

    /* Button */
    button {
      width:100%;
      padding:15px;
      font-size:1.2rem;
      border:none;
      border-radius:15px;
      margin-top:10px;
      background: linear-gradient(135deg, #00c6ff, #0072ff);
      color:#fff;
      font-weight:600;
      letter-spacing:1px;
      cursor:pointer;
      transition:0.4s;
    }
    button:hover {
      background: linear-gradient(135deg, #0072ff, #00c6ff);
      transform: scale(1.05);
      box-shadow: 0 8px 20px rgba(0,198,255,0.5);
    }

    /* Result */
    .result {
      margin-top:25px;
      padding:20px;
      border-radius:15px;
      font-size:1.4rem;
      text-align:center;
      font-weight:bold;
      animation: popIn 0.8s ease;
    }
    .healthy {
      background: rgba(0,255,150,0.2);
      border: 2px solid #00e676;
      color:#00e676;
      text-shadow: 0 0 12px #00e676;
    }
    .danger {
      background: rgba(255,0,80,0.2);
      border: 2px solid #ff1744;
      color:#ff1744;
      text-shadow: 0 0 12px #ff1744;
    }
    @keyframes popIn {
      from { opacity:0; transform: scale(0.8); }
      to { opacity:1; transform: scale(1); }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="banner">
      <img src=" https://cdn-icons-png.flaticon.com/512/616/616408.png " alt="House Icon">
    </div>
    <h1>House Price Prediction AI</h1>
    <h2>Enter details to check risk instantly ⚡</h2>
    
    <form method="POST">
      <div class="input-group">
        <img src="https://cdn-icons-png.flaticon.com/512/854/854878.png">
        <input type="text" placeholder="Longitude" name="a">
      </div>
      <div class="input-group">
        <img src="https://cdn-icons-png.flaticon.com/512/854/854878.png">
        <input type="text" placeholder="Latitude" name="b">
      </div>
      <div class="input-group">
        <img src="https://cdn-icons-png.flaticon.com/512/2933/2933245.png">
        <input type="text" placeholder="Housing Median Age" name="c">
      </div>
      <div class="input-group">
        <img src="https://cdn-icons-png.flaticon.com/512/1828/1828859.png">
        <input type="text" placeholder="Total Rooms" name="d">
      </div>
      <div class="input-group">
        <img src="https://cdn-icons-png.flaticon.com/512/1828/1828859.png">
        <input type="text" placeholder="Total Bedrooms" name="e">
      </div>
      <div class="input-group">
        <img src="https://cdn-icons-png.flaticon.com/512/1077/1077114.png">
        <input type="text" placeholder="Population" name="f">
      </div>
      <div class="input-group">
        <img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png">
        <input type="text" placeholder="Households" name="g">
      </div>
      <div class="input-group">
        <img src="https://cdn-icons-png.flaticon.com/512/929/929417.png">
        <input type="text" placeholder="Median Income" name="h">
      </div>
      <div class="input-group">
        <img src="https://cdn-icons-png.flaticon.com/512/535/535183.png">
        <input type="text" placeholder="Ocean Proximity" name="i">
      </div>
      <button type="submit">Predict Now 🚀</button>
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
        i = request.form["i"]
        
        input_feature = [[a,b,c,d,e,f,g,h,i]]
        df = pd.DataFrame(input_feature,columns=['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','ocean_proximity'])
        final = pipeline.transform(df)
        prediction = model.predict(final)
        
        result = f"Predicted House Price: {prediction[0]}"
    return render_template_string(html_page, result=result)

if __name__ == "__main__":
    app.run(debug=True)
