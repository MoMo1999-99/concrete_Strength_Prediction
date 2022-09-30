from flask import Flask ,render_template,request,jsonify
import joblib
import xgboost as xgb

app = Flask(__name__)
model = xgb.XGBRegressor()
model.load_model('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/' , methods=['Get'])
def home():
    
    return render_template('index.html')


@app.route('/predict' , methods=['Get'])
def predict():
    
    inp_data = [
        
        request.args.get('Cement in kg/m3'),
        request.args.get('Blast furnace slag in kg/m3'),
        request.args.get('Fly ash in kg/m3'),
        request.args.get('Water in liters/m3'),
        request.args.get('Superplasticizer additive in kg/m3'),
        request.args.get('Coarse aggregate (gravel) in kg/m3'),
        request.args.get('Fine aggregate (sand) in kg/m3'),
        request.args.get('Age of the sample in days')
        
    ]

    stree = model.predict(scaler.transform([inp_data]))[0]
    strength = str(stree)
    return render_template('index.html' , strength = strength)

if  __name__ == '__main__':
    
    app.run(debug=True)
