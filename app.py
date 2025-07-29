
from flask import Flask, request, render_template, jsonify,url_for
import joblib
import numpy as np
import plotly.graph_objects as go
app = Flask(__name__)


model = joblib.load('predictor_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:

        strike_diff = (float(request.form['strike_diff100'])/100)
        td_acc_diff = (float(request.form['td_acc_diff100'])/100)
        avgsubatt = float(request.form['avgsubattblue'])-float(request.form['avgsubattred'])
        redwinpct = (float(request.form['redwinpct100'])/100)
        bluewinpct = (float(request.form['bluewinpct100'])/100)
        BlueCurrentWinStreak = int(request.form['BlueCurrentWinStreak'])  
        RedCurrentWinStreak = int(request.form['RedCurrentWinStreak'])    
        HeightDif = float(request.form['Heightblue'])-float(request.form['Heightred'])
        ReachDif = float(request.form['Reachblue'])-float(request.form['Reachred'])
        RGrapplerBStriker = int(request.form['RGrapplerBStriker'])       
        BGrapplerRStriker = int(request.form['BGrapplerRStriker'])

        ranked = request.form.get('ranked', 'none')
        redranked = 1 if ranked == 'red' else 0
        blueranked = 1 if ranked == 'blue' else 0

        features = [strike_diff, td_acc_diff,avgsubatt,BlueCurrentWinStreak,RedCurrentWinStreak,HeightDif,ReachDif,redwinpct,bluewinpct,redranked,blueranked,RGrapplerBStriker,BGrapplerRStriker]

        final_input = scaler.transform(np.array(features).reshape(1, -1))
        prediction = model.predict(final_input)[0]
        proba = model.predict_proba(final_input)[0] 
        blue_prob = round(proba[0] * 100, 1) 
        red_prob = round(proba[1] * 100, 1)
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=red_prob,
            domain={'x': [0, 0.45], 'y': [0, 1]},
            title={'text': "RED", 'font': {'color': "#d20a0a"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "#d20a0a"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#333",
                'steps': [
                    {'range': [0, 50], 'color': "#ffebee"},
                    {'range': [50, 100], 'color': "#ffcdd2"}
                ]
            }
        ))

        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=blue_prob,
            domain={'x': [0.55, 1], 'y': [0, 1]},
            title={'text': "BLUE", 'font': {'color': "#1e88e5"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "#1e88e5"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#333",
                'steps': [
                    {'range': [0, 50], 'color': "#e3f2fd"},
                    {'range': [50, 100], 'color': "#bbdefb"}
                ]
            }
        ))

        fig.update_layout(
            margin=dict(t=50, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            annotations=[
                dict(
                    text=f"<b>RED {red_prob}% vs BLUE {blue_prob}%</b>",
                    x=0.5, y=0.1,
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(size=16)
                )
            ]
        )
        winner = "RED" if prediction == 1 else "RED"
        return jsonify({
            'figure': fig.to_dict(),
            'prediction_text': f"Red: {red_prob}% | Blue: {blue_prob}%",
            'winner':winner
        })


                            
    

    except ValueError as e:
        return f"Invalid input: {str(e)}", 400
    except KeyError as e:
        return f"Missing field: {str(e)}", 400
    except Exception as e:
        return f"An error occurred: {str(e)}", 500    
if __name__ == '__main__':
    app.run(debug=True)