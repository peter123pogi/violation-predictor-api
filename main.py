from flask import Flask, jsonify, request
from flask_cors import CORS
from webpush import WebPush
from model_training.config_model import model_vp, model_cca

app = Flask(__name__)
CORS(app)


@app.route('/python/incident/risk', methods=['POST'])
def incident_risk():
    data = request.get_json()
    if not data or 'list' not in data:
        return jsonify({"status": "error", "message": "Invalid payload"}), 400
    
    df = model_vp.generate_watchlist(data['list'])
        
    return jsonify({
        'data': df,
    })

@app.route('/python/complaint/context', methods=['POST'])
def analyze_complaint_context():
    data = request.get_json()
    if not data or 'complaint_text' not in data:
        return jsonify({"status": "error", "message": "Invalid payload"}), 400

    complaint_text = data['complaint_text']
    results = model_cca.analyze(complaint_text)
    
    return jsonify({
        'data': results
    })

@app.route('/python/incident/student/remark', methods=['POST'])
def predict_remark():
    data = request.get_json()
    if not data or 'student' not in data:
        return jsonify({"status": "error", "message": "Invalid payload"}), 400

    d = data['student'][0]

    pred_tree = model_vp.predict_disciplinary_action(d)
    pred_lg = model_vp.predict_reoffense_risk(d)
    
    return jsonify({
        'data': pred_tree,
        'data2': pred_lg
    })

@app.route('/python/webpush', methods=['POST'])
def pushNotification():
    json_data = request.get_json()
    data = {
        'title': json_data.get('title'),
        'body': json_data.get('body'),
        'icon': json_data.get('icon'),
        'url': json_data.get('url')
    }
    errors = []
    sub = json_data.get('subscription')
    for s in sub:
        wp = WebPush(data, s['endpoint'], s['public_key'], s['auth'])
        errors.append(wp.push())
    
    print(errors)
    return jsonify({'status': 'success', 'message': 'Data received', 'errors': errors }) 

if __name__ == '__main__':
    app.run(port=5000, debug=True)