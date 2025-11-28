from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from webpush import WebPush
from model_training.config_model import model_vp, model_cca

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/python/incident/test-data/append', methods=['POST'])
def append_logistic_test_data():
    data = request.get_json()
    if not data or 'list' not in data:
        return jsonify({"status": "error", "message": "Invalid payload"}), 400
    
    model_vp.append_test_data(data['list'])
        
    return jsonify({
        'status': 'success',
        'message': 'Test data appended successfully'
    })

@app.route('/python/incident/risk', methods=['POST'])
def incident_risk():
    data = request.get_json()
    if not data or 'list' not in data:
        return jsonify({"status": "error", "message": "Invalid payload"}), 400
    
    df = model_vp.get_student_risk_monitoring(data['list'])
        
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

    
    d = {} if data['student'] == [] else data['student'][0]



    pred_lg = model_vp.predict_reoffense_risk(d)
    insights = model_vp.generate_insight(pred_lg)
    recommendation = model_vp.generate_recommendation(d, pred_lg.get('risk_level'))
    
    return jsonify({
        'data': {
            'pred_lg': pred_lg,
            'insights': insights,
            'recommendation': recommendation
        }
    })

@app.route('/python/webpush', methods=['POST'])
def push_notification():
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


@app.route('/python/webpush/check-subscription-expiration', methods=['POST'])
def check_expiration():
    data = request.get_json()
    if not data or "list" not in data:
        return jsonify({
            "status": "error",
            "message": "Missing 'list'"
        }), 400

    endpoints = data["list"]
    print(endpoints)
    results = []

    for item in endpoints:
        endpoint = item.get("endpoint")
        p256dh = item.get("public_key")
        auth = item.get("auth")

        if not endpoint or not p256dh or not auth:
            results.append({
                "endpoint": endpoint,
                "status": "invalid",
                "message": "Missing subscription keys"
            })
            continue

        # Instance of WebPush for this record
        wp = WebPush(
            data={"title": "", "body": "", "icon": "", "url": ""},
            endpoint=endpoint,
            public_key=p256dh,
            auth=auth
        )

        # Check expiration
        result = wp.check_expired_subscription(endpoint)
        results.append(result)
        
        print(result)
    return jsonify({
        "status": "success",
        "results": results
    })

if __name__ == '__main__':
    app.run(port=7860, debug=True)