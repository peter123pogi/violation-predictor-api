from model_training.violation_risk_predictor import ViolationRiskPredictor
from model_training.complaint_context_analyzer import ComplaintContextAnalyzer

files = [
    'student_behavior_data',
    'violation_list',
]

model_vp = ViolationRiskPredictor(files[0])
model_cca = ComplaintContextAnalyzer()

