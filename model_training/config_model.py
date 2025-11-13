from model_training.violation_predictor import ViolationPredictor
from model_training.complaint_context_analyzer import ComplaintContextAnalyzer

files = [
    'student_behavior_data',
    'complaints_dataset_10k_mixed',
    'violation_keywords_dataset (1)',
]

model_vp = ViolationPredictor(files[0])
model_cca = ComplaintContextAnalyzer(files[1], files[2])
""" 
complaint = 'si earle nakita ko sya nagasmoke sa tabi ng canteen'
print(model_cca._tokenize(complaint))
results = model_cca.analyze(complaint)

print("\n🔍 Complaint Context Analysis:")
print(f"Complaint: {complaint}")
print("Predicted Violation Categories and Similarities:")
print(results)
"""


