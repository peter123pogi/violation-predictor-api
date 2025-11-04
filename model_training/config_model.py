from model_training.violation_predictor import ViolationPredictor
from model_training.complaint_context_analyzer import ComplaintContextAnalyzer

files = [
    'offense_behavior_dataset_v3_with_next_violation',
    'complaints_dataset_v1',
    'complaints_dataset_50k_expanded',
    'complaints_dataset_10k_mixed',
    'violation_keywords_dataset (1)'
]

model_vp = ViolationPredictor(files[0])
model_cca = ComplaintContextAnalyzer(files[3], files[4])
""" 
complaint = 'si earle nakita ko sya nagasmoke sa tabi ng canteen'
print(model_cca._tokenize(complaint))
results = model_cca.analyze(complaint)

print("\n🔍 Complaint Context Analysis:")
print(f"Complaint: {complaint}")
print("Predicted Violation Categories and Similarities:")
print(results)
"""


