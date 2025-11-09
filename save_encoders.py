"""
Save drug encoders for deployment
"""
import pickle
from pathlib import Path
from pipeline import DrugSensitivityPipeline

print("Loading pipeline to extract encoders...")

pipeline = DrugSensitivityPipeline()
pipeline.load_gdsc_data()
pipeline.load_depmap_expression()
pipeline.load_model_mapping()
pipeline.merge_datasets()
pipeline.encode_drug_features()

# Save encoders
output_path = Path("saved_model/drug_encoders.pkl")
with open(output_path, 'wb') as f:
    pickle.dump(pipeline.drug_encoders, f)

print(f"\nDrug encoders saved to: {output_path}")
print(f"   - Target encoder: {len(pipeline.drug_encoders['target'].classes_)} classes")
print(f"   - Pathway encoder: {len(pipeline.drug_encoders['pathway'].classes_)} classes")
print(f"   - Drug encoder: {len(pipeline.drug_encoders['drug'].classes_)} classes")
