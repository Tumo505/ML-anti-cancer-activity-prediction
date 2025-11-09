"""
Extract minimal metadata for deployment (drug and cell line lists)
This creates a small JSON file instead of deploying the full 800MB dataset
"""

import pandas as pd
import json
from pathlib import Path
from pipeline import DrugSensitivityPipeline

print("Extracting metadata for deployment...")

# Load the pipeline
pipeline = DrugSensitivityPipeline()
pipeline.load_gdsc_data()
pipeline.load_depmap_expression()
pipeline.load_model_mapping()
pipeline.merge_datasets()
pipeline.encode_drug_features()
pipeline.load_smiles_data()

# Extract lists
metadata = {
    'drugs': sorted(pipeline.merged_data['DRUG_NAME'].unique().tolist()),
    'targets': sorted(pipeline.merged_data['PUTATIVE_TARGET'].dropna().unique().tolist()),
    'pathways': sorted(pipeline.merged_data['PATHWAY_NAME'].dropna().unique().tolist()),
    'cell_lines': sorted(pipeline.expression_data.index.tolist()),
    'smiles': {},
    'drug_info': {}  # Maps drug name -> {target, pathway, smiles}
}

# Create drug information mapping (for auto-fill functionality)
drug_info_df = pipeline.merged_data.groupby('DRUG_NAME').agg({
    'PUTATIVE_TARGET': 'first',
    'PATHWAY_NAME': 'first'
}).reset_index()

for _, row in drug_info_df.iterrows():
    drug_name = row['DRUG_NAME']
    metadata['drug_info'][drug_name] = {
        'target': row['PUTATIVE_TARGET'] if pd.notna(row['PUTATIVE_TARGET']) else '',
        'pathway': row['PATHWAY_NAME'] if pd.notna(row['PATHWAY_NAME']) else '',
        'smiles': ''
    }

# Add SMILES data to drug_info mapping
if pipeline.smiles_data is not None:
    metadata['smiles'] = dict(zip(
        pipeline.smiles_data['DRUG_NAME'].tolist(),
        pipeline.smiles_data['SMILES'].tolist()
    ))
    
    # Merge SMILES into drug_info
    for drug_name, smiles in metadata['smiles'].items():
        if drug_name in metadata['drug_info']:
            metadata['drug_info'][drug_name]['smiles'] = smiles

# Save to JSON
output_path = Path("deployment_metadata.json")
with open(output_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\nMetadata extracted successfully!")
print(f"Saved to: {output_path}")
print(f"Stats:")
print(f"   - Drugs: {len(metadata['drugs'])}")
print(f"   - Targets: {len(metadata['targets'])}")
print(f"   - Pathways: {len(metadata['pathways'])}")
print(f"   - Cell lines: {len(metadata['cell_lines'])}")
print(f"   - SMILES structures: {len(metadata['smiles'])}")
print(f"   - File size: {output_path.stat().st_size / 1024:.1f} KB")
