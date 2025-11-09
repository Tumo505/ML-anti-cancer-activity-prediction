"""
Gradio Web Interface for Drug Sensitivity Prediction
Provides user-friendly interface for making predictions with the trained pan-drug model
"""

import gradio as gr
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from pipeline import DrugSensitivityPipeline
import warnings
warnings.filterwarnings('ignore')


class DrugSensitivityApp:
    """Gradio application for drug sensitivity prediction"""
    
    def __init__(self):
        self.pipeline = None
        self.model = None
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        self.drug_list = None
        self.target_list = None
        self.pathway_list = None
        self.model_loaded = False
        
    def load_model(self):
        """Load or train the model"""
        if self.model_loaded:
            return "Model already loaded"
        
        try:
            # Try to load saved model
            model_path = Path("saved_model")
            if model_path.exists():
                with open(model_path / "model.pkl", "rb") as f:
                    self.model = pickle.load(f)
                with open(model_path / "scaler.pkl", "rb") as f:
                    self.scaler = pickle.load(f)
                with open(model_path / "imputer.pkl", "rb") as f:
                    self.imputer = pickle.load(f)
                with open(model_path / "feature_names.pkl", "rb") as f:
                    self.feature_names = pickle.load(f)
                
                self.pipeline = DrugSensitivityPipeline()
                self.pipeline.load_gdsc_data()
                self.pipeline.load_depmap_expression()
                self.pipeline.load_model_mapping()
                self.pipeline.merge_datasets()
                self.pipeline.encode_drug_features()
                
                # Load SMILES data for molecular fingerprints
                print("Loading SMILES data for molecular fingerprints...")
                self.pipeline.load_smiles_data()
                
                self.drug_list = sorted(self.pipeline.merged_data['DRUG_NAME'].unique().tolist())
                self.target_list = sorted(self.pipeline.merged_data['PUTATIVE_TARGET'].dropna().unique().tolist())
                self.pathway_list = sorted(self.pipeline.merged_data['PATHWAY_NAME'].dropna().unique().tolist())
                
                self.model_loaded = True
                return "Model loaded successfully from saved files"
            else:
                return "No saved model found. Please train the model first using pipeline.py"
                
        except Exception as e:
            return f"Error loading model: {str(e)}"
    
    def predict_drug_sensitivity(self, drug_name, target, pathway, expression_file, cell_line_id):
        """Make drug sensitivity prediction"""
        if not self.model_loaded:
            return "Please load the model first", None, None
        
        try:
            # Handle expression data input
            if expression_file is not None:
                # User uploaded a file
                expr_df = pd.read_csv(expression_file.name)
                
                # Check if it has cell line ID column
                if 'ModelID' in expr_df.columns:
                    cell_lines = expr_df['ModelID'].tolist()
                    expr_df = expr_df.set_index('ModelID')
                elif 'cell_line' in expr_df.columns:
                    cell_lines = expr_df['cell_line'].tolist()
                    expr_df = expr_df.set_index('cell_line')
                else:
                    # No ID column, use row numbers
                    cell_lines = [f"Sample_{i+1}" for i in range(len(expr_df))]
                
                # Get gene columns (columns with gene format like "BRAF (673)")
                gene_cols = [col for col in expr_df.columns if '(' in col][:1000]
                
                # If we have fewer than 1000 genes, pad with zeros
                if len(gene_cols) < 1000:
                    expression_data = np.zeros((len(expr_df), 1000))
                    expression_data[:, :len(gene_cols)] = expr_df[gene_cols].values
                else:
                    expression_data = expr_df[gene_cols].values
                
            elif cell_line_id:
                # User selected a cell line from the database
                if cell_line_id not in self.pipeline.expression_data.index:
                    return f"Cell line {cell_line_id} not found in database", None, None
                
                # Get first 1000 genes using iloc (position-based indexing)
                expression_data = self.pipeline.expression_data.loc[cell_line_id].iloc[:1000].values.reshape(1, -1)
                cell_lines = [cell_line_id]
            else:
                return "Please either upload expression data or select a cell line", None, None
            
            # Encode drug features
            try:
                target_encoded = self.pipeline.drug_encoders['target'].transform([target])[0]
            except:
                target_encoded = -1
            
            try:
                pathway_encoded = self.pipeline.drug_encoders['pathway'].transform([pathway])[0]
            except:
                pathway_encoded = -1
            
            # Try to encode drug name
            if drug_name and drug_name in self.pipeline.drug_encoders['drug'].classes_:
                drug_encoded = self.pipeline.drug_encoders['drug'].transform([drug_name])[0]
            else:
                drug_encoded = len(self.pipeline.drug_encoders['drug'].classes_) // 2
            
            # Generate molecular fingerprints for the drug
            n_samples = expression_data.shape[0]
            molecular_fp = np.zeros((n_samples, 256))  # Default: zeros for unknown drugs
            
            if drug_name and self.pipeline.smiles_data is not None:
                # Try to get SMILES and generate fingerprint
                drug_smiles_data = self.pipeline.smiles_data[
                    self.pipeline.smiles_data['DRUG_NAME'].str.lower() == drug_name.lower()
                ]
                
                if not drug_smiles_data.empty:
                    # Generate fingerprint for this drug
                    try:
                        from rdkit import Chem
                        from rdkit.Chem import rdMolDescriptors
                        
                        smiles = drug_smiles_data.iloc[0]['SMILES']
                        # Clean SMILES (remove trailing commas)
                        smiles = smiles.rstrip(',').strip()
                        
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is not None:
                            # Generate Morgan fingerprint
                            gen = rdMolDescriptors.GetMorganGenerator(radius=2, fpSize=256)
                            fp = gen.GetFingerprint(mol)
                            fp_array = np.array(fp)
                            molecular_fp = np.tile(fp_array, (n_samples, 1))
                    except Exception as e:
                        print(f"Warning: Could not generate fingerprint for {drug_name}: {e}")
            
            # Build feature matrix
            drug_features = np.array([[target_encoded, pathway_encoded, drug_encoded]])
            drug_features_repeated = np.repeat(drug_features, n_samples, axis=0)
            
            # Combine features: expression + drug metadata + molecular fingerprints
            X = np.hstack([expression_data, drug_features_repeated, molecular_fp])
            
            # Preprocess
            X_imputed = self.imputer.transform(X)
            X_scaled = self.scaler.transform(X_imputed)
            
            # Predict
            predictions = self.model.predict(X_scaled)
            
            # Get feature importance for top genes
            feature_importance = pd.DataFrame({
                'feature': self.feature_names[:1000],  # Gene names only
                'importance': self.model.feature_importances_[:1000]
            }).sort_values('importance', ascending=False).head(15)
            
            # Create results dataframe
            results_df = pd.DataFrame({
                'Cell Line': cell_lines,
                'Drug': [drug_name] * n_samples,
                'Target': [target] * n_samples,
                'Pathway': [pathway] * n_samples,
                'Predicted AUC': predictions,
                'Interpretation': ['Sensitive (AUC < 0.5)' if p < 0.5 
                                  else 'Moderate (0.5 ≤ AUC < 0.8)' if p < 0.8 
                                  else 'Resistant (AUC ≥ 0.8)' 
                                  for p in predictions]
            })
            
            # Create interpretation text
            avg_auc = predictions.mean()
            if avg_auc < 0.5:
                interpretation = f"SENSITIVE: Average predicted AUC = {avg_auc:.3f}\n\n"
                interpretation += "The cell line(s) are predicted to be SENSITIVE to this drug.\n"
                interpretation += "Lower AUC values indicate better drug response.\n"
                interpretation += "This drug is a good candidate for treatment."
            elif avg_auc < 0.8:
                interpretation = f"MODERATE: Average predicted AUC = {avg_auc:.3f}\n\n"
                interpretation += "The cell line(s) show MODERATE sensitivity to this drug.\n"
                interpretation += "Response may vary - consider combination therapy."
            else:
                interpretation = f"RESISTANT: Average predicted AUC = {avg_auc:.3f}\n\n"
                interpretation += "The cell line(s) are predicted to be RESISTANT to this drug.\n"
                interpretation += "Higher AUC values indicate drug resistance.\n"
                interpretation += "Consider alternative therapeutic options."
            
            return interpretation, results_df, feature_importance
            
        except Exception as e:
            return f"Error during prediction: {str(e)}", None, None
    
    def get_drug_info(self, drug_name):
        """Get target and pathway for a selected drug"""
        if not self.model_loaded:
            return "Unknown", "Unknown"
        
        if drug_name in self.drug_list:
            drug_data = self.pipeline.merged_data[
                self.pipeline.merged_data['DRUG_NAME'] == drug_name
            ].iloc[0]
            return drug_data['PUTATIVE_TARGET'], drug_data['PATHWAY_NAME']
        return "", ""
    
    def list_available_cell_lines(self):
        """List available cell lines"""
        if not self.model_loaded:
            return []
        return self.pipeline.expression_data.index.tolist()
    
    def export_to_csv(self, dataframe, filename):
        """Export DataFrame to CSV file"""
        if dataframe is None or len(dataframe) == 0:
            return None
        
        try:
            output_path = Path(filename)
            dataframe.to_csv(output_path, index=False)
            return str(output_path)
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return None
    
    def batch_predict(self, batch_file):
        """Run batch predictions from uploaded CSV file"""
        if batch_file is None:
            return pd.DataFrame({"Error": ["Please upload a CSV file"]}), None
        
        try:
            # Read the uploaded file
            batch_df = pd.read_csv(batch_file.name)
            print(f"\n{'='*50}")
            print(f"Batch prediction started")
            print(f"Number of predictions: {len(batch_df)}")
            print(f"Columns: {list(batch_df.columns)}")
            print(f"{'='*50}\n")
            
            # Validate required columns
            required_cols = ['Drug_Name']
            missing_cols = [col for col in required_cols if col not in batch_df.columns]
            if missing_cols:
                return pd.DataFrame({"Error": [f"Missing required columns: {missing_cols}"]}), None
            
            # Process each row
            results_list = []
            
            for idx, row in batch_df.iterrows():
                try:
                    drug_name = row['Drug_Name']
                    drug_target = row.get('Target', '')
                    drug_pathway = row.get('Pathway', '')
                    
                    # Check if cell line ID is provided or if we need to use expression data
                    if 'Cell_Line_ID' in batch_df.columns and pd.notna(row['Cell_Line_ID']):
                        cell_line_id = row['Cell_Line_ID']
                        
                        # Get expression data from database
                        if cell_line_id not in self.pipeline.expression_df.index:
                            results_list.append({
                                'Row': idx + 1,
                                'Cell_Line_ID': cell_line_id,
                                'Drug': drug_name,
                                'Target': drug_target,
                                'Pathway': drug_pathway,
                                'Predicted_AUC': None,
                                'Interpretation': f"Cell line {cell_line_id} not found in database",
                                'Status': 'Error'
                            })
                            continue
                        
                        expression_data = self.pipeline.expression_df.loc[cell_line_id].iloc[:1000].values
                    else:
                        # Extract gene expression from row (columns after metadata)
                        metadata_cols = ['Drug_Name', 'Target', 'Pathway', 'Cell_Line_ID']
                        gene_cols = [col for col in batch_df.columns if col not in metadata_cols]
                        
                        if len(gene_cols) == 0:
                            results_list.append({
                                'Row': idx + 1,
                                'Cell_Line_ID': 'N/A',
                                'Drug': drug_name,
                                'Target': drug_target,
                                'Pathway': drug_pathway,
                                'Predicted_AUC': None,
                                'Interpretation': 'No gene expression data provided',
                                'Status': 'Error'
                            })
                            continue
                        
                        expression_data = row[gene_cols].values[:1000]
                        cell_line_id = f"Sample_{idx + 1}"
                    
                    # Pad or truncate to 1000 genes
                    if len(expression_data) < 1000:
                        expression_data = np.pad(expression_data, (0, 1000 - len(expression_data)), constant_values=0)
                    else:
                        expression_data = expression_data[:1000]
                    
                    # Get drug encoding
                    drug_data = self.pipeline.merged_data[
                        self.pipeline.merged_data['DRUG_NAME'] == drug_name
                    ]
                    
                    if len(drug_data) == 0:
                        # Drug not in training set, use provided target/pathway
                        if not drug_target or not drug_pathway:
                            results_list.append({
                                'Row': idx + 1,
                                'Cell_Line_ID': cell_line_id,
                                'Drug': drug_name,
                                'Target': drug_target,
                                'Pathway': drug_pathway,
                                'Predicted_AUC': None,
                                'Interpretation': 'Drug not found and Target/Pathway not provided',
                                'Status': 'Error'
                            })
                            continue
                        
                        # Use label encoders
                        try:
                            target_encoded = self.pipeline.target_encoder.transform([drug_target])[0]
                        except:
                            target_encoded = -1
                        
                        try:
                            pathway_encoded = self.pipeline.pathway_encoder.transform([drug_pathway])[0]
                        except:
                            pathway_encoded = -1
                        
                        try:
                            drug_encoded = self.pipeline.drug_encoder.transform([drug_name])[0]
                        except:
                            drug_encoded = -1
                    else:
                        target_encoded = drug_data['target_encoded'].iloc[0]
                        pathway_encoded = drug_data['pathway_encoded'].iloc[0]
                        drug_encoded = drug_data['drug_encoded'].iloc[0]
                        drug_target = drug_data['PUTATIVE_TARGET'].iloc[0] if not drug_target else drug_target
                        drug_pathway = drug_data['PATHWAY_NAME'].iloc[0] if not drug_pathway else drug_pathway
                    
                    # Generate molecular fingerprints
                    from rdkit import Chem
                    from rdkit.Chem import rdMolDescriptors
                    
                    smiles_data = self.pipeline.smiles_df[
                        self.pipeline.smiles_df['drug_name'] == drug_name
                    ]
                    
                    if len(smiles_data) > 0 and pd.notna(smiles_data['smiles'].iloc[0]):
                        smiles = str(smiles_data['smiles'].iloc[0]).rstrip(',').strip()
                        mol = Chem.MolFromSmiles(smiles)
                        
                        if mol is not None:
                            gen = rdMolDescriptors.GetMorganGenerator(radius=2, fpSize=256)
                            fp = gen.GetFingerprint(mol)
                            fp_array = np.array(fp)
                        else:
                            fp_array = np.zeros(256)
                    else:
                        fp_array = np.zeros(256)
                    
                    # Build feature vector
                    drug_features = np.array([target_encoded, pathway_encoded, drug_encoded])
                    X = np.hstack([expression_data, drug_features, fp_array]).reshape(1, -1)
                    
                    # Make prediction
                    X_imputed = self.imputer.transform(X)
                    X_scaled = self.scaler.transform(X_imputed)
                    predicted_auc = self.model.predict(X_scaled)[0]
                    
                    # Interpret result
                    if predicted_auc < 0.5:
                        interpretation = "Sensitive (AUC < 0.5)"
                    elif predicted_auc < 0.8:
                        interpretation = "Moderate (0.5 ≤ AUC < 0.8)"
                    else:
                        interpretation = "Resistant (AUC ≥ 0.8)"
                    
                    results_list.append({
                        'Row': idx + 1,
                        'Cell_Line_ID': cell_line_id,
                        'Drug': drug_name,
                        'Target': drug_target,
                        'Pathway': drug_pathway,
                        'Predicted_AUC': round(predicted_auc, 4),
                        'Interpretation': interpretation,
                        'Status': 'Success'
                    })
                    
                    print(f"✅ Row {idx + 1}: {drug_name} on {cell_line_id} -> AUC = {predicted_auc:.4f}")
                    
                except Exception as e:
                    print(f"❌ Error processing row {idx + 1}: {str(e)}")
                    results_list.append({
                        'Row': idx + 1,
                        'Cell_Line_ID': row.get('Cell_Line_ID', 'N/A'),
                        'Drug': row.get('Drug_Name', 'N/A'),
                        'Target': row.get('Target', 'N/A'),
                        'Pathway': row.get('Pathway', 'N/A'),
                        'Predicted_AUC': None,
                        'Interpretation': f'Error: {str(e)}',
                        'Status': 'Error'
                    })
            
            # Create results DataFrame
            results_df = pd.DataFrame(results_list)
            
            # Save to temp file for download
            import tempfile
            import datetime
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_file = tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.csv',
                delete=False,
                prefix=f'batch_results_{timestamp}_'
            )
            temp_path = temp_file.name
            temp_file.close()
            
            results_df.to_csv(temp_path, index=False)
            
            print(f"\n{'='*50}")
            print(f"Batch prediction completed!")
            print(f"Total predictions: {len(results_df)}")
            print(f"Successful: {len(results_df[results_df['Status'] == 'Success'])}")
            print(f"Errors: {len(results_df[results_df['Status'] == 'Error'])}")
            print(f"Results saved to: {temp_path}")
            print(f"{'='*50}\n")
            
            return results_df, temp_path
            
        except Exception as e:
            print(f"❌ Batch prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame({"Error": [str(e)]}), None


def create_interface():
    """Create and launch Gradio interface"""
    app = DrugSensitivityApp()
    
    # Load model on startup
    load_status = app.load_model()
    print(load_status)
    
    with gr.Blocks(title="Drug Sensitivity Prediction", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # Drug Sensitivity Prediction System
        
        Predict cancer drug sensitivity using gene expression profiles and drug metadata.
        
        **Model Performance:** R² = 0.60, Pearson = 0.78 (373 drugs trained)
        """)
        
        with gr.Tab("Single Prediction"):
            gr.Markdown("""
            ### Make a prediction for a single drug-cell line combination
            
            You can either:
            - Select a drug from the database (auto-fills target and pathway)
            - Or manually enter drug information for a new drug
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Drug Information")
                    
                    drug_dropdown = gr.Dropdown(
                        choices=app.drug_list if app.model_loaded else [],
                        label="Select Drug (from trained drugs)",
                        info="Optional: Select a drug to auto-fill target and pathway"
                    )
                    
                    drug_name_input = gr.Textbox(
                        label="Or Enter Drug Name",
                        placeholder="e.g., PLX-4720, Imatinib"
                    )
                    
                    target_input = gr.Textbox(
                        label="Drug Target",
                        placeholder="e.g., BRAF, BCR-ABL, EGFR"
                    )
                    
                    pathway_input = gr.Textbox(
                        label="Drug Pathway",
                        placeholder="e.g., ERK MAPK signaling"
                    )
                
                with gr.Column():
                    gr.Markdown("#### Cell Line Information")
                    
                    with gr.Tab("Upload Expression File"):
                        expression_file = gr.File(
                            label="Upload Gene Expression CSV",
                            file_types=[".csv"]
                        )
                        gr.Markdown("""
                        **CSV Format:** 
                        - First column: Cell line ID (optional)
                        - Other columns: Gene expression values
                        - Column names: GENE (EntrezID) format
                        - Example: BRAF (673), TP53 (7157)
                        """)
                    
                    with gr.Tab("Select from Database"):
                        cell_line_dropdown = gr.Dropdown(
                            choices=app.list_available_cell_lines() if app.model_loaded else [],
                            label="Select Cell Line",
                            info="Choose from 1,699 available cell lines"
                        )
            
            predict_btn = gr.Button("Predict Drug Sensitivity", variant="primary")
            
            with gr.Row():
                with gr.Column():
                    interpretation_output = gr.Textbox(
                        label="Prediction Interpretation",
                        lines=8
                    )
                
                with gr.Column():
                    results_output = gr.Dataframe(
                        label="Detailed Results",
                        headers=["Cell Line", "Drug", "Target", "Pathway", "Predicted AUC", "Interpretation"]
                    )
            
            # Export buttons for results
            with gr.Row():
                export_results_btn = gr.Button("📥 Download Prediction Results (CSV)", size="sm")
            results_download = gr.File(label="Download Results CSV", visible=True)
            
            biomarkers_output = gr.Dataframe(
                label="Top 15 Predictive Genes (Biomarkers)",
                headers=["Gene", "Importance"]
            )
            
            # Export button for biomarkers
            with gr.Row():
                export_biomarkers_btn = gr.Button("📥 Download Biomarkers (CSV)", size="sm")
            biomarkers_download = gr.File(label="Download Biomarkers CSV", visible=True)
            
            # Auto-fill target and pathway when drug is selected
            drug_dropdown.change(
                fn=app.get_drug_info,
                inputs=[drug_dropdown],
                outputs=[target_input, pathway_input]
            )
            
            # Helper function to get the correct drug name
            def get_drug_name():
                return drug_name_input.value or drug_dropdown.value
            
            # Prediction
            predict_btn.click(
                fn=lambda name, tgt, path, file, cell: app.predict_drug_sensitivity(
                    name or "", tgt, path, file, cell
                ),
                inputs=[
                    drug_name_input,
                    target_input,
                    pathway_input,
                    expression_file,
                    cell_line_dropdown
                ],
                outputs=[interpretation_output, results_output, biomarkers_output]
            )
            
            # Export handlers
            def export_results(results_df):
                """Export prediction results to CSV"""
                import tempfile
                import datetime
                import pandas as pd
                import os
                
                # Check if we have valid data
                if results_df is None:
                    print("Export failed: No results data")
                    return None
                    
                if not isinstance(results_df, pd.DataFrame):
                    print(f"Export failed: Expected DataFrame, got {type(results_df)}")
                    return None
                    
                if len(results_df) == 0:
                    print("Export failed: Empty results DataFrame")
                    return None
                
                try:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    # Create temp file and close it immediately so we can write to it
                    temp_file = tempfile.NamedTemporaryFile(
                        mode='w', 
                        suffix='.csv', 
                        delete=False,
                        prefix=f'prediction_results_{timestamp}_'
                    )
                    temp_path = temp_file.name
                    temp_file.close()
                    
                    # Now write the CSV
                    results_df.to_csv(temp_path, index=False)
                    print(f"✅ Results exported to: {temp_path}")
                    print(f"   File exists: {os.path.exists(temp_path)}")
                    print(f"   File size: {os.path.getsize(temp_path)} bytes")
                    return temp_path
                except Exception as e:
                    print(f"❌ Export error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return None
            
            def export_biomarkers(biomarkers_df):
                """Export biomarkers to CSV"""
                import tempfile
                import datetime
                import pandas as pd
                import os
                
                # Check if we have valid data
                if biomarkers_df is None:
                    print("Export failed: No biomarkers data")
                    return None
                    
                if not isinstance(biomarkers_df, pd.DataFrame):
                    print(f"Export failed: Expected DataFrame, got {type(biomarkers_df)}")
                    return None
                    
                if len(biomarkers_df) == 0:
                    print("Export failed: Empty biomarkers DataFrame")
                    return None
                
                try:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    # Create temp file and close it immediately so we can write to it
                    temp_file = tempfile.NamedTemporaryFile(
                        mode='w', 
                        suffix='.csv', 
                        delete=False,
                        prefix=f'biomarkers_{timestamp}_'
                    )
                    temp_path = temp_file.name
                    temp_file.close()
                    
                    # Now write the CSV
                    biomarkers_df.to_csv(temp_path, index=False)
                    print(f"✅ Biomarkers exported to: {temp_path}")
                    print(f"   File exists: {os.path.exists(temp_path)}")
                    print(f"   File size: {os.path.getsize(temp_path)} bytes")
                    return temp_path
                except Exception as e:
                    print(f"❌ Export error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return None
            
            export_results_btn.click(
                fn=export_results,
                inputs=[results_output],
                outputs=[results_download]
            )
            
            export_biomarkers_btn.click(
                fn=export_biomarkers,
                inputs=[biomarkers_output],
                outputs=[biomarkers_download]
            )
        
        with gr.Tab("Batch Prediction"):
            gr.Markdown("""
            ### Upload a CSV file with multiple cell lines and drugs
            
            **CSV Format:**
            - Columns: Cell_Line_ID, Drug_Name, Target, Pathway, [Gene expression columns...]
            - Each row represents one prediction to make
            """)
            
            batch_file = gr.File(
                label="Upload Batch CSV",
                file_types=[".csv"]
            )
            
            batch_predict_btn = gr.Button("Run Batch Prediction", variant="primary")
            
            batch_results = gr.Dataframe(
                label="Batch Prediction Results"
            )
            
            batch_download = gr.File(
                label="Download Results"
            )
            
            # Batch prediction event handler
            batch_predict_btn.click(
                fn=app.batch_predict,
                inputs=[batch_file],
                outputs=[batch_results, batch_download]
            )
        
        with gr.Tab("Model Information"):
            gr.Markdown(f"""
            ### Model Details
            
            **Architecture:** XGBoost Regressor (GPU-accelerated)
            
            **Training Data:**
            - 373 unique drugs
            - 244,828 drug-cell line experiments
            - 714 cancer cell lines
            - 1,000 gene expression features
            
            **Performance Metrics:**
            - Test R²: 0.5987
            - Test Pearson Correlation: 0.7799
            - Test RMSE: 0.1214
            - Cross-Validation R²: 0.5941 ± 0.0057
            
            **Input Features:**
            - Top 1,000 most variable genes (expression)
            - Drug target protein (encoded)
            - Drug biological pathway (encoded)
            - Drug identity (encoded)
            
            **Output:**
            - Predicted AUC (Area Under Curve) from 0 to 1
            - Lower AUC = More sensitive to drug
            - Higher AUC = More resistant to drug
            
            **Capabilities:**
            - Predict sensitivity for 373 trained drugs
            - Generalize to new drugs (with target + pathway)
            - Identify predictive biomarkers
            - Batch predictions for multiple samples
            
            **Best Predicted Drugs:**
            - Imatinib (BCR-ABL): R² = 0.73
            - ACY-1215 (HDAC): R² = 0.53
            - Ponatinib (BCR-ABL/VEGFR): R² = 0.43
            
            **Model Status:** {load_status}
            """)
        
        with gr.Tab("Help & Examples"):
            gr.Markdown("""
            ### How to Use
            
            #### Option 1: Select from Database
            1. Go to "Single Prediction" tab
            2. Select a drug from the dropdown (e.g., "Imatinib")
            3. Target and pathway will auto-fill
            4. Select a cell line from database (e.g., "ACH-000001")
            5. Click "Predict Drug Sensitivity"
            
            #### Option 2: Upload Expression Data
            1. Prepare a CSV file with gene expression:
               ```
               ModelID,BRAF (673),TP53 (7157),KRAS (3845),...
               MY_CELL_LINE_1,5.2,3.1,4.8,...
               MY_CELL_LINE_2,6.1,2.9,5.3,...
               ```
            2. Enter drug information (name, target, pathway)
            3. Upload the CSV file
            4. Click "Predict Drug Sensitivity"
            
            #### Option 3: Predict New Drug
            1. Enter new drug name (e.g., "My-BRAF-Inhibitor")
            2. Enter target (e.g., "BRAF")
            3. Enter pathway (e.g., "ERK MAPK signaling")
            4. Provide cell line data
            5. Model will generalize based on target/pathway similarity
            
            ### Interpreting Results
            
            **AUC Values:**
            - 0.0 - 0.5: SENSITIVE (good drug candidate)
            - 0.5 - 0.8: MODERATE (may respond with combination)
            - 0.8 - 1.0: RESISTANT (consider alternatives)
            
            **Biomarkers:**
            - Top genes driving the prediction
            - Higher importance = stronger influence
            - Can be used for patient stratification
            
            ### Example Drugs
            
            **Targeted Therapies:**
            - Imatinib (BCR-ABL inhibitor) - CML, GIST
            - PLX-4720 (BRAF inhibitor) - Melanoma
            - Afatinib (EGFR/ERBB2 inhibitor) - Lung cancer
            - Olaparib (PARP inhibitor) - Ovarian cancer
            
            **Chemotherapy:**
            - Cisplatin (DNA crosslinker)
            - Doxorubicin (Anthracycline)
            - Gemcitabine (Antimetabolite)
            - 5-Fluorouracil (Thymidylate synthase)
            """)
    
    return demo


if __name__ == "__main__":
    print("Starting Drug Sensitivity Prediction App...")
    print("="*50)
    
    try:
        demo = create_interface()
        print("\nLaunching Gradio interface...")
        print(f"Server will be available at: http://localhost:7860")
        print("="*50)
        
        demo.launch(
            share=False,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
            quiet=False,
            debug=True
        )
    except Exception as e:
        print(f"\n ERROR: Failed to start application")
        print(f"Error details: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Make sure Gradio is installed: pip install gradio")
        print("2. Check if port 7860 is available")
        print("3. Ensure the model is trained: python pipeline.py")
        print("4. Check if saved_model/ directory exists")
