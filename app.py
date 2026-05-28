"""
Gradio Web Interface for Drug Sensitivity Prediction
Provides user-friendly interface for making predictions with the trained pan-drug model
"""

import gradio as gr
import pandas as pd
import numpy as np
import pickle
import json
import io
import base64
from pathlib import Path
from pipeline import DrugSensitivityPipeline
import warnings
warnings.filterwarnings('ignore')

# SHAP for model explainability
try:
    import shap
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server
    import matplotlib.pyplot as plt
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Explainability features will be disabled.")


class DrugSensitivityApp:
    """Gradio application for drug sensitivity prediction"""
    
    def __init__(self):
        self.pipeline = None
        self.model = None
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        self.drug_encoders = None  # Separate encoders for deployment mode
        self.drug_list = None
        self.target_list = None
        self.pathway_list = None
        self.cell_line_list = None
        self.cell_line_name_to_id = {}  # CellLineName -> ModelID mapping
        self.cell_line_id_to_name = {}  # ModelID -> CellLineName mapping
        self.drug_info = {}  # drug -> {target, pathway} mapping
        self.smiles_data = {}  # drug -> SMILES mapping
        self.model_loaded = False
        self.shap_explainer = None  # SHAP TreeExplainer for XGBoost

    def _load_deployment_metadata(self, model_path):
        """Load compact metadata for hosted deployments without raw data files."""
        metadata_file = model_path / "deployment_metadata.pkl"
        if not metadata_file.exists():
            return False

        print("Loading compact deployment metadata...")
        with open(metadata_file, "rb") as f:
            metadata = pickle.load(f)

        self.drug_list = metadata.get("drug_list", [])
        self.target_list = metadata.get("target_list", [])
        self.pathway_list = metadata.get("pathway_list", [])
        self.cell_line_name_to_id = metadata.get("cell_line_name_to_id", {})
        self.cell_line_id_to_name = metadata.get("cell_line_id_to_name", {})
        self.cell_line_list = metadata.get("cell_line_list", sorted(self.cell_line_name_to_id.keys()))
        self.drug_info = metadata.get("drug_info", {})
        self.smiles_data = metadata.get("smiles_data", {})

        # The prediction path expects a pipeline object with expression_data.
        self.pipeline = DrugSensitivityPipeline()
        self.pipeline.expression_data = metadata.get("expression_data")
        self.pipeline.model_mapping = metadata.get("model_mapping")

        if self.pipeline.expression_data is None:
            raise ValueError("deployment_metadata.pkl is missing expression_data")

        if self.smiles_data:
            self.pipeline.smiles_data = pd.DataFrame(
                [{"DRUG_NAME": drug, "SMILES": smiles} for drug, smiles in self.smiles_data.items()]
            )

        print(f"Loaded deployment metadata: {len(self.drug_list)} drugs, {len(self.cell_line_list)} cell lines")
        return True

    def _initialize_shap_explainer(self):
        """Initialize SHAP explainer if the dependency and model support it."""
        if not SHAP_AVAILABLE:
            return

        try:
            print("Initializing SHAP TreeExplainer...")
            self.shap_explainer = shap.TreeExplainer(
                self.model,
                model_output="raw",
                feature_perturbation="tree_path_dependent"
            )
            print("SHAP TreeExplainer initialized successfully")
        except Exception as e:
            print(f"Warning: TreeExplainer failed ({e})")
            try:
                print("Trying SHAP with model predict function...")
                self.shap_explainer = "on_demand"
                print("SHAP will compute explanations on-demand")
            except Exception as e2:
                print(f"Warning: Could not initialize SHAP explainer: {e2}")
                self.shap_explainer = None
        
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
                with open(model_path / "drug_encoders.pkl", "rb") as f:
                    self.drug_encoders = pickle.load(f)

                if self._load_deployment_metadata(model_path):
                    self._initialize_shap_explainer()
                    self.model_loaded = True
                    return "Model loaded successfully with compact deployment metadata"
                
                # Load full dataset with all data
                print("Loading full dataset...")
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
                
                # Build cell line name mappings (ModelID <-> CellLineName)
                model_mapping = self.pipeline.model_mapping
                for _, row in model_mapping.iterrows():
                    model_id = row['ModelID']
                    cell_name = row['CellLineName']
                    if model_id in self.pipeline.expression_data.index:
                        self.cell_line_name_to_id[cell_name] = model_id
                        self.cell_line_id_to_name[model_id] = cell_name
                
                # Use cell line names for the dropdown list
                self.cell_line_list = sorted(self.cell_line_name_to_id.keys())
                
                # Build drug info mapping
                self.drug_info = {}
                for drug in self.drug_list:
                    drug_data = self.pipeline.merged_data[self.pipeline.merged_data['DRUG_NAME'] == drug].iloc[0]
                    self.drug_info[drug] = {
                        'target': str(drug_data['PUTATIVE_TARGET']) if pd.notna(drug_data['PUTATIVE_TARGET']) else '',
                        'pathway': str(drug_data['PATHWAY_NAME']) if pd.notna(drug_data['PATHWAY_NAME']) else ''
                    }
                
                # Build SMILES mapping
                if self.pipeline.smiles_data is not None:
                    self.smiles_data = dict(zip(
                        self.pipeline.smiles_data['DRUG_NAME'].tolist(),
                        self.pipeline.smiles_data['SMILES'].tolist()
                    ))
                
                print(f"Loaded full dataset: {len(self.drug_list)} drugs, {len(self.cell_line_list)} cell lines")
                
                self._initialize_shap_explainer()
                
                self.model_loaded = True
                return "Model loaded successfully with full database"
            else:
                return "No saved model found. Please train the model first using pipeline.py"
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error loading model: {str(e)}"
    
    def predict_drug_sensitivity(self, drug_name, target, pathway, expression_file, cell_line_id):
        """Make drug sensitivity prediction"""
        if not self.model_loaded:
            return "Please load the model first", None, None, None
        
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
                # User selected a cell line from the database (by name)
                # Convert cell line name to ModelID for lookup
                model_id = self.cell_line_name_to_id.get(cell_line_id, cell_line_id)
                
                if model_id not in self.pipeline.expression_data.index:
                    return f"Cell line {cell_line_id} not found in database", None, None, None
                
                # Get first 1000 genes using iloc (position-based indexing)
                expression_data = self.pipeline.expression_data.loc[model_id].iloc[:1000].values.reshape(1, -1)
                # Use the actual cell line name for display
                display_name = self.cell_line_id_to_name.get(model_id, cell_line_id)
                cell_lines = [display_name]
            else:
                return "Please either upload expression data or select a cell line", None, None, None
            
            # Get drug encoders (deployment mode uses self.drug_encoders, dev mode uses self.pipeline.drug_encoders)
            encoders = self.drug_encoders if self.drug_encoders else (self.pipeline.drug_encoders if self.pipeline else None)
            
            if not encoders:
                return "Error: Drug encoders not loaded", None, None, None
            
            # Encode drug features
            try:
                target_encoded = encoders['target'].transform([target if target else 'Unknown'])[0]
            except:
                target_encoded = -1
            
            try:
                pathway_encoded = encoders['pathway'].transform([pathway if pathway else 'Unknown'])[0]
            except:
                pathway_encoded = -1
            
            # Try to encode drug name
            if drug_name and drug_name in encoders['drug'].classes_:
                drug_encoded = encoders['drug'].transform([drug_name])[0]
            else:
                drug_encoded = len(encoders['drug'].classes_) // 2
            
            # Generate molecular fingerprints for the drug
            n_samples = expression_data.shape[0]
            molecular_fp = np.zeros((n_samples, 256))  # Default: zeros for unknown drugs
            
            # Try to get SMILES from metadata dict first (deployment mode)
            smiles = None
            if drug_name:
                # Case-insensitive lookup in metadata
                smiles = self.smiles_data.get(drug_name)
                if not smiles:
                    # Try case-insensitive search
                    for key, value in self.smiles_data.items():
                        if key.lower() == drug_name.lower():
                            smiles = value
                            break
                
                # Fallback to pipeline (development mode)
                if not smiles and self.pipeline and hasattr(self.pipeline, 'smiles_data') and self.pipeline.smiles_data is not None:
                    drug_smiles_data = self.pipeline.smiles_data[
                        self.pipeline.smiles_data['DRUG_NAME'].str.lower() == drug_name.lower()
                    ]
                    if not drug_smiles_data.empty:
                        smiles = drug_smiles_data.iloc[0]['SMILES']
                
                # Generate fingerprint if we found SMILES
                if smiles:
                    try:
                        from rdkit import Chem
                        from rdkit.Chem import rdMolDescriptors
                        
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
            
            return interpretation, results_df, feature_importance, X_scaled
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error during prediction: {str(e)}", None, None, None
    
    def generate_shap_explanation(self, X_scaled, drug_name, cell_line_name, top_n=20):
        """
        Generate SHAP-based explanation for a prediction.
        
        Returns:
            - shap_plot: Base64 encoded waterfall plot image
            - shap_df: DataFrame with top SHAP values
            - explanation_text: Human-readable interpretation
        """
        if not SHAP_AVAILABLE:
            return None, None, "SHAP explainability not available. Install shap package."
        
        if self.shap_explainer is None:
            return None, None, "SHAP explainer could not be initialized for this model."
        
        try:
            # Handle on-demand SHAP computation
            if self.shap_explainer == "on_demand":
                # Use TreeExplainer directly on-demand
                try:
                    explainer = shap.TreeExplainer(self.model)
                    shap_values = explainer.shap_values(X_scaled)
                    base_value = explainer.expected_value
                except:
                    # Fall back to XGBoost's native additive tree contributions.
                    return self._generate_native_xgboost_shap_explanation(X_scaled, drug_name, cell_line_name, top_n)
            else:
                # Use pre-initialized explainer
                shap_result = self.shap_explainer(X_scaled)
                
                # Extract SHAP values based on result type
                if hasattr(shap_result, 'values'):
                    shap_values = shap_result.values
                    base_value = shap_result.base_values[0] if hasattr(shap_result, 'base_values') else 0.73
                else:
                    shap_values = shap_result
                    base_value = self.shap_explainer.expected_value if hasattr(self.shap_explainer, 'expected_value') else 0.73
            
            # For single sample, get first row
            if len(shap_values.shape) == 1:
                sample_shap = shap_values
            else:
                sample_shap = shap_values[0]
            
            if isinstance(base_value, np.ndarray):
                base_value = float(base_value[0])
            else:
                base_value = float(base_value)
            
            # Create feature names with readable labels
            feature_labels = []
            for i, name in enumerate(self.feature_names):
                if i < 1000:
                    # Gene features - extract gene name
                    if '(' in name:
                        gene_name = name.split('(')[0].strip()
                        feature_labels.append(f"Gene: {gene_name}")
                    else:
                        feature_labels.append(f"Gene: {name}")
                elif name == 'target_encoded':
                    feature_labels.append("Drug Target")
                elif name == 'pathway_encoded':
                    feature_labels.append("Drug Pathway")
                elif name == 'drug_encoded':
                    feature_labels.append("Drug Identity")
                elif name.startswith('fp_'):
                    feature_labels.append(f"MolFP_{name.split('_')[1]}")
                else:
                    feature_labels.append(name)
            
            # Create DataFrame with SHAP values
            shap_df = pd.DataFrame({
                'Feature': feature_labels,
                'SHAP Value': sample_shap,
                'Abs SHAP': np.abs(sample_shap),
                'Direction': ['↑ Increases AUC (Resistance)' if v > 0 else '↓ Decreases AUC (Sensitivity)' for v in sample_shap]
            })
            
            # Filter out Drug Identity, Drug Pathway, and Drug Target features
            exclude_features = ['Drug Identity', 'Drug Pathway', 'Drug Target']
            shap_df = shap_df[~shap_df['Feature'].isin(exclude_features)]
            shap_df = shap_df.sort_values('Abs SHAP', ascending=False)
            
            # Get top N features
            top_shap = shap_df.head(top_n).copy()
            top_shap['SHAP Value'] = top_shap['SHAP Value'].round(4)
            top_shap = top_shap[['Feature', 'SHAP Value', 'Direction']]
            
            # Generate waterfall plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Sort by absolute value for plotting
            plot_data = shap_df.head(top_n).sort_values('SHAP Value')
            colors = ['#ff6b6b' if v > 0 else '#4ecdc4' for v in plot_data['SHAP Value']]
            
            y_pos = np.arange(len(plot_data))
            ax.barh(y_pos, plot_data['SHAP Value'], color=colors, edgecolor='white', linewidth=0.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(plot_data['Feature'], fontsize=9)
            ax.set_xlabel('SHAP Value (Impact on Predicted AUC)', fontsize=11)
            ax.set_title(f'Feature Contributions for {drug_name} on {cell_line_name}', fontsize=12, fontweight='bold')
            ax.axvline(x=0, color='black', linewidth=0.8)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#ff6b6b', label='Increases AUC (→ Resistance)'),
                Patch(facecolor='#4ecdc4', label='Decreases AUC (→ Sensitivity)')
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
            
            plt.tight_layout()
            
            # Convert plot to base64 image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            buf.seek(0)
            plt.close(fig)
            
            # Save to temp file for Gradio
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            temp_file.write(buf.getvalue())
            temp_file.close()
            
            # Calculate predicted AUC from SHAP
            predicted_auc = float(base_value) + float(sample_shap.sum())
            
            # Identify key drivers
            top_positive = shap_df[shap_df['SHAP Value'] > 0].head(3)
            top_negative = shap_df[shap_df['SHAP Value'] < 0].head(3)
            
            explanation_text = f"""## SHAP Explanation for {drug_name} on {cell_line_name}

**Base prediction (average):** {base_value:.3f}
**Final prediction:** {predicted_auc:.3f}
**Total SHAP adjustment:** {sample_shap.sum():+.3f}

### Top Resistance Drivers (↑ AUC):
"""
            for _, row in top_positive.iterrows():
                explanation_text += f"- **{row['Feature']}**: {row['SHAP Value']:+.4f}\n"
            
            explanation_text += "\n### Top Sensitivity Drivers (↓ AUC):\n"
            for _, row in top_negative.iterrows():
                explanation_text += f"- **{row['Feature']}**: {row['SHAP Value']:+.4f}\n"
            
            explanation_text += f"\n### Interpretation:\n"
            if predicted_auc < 0.5:
                explanation_text += "The model predicts **SENSITIVITY** primarily driven by the features listed above that decrease AUC."
            elif predicted_auc < 0.8:
                explanation_text += "The model predicts **MODERATE** response with competing factors pushing toward both sensitivity and resistance."
            else:
                explanation_text += "The model predicts **RESISTANCE** primarily driven by the features listed above that increase AUC."
            
            return temp_file.name, top_shap, explanation_text
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, None, f"Error generating SHAP explanation: {str(e)}"
    
    def _generate_native_xgboost_shap_explanation(self, X_scaled, drug_name, cell_line_name, top_n=20):
        """
        Generate native XGBoost TreeSHAP-style contributions.
        XGBoost's pred_contribs output is additive: contributions + bias = prediction.
        """
        try:
            import xgboost as xgb

            contrib = self.model.get_booster().predict(xgb.DMatrix(X_scaled), pred_contribs=True)
            sample_contrib = contrib[0, :-1]
            base_value = float(contrib[0, -1])
            prediction = float(base_value + sample_contrib.sum())

            feature_labels = []
            for i, name in enumerate(self.feature_names):
                if i < 1000:
                    gene_name = name.split('(')[0].strip() if '(' in name else name
                    feature_labels.append(f"Gene: {gene_name}")
                elif name == 'target_encoded':
                    feature_labels.append("Drug Target")
                elif name == 'pathway_encoded':
                    feature_labels.append("Drug Pathway")
                elif name == 'drug_encoded':
                    feature_labels.append("Drug Identity")
                elif name.startswith('fp_'):
                    feature_labels.append(f"MolFP_{name.split('_')[1]}")
                else:
                    feature_labels.append(name)

            shap_df = pd.DataFrame({
                'Feature': feature_labels,
                'SHAP Value': sample_contrib,
                'Abs SHAP': np.abs(sample_contrib),
                'Direction': [
                    'Increases AUC (Resistance)' if value > 0 else 'Decreases AUC (Sensitivity)'
                    for value in sample_contrib
                ],
            })

            shap_df = shap_df[~shap_df['Feature'].isin(['Drug Identity', 'Drug Pathway', 'Drug Target'])]
            shap_df = shap_df.sort_values('Abs SHAP', ascending=False)

            top_shap = shap_df.head(top_n).copy()
            top_shap_display = top_shap[['Feature', 'SHAP Value', 'Direction']].copy()
            top_shap_display['SHAP Value'] = top_shap_display['SHAP Value'].round(4)

            fig, ax = plt.subplots(figsize=(10, 8))
            plot_data = shap_df.head(top_n).sort_values('SHAP Value')
            colors = ['#ff6b6b' if value > 0 else '#4ecdc4' for value in plot_data['SHAP Value']]

            y_pos = np.arange(len(plot_data))
            ax.barh(y_pos, plot_data['SHAP Value'], color=colors, edgecolor='white', linewidth=0.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(plot_data['Feature'], fontsize=9)
            ax.set_xlabel('Native XGBoost SHAP Contribution', fontsize=11)
            ax.set_title(f'Feature Contributions for {drug_name} on {cell_line_name}', fontsize=12, fontweight='bold')
            ax.axvline(x=0, color='black', linewidth=0.8)

            from matplotlib.patches import Patch
            ax.legend(
                handles=[
                    Patch(facecolor='#ff6b6b', label='Resistance'),
                    Patch(facecolor='#4ecdc4', label='Sensitivity'),
                ],
                loc='lower right',
                fontsize=9,
            )
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            buf.seek(0)
            plt.close(fig)

            import tempfile
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            temp_file.write(buf.getvalue())
            temp_file.close()

            top_positive = shap_df[shap_df['SHAP Value'] > 0].head(3)
            top_negative = shap_df[shap_df['SHAP Value'] < 0].head(3)

            explanation_text = f"""## Native XGBoost SHAP Explanation for {drug_name} on {cell_line_name}

*Using XGBoost's native `pred_contribs=True` tree contribution values.*

**Base value:** {base_value:.3f}
**Predicted AUC:** {prediction:.3f}

### Top Features Contributing to Resistance:
"""
            for _, row in top_positive.iterrows():
                explanation_text += f"- **{row['Feature']}**: SHAP={row['SHAP Value']:+.4f}\n"

            explanation_text += "\n### Top Features Contributing to Sensitivity:\n"
            for _, row in top_negative.iterrows():
                explanation_text += f"- **{row['Feature']}**: SHAP={row['SHAP Value']:+.4f}\n"

            explanation_text += "\n### Interpretation:\n"
            if prediction < 0.5:
                explanation_text += "The model predicts **SENSITIVITY** - key features push the prediction toward lower AUC."
            elif prediction < 0.8:
                explanation_text += "The model predicts **MODERATE** response - features show mixed signals."
            else:
                explanation_text += "The model predicts **RESISTANCE** - key features push the prediction toward higher AUC."

            return temp_file.name, top_shap_display, explanation_text

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, None, f"Error generating native XGBoost SHAP explanation: {str(e)}"

    def _generate_feature_importance_explanation(self, X_scaled, drug_name, cell_line_name, top_n=20):
        """
        Generate a feature importance-based explanation as a fallback when SHAP fails.
        Uses the model's built-in feature importance combined with input values.
        """
        try:
            # Get XGBoost feature importance
            importance = self.model.feature_importances_
            
            # Calculate pseudo-SHAP: importance * (feature_value - mean)
            # For scaled data, mean is roughly 0, so we use importance * value
            feature_values = X_scaled.flatten() if hasattr(X_scaled, 'flatten') else X_scaled[0]
            pseudo_shap = importance * feature_values
            
            # Create feature labels
            feature_labels = []
            for i, name in enumerate(self.feature_names):
                if i < 1000:
                    if '(' in name:
                        gene_name = name.split('(')[0].strip()
                        feature_labels.append(f"Gene: {gene_name}")
                    else:
                        feature_labels.append(f"Gene: {name}")
                elif name == 'target_encoded':
                    feature_labels.append("Drug Target")
                elif name == 'pathway_encoded':
                    feature_labels.append("Drug Pathway")
                elif name == 'drug_encoded':
                    feature_labels.append("Drug Identity")
                elif name.startswith('fp_'):
                    feature_labels.append(f"MolFP_{name.split('_')[1]}")
                else:
                    feature_labels.append(name)
            
            # Create DataFrame
            fi_df = pd.DataFrame({
                'Feature': feature_labels,
                'Importance': importance,
                'Feature Value': feature_values,
                'Contribution': pseudo_shap,
                'Abs Contribution': np.abs(pseudo_shap),
                'Direction': ['↑ Contributes to Resistance' if v > 0 else '↓ Contributes to Sensitivity' for v in pseudo_shap]
            })
            
            # Filter out Drug Identity, Drug Pathway, and Drug Target features
            exclude_features = ['Drug Identity', 'Drug Pathway', 'Drug Target']
            fi_df = fi_df[~fi_df['Feature'].isin(exclude_features)]
            fi_df = fi_df.sort_values('Abs Contribution', ascending=False)
            
            # Get top N features
            top_fi = fi_df.head(top_n).copy()
            top_fi_display = top_fi[['Feature', 'Contribution', 'Direction']].copy()
            top_fi_display.columns = ['Feature', 'SHAP Value', 'Direction']  # Match SHAP format
            top_fi_display['SHAP Value'] = top_fi_display['SHAP Value'].round(4)
            
            # Generate bar plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            plot_data = fi_df.head(top_n).sort_values('Contribution')
            colors = ['#ff6b6b' if v > 0 else '#4ecdc4' for v in plot_data['Contribution']]
            
            y_pos = np.arange(len(plot_data))
            ax.barh(y_pos, plot_data['Contribution'], color=colors, edgecolor='white', linewidth=0.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(plot_data['Feature'], fontsize=9)
            ax.set_xlabel('Feature Contribution (Importance × Value)', fontsize=11)
            ax.set_title(f'Feature Contributions for {drug_name} on {cell_line_name}\n(Based on Feature Importance)', fontsize=12, fontweight='bold')
            ax.axvline(x=0, color='black', linewidth=0.8)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#ff6b6b', label='→ Resistance'),
                Patch(facecolor='#4ecdc4', label='→ Sensitivity')
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
            
            plt.tight_layout()
            
            # Save plot
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            buf.seek(0)
            plt.close(fig)
            
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            temp_file.write(buf.getvalue())
            temp_file.close()
            
            # Get model prediction for reference
            prediction = float(self.model.predict(X_scaled)[0])
            
            # Top contributors
            top_positive = fi_df[fi_df['Contribution'] > 0].head(3)
            top_negative = fi_df[fi_df['Contribution'] < 0].head(3)
            
            explanation_text = f"""## Feature Importance Analysis for {drug_name} on {cell_line_name}

⚠️ *Note: Using feature importance analysis (SHAP TreeExplainer not available for this model)*

**Predicted AUC:** {prediction:.3f}

### Top Features Contributing to Resistance:
"""
            for _, row in top_positive.iterrows():
                explanation_text += f"- **{row['Feature']}**: Importance={row['Importance']:.4f}, Value={row['Feature Value']:.3f}\n"
            
            explanation_text += "\n### Top Features Contributing to Sensitivity:\n"
            for _, row in top_negative.iterrows():
                explanation_text += f"- **{row['Feature']}**: Importance={row['Importance']:.4f}, Value={row['Feature Value']:.3f}\n"
            
            explanation_text += f"\n### Interpretation:\n"
            if prediction < 0.5:
                explanation_text += "The model predicts **SENSITIVITY** - key features have values that push the prediction toward lower AUC."
            elif prediction < 0.8:
                explanation_text += "The model predicts **MODERATE** response - features show mixed signals."
            else:
                explanation_text += "The model predicts **RESISTANCE** - key features have values that push the prediction toward higher AUC."
            
            return temp_file.name, top_fi_display, explanation_text
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, None, f"Error generating feature importance explanation: {str(e)}"
    
    def get_drug_info(self, drug_name):
        """Get target and pathway for a selected drug"""
        if not self.model_loaded or not drug_name:
            return "", ""
        
        # Try metadata first (deployment mode)
        if drug_name in self.drug_info:
            info = self.drug_info[drug_name]
            return info.get('target', ''), info.get('pathway', '')
        
        # Fallback to pipeline (development mode)
        if self.pipeline and hasattr(self.pipeline, 'merged_data'):
            if drug_name in self.drug_list:
                drug_data = self.pipeline.merged_data[
                    self.pipeline.merged_data['DRUG_NAME'] == drug_name
                ].iloc[0]
                return drug_data['PUTATIVE_TARGET'], drug_data['PATHWAY_NAME']
        
        return "", ""
    
    def list_available_cell_lines(self):
        """List available cell lines by name - only return if expression data is actually available"""
        if not self.model_loaded:
            return []
        
        # Only return cell lines if we have the actual expression data
        if self.pipeline and hasattr(self.pipeline, 'expression_data') and self.cell_line_list:
            return self.cell_line_list  # Returns cell line names, not ModelIDs
        
        # In deployment mode, we have metadata but not the actual expression data
        # So return empty list to indicate database is not available
        return []
    
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
                    
                    print(f"Row {idx + 1}: {drug_name} on {cell_line_id} -> AUC = {predicted_auc:.4f}")
                    
                except Exception as e:
                    print(f"Error processing row {idx + 1}: {str(e)}")
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
            print(f"Batch prediction error: {str(e)}")
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
        
        # Show database status
        if app.model_loaded and app.pipeline:
            gr.Markdown("""
            **Full Database Mode** 
            - 378 drugs available with auto-fill for targets and pathways
            - 1,699 cell lines with gene expression data
            - Or upload your own expression data
            """)
        else:
            gr.Markdown("""
            **Loading...** Please wait while the database loads
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
                export_results_btn = gr.Button("Download Prediction Results (CSV)", size="sm")
            results_download = gr.File(label="Download Results CSV", visible=True)
            
            biomarkers_output = gr.Dataframe(
                label="Top 15 Predictive Genes (Biomarkers)",
                headers=["Gene", "Importance"]
            )
            
            # Export button for biomarkers
            with gr.Row():
                export_biomarkers_btn = gr.Button("Download Biomarkers (CSV)", size="sm")
            biomarkers_download = gr.File(label="Download Biomarkers CSV", visible=True)
            
            # SHAP Explainability Section
            gr.Markdown("---")
            gr.Markdown("### SHAP Explainability (What's Driving the Prediction?)")
            gr.Markdown("*SHAP values show how each feature contributes to the prediction. Positive values push toward resistance, negative toward sensitivity.*")
            
            with gr.Row():
                shap_btn = gr.Button("Generate SHAP Explanation", variant="secondary")
            
            with gr.Row():
                with gr.Column(scale=1):
                    shap_explanation_text = gr.Markdown(
                        value="*Click 'Generate SHAP Explanation' after making a prediction*",
                        label="SHAP Analysis"
                    )
                    shap_table = gr.Dataframe(
                        label="Top Feature Contributions (SHAP Values)",
                        headers=["Feature", "SHAP Value", "Direction"]
                    )
                with gr.Column(scale=1):
                    shap_plot = gr.Image(
                        label="SHAP Waterfall Plot",
                        type="filepath"
                    )
            
            # Hidden state to store X_scaled for SHAP computation
            x_scaled_state = gr.State(value=None)
            drug_state = gr.State(value=None)
            cell_state = gr.State(value=None)
            
            # Auto-fill target and pathway when drug is selected
            drug_dropdown.change(
                fn=app.get_drug_info,
                inputs=[drug_dropdown],
                outputs=[target_input, pathway_input]
            )
            
            # Prediction function that stores X_scaled for later SHAP computation
            def predict_and_store(dropdown_drug, name, tgt, path, file, cell):
                """Run prediction and store X_scaled for SHAP"""
                # Use dropdown selection first, fall back to manual entry
                drug_name = dropdown_drug if dropdown_drug else (name if name else None)
                cell_name = cell if cell else "Uploaded Sample"
                
                print(f"[DEBUG] predict_and_store - dropdown_drug: {dropdown_drug}, name: {name}, drug_name: {drug_name}")
                
                # Get prediction results
                result = app.predict_drug_sensitivity(drug_name or "", tgt, path, file, cell)
                interpretation, results_df, biomarkers, X_scaled = result
                
                print(f"[DEBUG] X_scaled type: {type(X_scaled)}, drug_name stored: {drug_name}")
                
                # Return results and store state for SHAP
                return interpretation, results_df, biomarkers, X_scaled, drug_name, cell_name
            
            # SHAP explanation function using stored state
            def generate_shap_on_demand(x_scaled, drug_name, cell_name):
                """Generate SHAP explanation using stored prediction data"""
                if x_scaled is None:
                    return "*Please run a prediction first before generating SHAP explanation*", None, None
                
                if drug_name is None:
                    return "*Please select a drug to explain*", None, None
                
                try:
                    cell_display = cell_name if cell_name else "Selected Sample"
                    print(f"Generating SHAP explanation for {drug_name} on {cell_display}...")
                    print(f"X_scaled shape: {x_scaled.shape if hasattr(x_scaled, 'shape') else 'N/A'}")
                    
                    result = app.generate_shap_explanation(
                        x_scaled, drug_name, cell_display, top_n=20
                    )
                    
                    if result is None:
                        return "*Error: SHAP explanation returned None*", None, None
                    
                    shap_img, shap_df, shap_text = result
                    print(f"SHAP result - img: {type(shap_img)}, df: {type(shap_df)}, text length: {len(shap_text) if shap_text else 0}")
                    
                    return shap_text, shap_df, shap_img
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return f"*Error generating SHAP explanation: {str(e)}*", None, None
            
            # Prediction button - runs prediction and stores state
            predict_btn.click(
                fn=predict_and_store,
                inputs=[
                    drug_dropdown,
                    drug_name_input,
                    target_input,
                    pathway_input,
                    expression_file,
                    cell_line_dropdown
                ],
                outputs=[interpretation_output, results_output, biomarkers_output,
                        x_scaled_state, drug_state, cell_state]
            )
            
            # SHAP button - generates explanation from stored state
            shap_btn.click(
                fn=generate_shap_on_demand,
                inputs=[x_scaled_state, drug_state, cell_state],
                outputs=[shap_explanation_text, shap_table, shap_plot]
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
                    print(f"Results exported to: {temp_path}")
                    print(f"   File exists: {os.path.exists(temp_path)}")
                    print(f"   File size: {os.path.getsize(temp_path)} bytes")
                    return temp_path
                except Exception as e:
                    print(f"Export error: {str(e)}")
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
                    print(f"Biomarkers exported to: {temp_path}")
                    print(f"   File exists: {os.path.exists(temp_path)}")
                    print(f"   File size: {os.path.getsize(temp_path)} bytes")
                    return temp_path
                except Exception as e:
                    print(f"Export error: {str(e)}")
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
        
        with gr.Tab("🔍 Explainability Guide"):
            gr.Markdown("""
            ### Understanding SHAP Explanations
            
            This platform uses **SHAP (SHapley Additive exPlanations)** to provide transparent, 
            interpretable predictions. SHAP values explain how each feature contributes to 
            the final prediction.
            
            ---
            
            #### What is SHAP?
            
            SHAP is a game-theoretic approach to explain machine learning predictions:
            - **Based on Shapley values** from cooperative game theory
            - **Locally accurate**: SHAP values sum to the difference between prediction and average
            - **Feature attribution**: Shows positive/negative contribution of each feature
            
            ---
            
            #### How to Interpret SHAP Values
            
            | SHAP Value | Meaning | Effect on Drug Response |
            |------------|---------|------------------------|
            | **Positive (+)** | Pushes AUC higher | Drives **RESISTANCE** |
            | **Negative (-)** | Pushes AUC lower | Drives **SENSITIVITY** |
            | **Near zero** | Little impact | Neutral effect |
            
            ---
            
            #### Reading the Waterfall Plot
            
            The waterfall plot shows:
            1. **Base value**: Average prediction across all training data (~0.73 AUC)
            2. **Red bars (→)**: Features pushing prediction toward resistance (higher AUC)
            3. **Teal bars (←)**: Features pushing prediction toward sensitivity (lower AUC)
            4. **Final prediction**: Sum of base value + all SHAP contributions
            
            ---
            
            #### Feature Types Explained
            
            | Feature Type | Description | Example |
            |--------------|-------------|---------|
            | **Gene: XXXX** | Gene expression level | Gene: BRAF, Gene: TP53 |
            | **Drug Target** | Encoded target protein | BRAF, EGFR, BCR-ABL |
            | **Drug Pathway** | Biological pathway | ERK MAPK signaling |
            | **Drug Identity** | Specific drug encoding | Imatinib, PLX-4720 |
            | **MolFP_XXX** | Molecular fingerprint bit | Chemical structure features |
            
            ---
            
            #### Clinical Interpretation Example
            
            **Scenario**: Predicting Imatinib on a leukemia cell line
            
            ```
            Base prediction: 0.73 (average AUC)
            
            Top Sensitivity Drivers (↓ AUC):
            - Gene: BCR-ABL fusion: -0.15 (target present → sensitive)
            - Drug Target (BCR-ABL): -0.08 (correct target match)
            
            Top Resistance Drivers (↑ AUC):
            - Gene: MDR1: +0.05 (drug efflux pump expressed)
            
            Final prediction: 0.55 (Moderate sensitivity)
            ```
            
            ---
            
            #### Using SHAP for Biomarker Discovery
            
            1. **Identify key genes**: Features with highest absolute SHAP values
            2. **Resistance markers**: Genes with consistently positive SHAP
            3. **Sensitivity markers**: Genes with consistently negative SHAP
            4. **Drug-specific patterns**: Compare SHAP across different drugs
            
            ---
            
            #### Limitations
            
            - SHAP explains the **model's reasoning**, not biological ground truth
            - High SHAP doesn't guarantee causal relationship
            - Correlated features may share attribution
            - Best used alongside domain expertise
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

        # Get port from environment variable (Render provides this)
        import os
        port = int(os.environ.get("PORT", 7860))

        demo.launch(
            share=False,
            server_name="0.0.0.0",
            server_port=port,
            show_error=True,
            quiet=False,
            debug=True
        )
    except Exception as e:
        print(f"\n ERROR: Failed to start application")
        print(f"Error details: {str(e)}")
        import traceback
        traceback.print_exc()
