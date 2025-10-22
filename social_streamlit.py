import streamlit as st
import pandas as pd

# Import your classes (adjust import paths as needed)
from preprocessing import DataPreprocessor
from social_classification import Classification
from social_regression import Regression

st.set_page_config(page_title="ML Model Evaluator", layout="wide")
st.title("📊 Supervised Machine Learning Evaluator")

# Initialize session state
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'model_instance' not in st.session_state:
    st.session_state.model_instance = None

# Sidebar: Task selection
st.sidebar.header("⚙️ Configuration")
task_selector = st.sidebar.radio("Select Task", ["Classification", "Regression"])

# Sidebar: Model selection
st.sidebar.subheader("Model Selection")
if task_selector == "Classification":
    model_choice = st.sidebar.selectbox("Choose a Model", [
        "Best (Auto-select)", "Logistic Regression", "Support Vector", "KNN", 
        "Decision Tree", "Random Forest", "Naive Bayes", "Gradient Boosting", 
        "AdaBoost", "Extra Trees", "QDA", "MLP", "Ridge Classifier"
    ])
    dictionary_model = {
        "Best (Auto-select)": 'best',
        "Logistic Regression": 'lr',
        "Support Vector": 'svc',
        "KNN": 'knn',
        "Decision Tree": 'dtc',
        "Random Forest": 'rfc',
        "Naive Bayes": 'gnb',
        "Gradient Boosting": 'gbc',
        "AdaBoost": 'abc',
        "Extra Trees": 'etc',
        "QDA": 'qda',
        "MLP": 'mlp',
        "Ridge Classifier": 'ridge'
    }
else:
    model_choice = st.sidebar.selectbox("Choose a Model", [
        "Best (Auto-select)", "Linear Regression", "Ridge Regression", 
        "Lasso Regression", "Elastic Net", "Bayesian Ridge", 
        "Support Vector", "Decision Tree", "Random Forest"
    ])
    dictionary_model = {
        "Best (Auto-select)": 'best',
        "Linear Regression": 'linreg',
        "Ridge Regression": 'ridge',
        "Lasso Regression": 'lasso',
        "Elastic Net": 'el',
        "Bayesian Ridge": 'br',
        "Support Vector": 'svr',
        "Decision Tree": 'dtr',
        "Random Forest": 'rfr'
    }

selected_model_key = dictionary_model.get(model_choice)

# Sidebar: Preprocessing options
st.sidebar.markdown("---")
st.sidebar.subheader("Preprocessing Options")
scale_method = st.sidebar.selectbox("Scaling Method", ["standard", "minmax"])
encode_method = st.sidebar.selectbox("Encoding Method", ["onehot", "label"])
impute_strategy = st.sidebar.selectbox("Imputation Strategy", ["mean", "median", "most_frequent"])

# Sidebar: Dataset selection
st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Selection")
dataset_option = st.sidebar.radio("Choose Dataset Source", ["Use Preloaded Dataset", "Upload Your Own"])

selected_dataset_path = None

if dataset_option == "Use Preloaded Dataset":
    dataset_choice = st.sidebar.selectbox("Select a Dataset", ["-- Select --", "social.csv"])
    if dataset_choice != "-- Select --":
        selected_dataset_path = dataset_choice
elif dataset_option == "Upload Your Own":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        filename = uploaded_file.name
        with open(filename, "wb") as f:
            f.write(uploaded_file.getbuffer())
        selected_dataset_path = filename

# Main content
st.markdown("---")
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"📋 Task: {task_selector}")

with col2:
    st.subheader(f"🤖 Model: {model_choice}")

# Dataset preview and feature selection
if selected_dataset_path is not None:
    try:
        df = pd.read_csv(selected_dataset_path)
        
        st.markdown("### 📊 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Total Rows", df.shape[0])
        with col_info2:
            st.metric("Total Columns", df.shape[1])
        with col_info3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        st.markdown("---")
        st.markdown("### 🎯 Feature Selection")
        
        col_feat, col_target = st.columns([3, 1])
        
        with col_feat:
            features = st.multiselect(
                "Select Features (Independent Variables)", 
                df.columns.tolist(),
                help="Choose one or more features for training"
            )
        
        with col_target:
            # Default to last column
            default_idx = len(df.columns) - 1
            target = st.selectbox(
                "Select Target (Dependent Variable)", 
                df.columns.tolist(),
                index=default_idx,
                help="Choose the target variable to predict"
            )
        
        # Validation
        if not features:
            st.warning("⚠️ Please select at least one feature.")
        elif target in features:
            st.error("❌ Target variable cannot be in the features list!")
        else:
            st.success(f"✅ Selected {len(features)} features and 1 target variable.")
            
            # Train button
            if st.button("🚀 Train Model", type="primary", use_container_width=True):
                with st.spinner("Training model... Please wait..."):
                    try:
                        # Step 1: Preprocess data
                        st.info("📊 Step 1/2: Preprocessing data...")
                        preprocessor = DataPreprocessor(selected_dataset_path)
                        X_train, X_test, y_train, y_test = preprocessor.preprocess(
                            features=features,
                            target=target,
                            scale_method=scale_method,
                            encode_method=encode_method,
                            impute_strategy=impute_strategy
                        )
                        
                        # Step 2: Train model
                        st.info("🤖 Step 2/2: Training model...")
                        if task_selector == "Classification":
                            model = Classification(X_train, X_test, y_train, y_test)
                        else:
                            model = Regression(X_train, X_test, y_train, y_test)
                        
                        results = model.train(key=selected_model_key)
                        
                        # Store in session state
                        st.session_state.trained = True
                        st.session_state.results = results
                        st.session_state.model_instance = model
                        st.session_state.task_type = task_selector
                        
                        st.success("✅ Training Complete!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Error during training: {str(e)}")
                        st.exception(e)
        
        # Display results if trained
        if st.session_state.trained and st.session_state.results:
            st.markdown("---")
            st.markdown("### 📈 Training Results")
            
            results = st.session_state.results
            model = st.session_state.model_instance
            
            # Display based on task type
            if st.session_state.task_type == "Classification":
                # Get best model info
                if selected_model_key == 'best':
                    best_model_key = results['best_model']
                    best_model_name = [name for name, key in dictionary_model.items() if key == best_model_key][0]
                    metrics = results['metrics']
                    
                    st.info(f"🏆 **Best Model Selected:** {best_model_name}")
                else:
                    metrics = results['metrics']
                
                # Display metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
                with col2:
                    st.metric("Precision", f"{metrics['precision']:.4f}")
                with col3:
                    st.metric("Recall", f"{metrics['recall']:.4f}")
                with col4:
                    st.metric("F1 Score", f"{metrics['f1']:.4f}")
                
                # Model comparison table (if best was selected)
                if selected_model_key == 'best':
                    st.markdown("#### 📊 Model Comparison")
                    comparison_df = model.get_model_comparison()
                    st.dataframe(comparison_df, use_container_width=True)
                
                # Confusion Matrix
                st.markdown("#### 🎯 Confusion Matrix")
                conf_matrix = model.get_confusion_matrix()
                st.write(conf_matrix)
                
                # Classification Report
                with st.expander("📋 Detailed Classification Report"):
                    st.text(model.get_classification_report())
            
            else:  # Regression
                # Get best model info
                if selected_model_key == 'best':
                    best_model_key = results['best_model']
                    best_model_name = [name for name, key in dictionary_model.items() if key == best_model_key][0]
                    metrics = results['metrics']
                    
                    st.info(f"🏆 **Best Model Selected:** {best_model_name}")
                else:
                    metrics = results['metrics']
                
                # Display metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("R² Score", f"{metrics['r2']:.4f}")
                with col2:
                    st.metric("MSE", f"{metrics['mse']:.4f}")
                with col3:
                    st.metric("RMSE", f"{metrics['rmse']:.4f}")
                with col4:
                    st.metric("MAE", f"{metrics['mae']:.4f}")
                
                # Model comparison table (if best was selected)
                if selected_model_key == 'best':
                    st.markdown("#### 📊 Model Comparison")
                    comparison_df = model.get_model_comparison()
                    st.dataframe(comparison_df, use_container_width=True)
            
            # Download predictions
            st.markdown("---")
            st.markdown("### 💾 Download Results")
            
            if task_selector == "Classification":
                predictions_df = pd.DataFrame({
                    'Actual': y_test,
                    'Predicted': metrics['predictions']
                })
            else:
                predictions_df = pd.DataFrame({
                    'Actual': y_test,
                    'Predicted': metrics['predictions']
                })
            
            csv = predictions_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name=f"{task_selector.lower()}_predictions.csv",
                mime="text/csv"
            )
            
    except FileNotFoundError:
        st.error(f"❌ File '{selected_dataset_path}' not found!")
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        st.exception(e)
else:
    st.info("👈 Please select a dataset from the sidebar to get started.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    Built with ❤️ by Ayush | Machine Learning Model Evaluator
    </div>
    """,
    unsafe_allow_html=True
)
