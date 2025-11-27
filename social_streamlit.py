import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle
import io

# Import your classes (adjust import paths as needed)
from preprocessing import DataPreprocessor
from social_classification import Classification
from social_regression import Regression

st.set_page_config(page_title="ML Model Evaluator", layout="wide", page_icon="🤓")
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

# Sidebar: Imbalance handling (only for classification)
if task_selector == "Classification":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Imbalance Handling")
    handle_imbalance = st.sidebar.checkbox("Handle Class Imbalance", value=False)
    
    if handle_imbalance:
        imbalance_strategy = st.sidebar.selectbox("Resampling Strategy", [
            "smote", "adasyn", "random_oversample", "random_undersample", 
            "smote_tomek", "smote_enn"
        ])
    else:
        imbalance_strategy = None
else:
    handle_imbalance = None
    imbalance_strategy = None

# Sidebar: Dataset selection
st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Selection")
dataset_option = st.sidebar.radio("Choose Dataset Source", ["Use Preloaded Dataset", "Upload Your Own"])

selected_dataset_path = None

if dataset_option == "Use Preloaded Dataset":
    dataset_choice = st.sidebar.selectbox("Select a Dataset", ["-- Select --", "social.csv", "Position_Salaries.csv", "50_Startups.csv"])
    if dataset_choice != "-- Select --":
        selected_dataset_path = 'datasets/' + dataset_choice
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
        st.dataframe(df.head(10), width="stretch")
        
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
            
            # Data Visualization BEFORE model training
            st.markdown("---")
            st.markdown("### 📊 Data Visualization")
            
            # For Classification: Show target distribution and check imbalance
            if task_selector == "Classification":
                target_counts = df[target].value_counts()
                
                # Check for imbalance
                min_count = target_counts.min()
                max_count = target_counts.max()
                imbalance_ratio = min_count / max_count
                
                col_viz1, col_viz2 = st.columns(2)
                
                with col_viz1:
                    st.markdown("#### Target Variable Distribution")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    target_counts.plot(kind='bar', ax=ax, color='steelblue')
                    ax.set_xlabel(target)
                    ax.set_ylabel('Count')
                    ax.set_title(f'Distribution of {target}')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    plt.close()
                
                with col_viz2:
                    st.markdown("#### Class Balance Analysis")
                    
                    # Display class counts
                    st.write("**Class Counts:**")
                    for class_label, count in target_counts.items():
                        percentage = (count / len(df)) * 100
                        st.write(f"- {class_label}: {count} ({percentage:.2f}%)")
                    
                    # Imbalance indicator
                    st.write(f"\n**Imbalance Ratio:** {imbalance_ratio:.3f}")
                    
                    if imbalance_ratio < 0.3:
                        st.error("⚠️ **Severe Class Imbalance Detected!**")
                    elif imbalance_ratio < 0.5:
                        st.warning("⚠️ **Moderate Class Imbalance Detected.**")
                    else:
                        st.success("✅ **Classes are relatively balanced.**")
            
            # For Regression: Show target distribution
            else:
                col_viz1, col_viz2 = st.columns(2)
                
                with col_viz1:
                    st.markdown("#### Target Variable Distribution")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.hist(df[target].dropna(), bins=30, color='steelblue', edgecolor='black')
                    ax.set_xlabel(target)
                    ax.set_ylabel('Frequency')
                    ax.set_title(f'Distribution of {target}')
                    st.pyplot(fig)
                    plt.close()
                
                with col_viz2:
                    st.markdown("#### Target Statistics")
                    stats = df[target].describe()
                    st.write(stats)
            
            # Feature correlation heatmap
            if len(features) > 1:
                st.markdown("#### Feature Correlation Heatmap")
                numeric_features = df[features].select_dtypes(include=['number']).columns.tolist()
                
                if len(numeric_features) > 1:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    corr_matrix = df[numeric_features].corr()
                    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, center=0)
                    ax.set_title('Feature Correlation Matrix')
                    st.pyplot(fig)
                    plt.close()
            
            # Train button
            if st.button("🚀 Train Model", type="primary", width="stretch"):
                with st.spinner("Training model... Please wait..."):
                    try:
                        # Step 1: Preprocess data
                        st.info("📊 Step 1/2: Preprocessing data...")
                        preprocessor = DataPreprocessor(selected_dataset_path)
                        
                        # Call preprocess - pass task_type from user selection
                        if task_selector == "Classification" and handle_imbalance:
                            X_train, X_test, y_train, y_test, distribution = preprocessor.preprocess(
                                features=features,
                                target=target,
                                scale_method=scale_method,
                                encode_method=encode_method,
                                impute_strategy=impute_strategy,
                                handle_imbalance=True,
                                imbalance_strategy=imbalance_strategy,
                                task_type='classification'
                            )
                            st.session_state.distribution = distribution
                        elif task_selector == "Classification":
                            X_train, X_test, y_train, y_test, distribution = preprocessor.preprocess(
                                features=features,
                                target=target,
                                scale_method=scale_method,
                                encode_method=encode_method,
                                impute_strategy=impute_strategy,
                                handle_imbalance=False,
                                imbalance_strategy=None,
                                task_type='classification'
                            )
                            st.session_state.distribution = distribution
                        else:
                            # Regression - no imbalance handling
                            X_train, X_test, y_train, y_test, distribution = preprocessor.preprocess(
                                features=features,
                                target=target,
                                scale_method=scale_method,
                                encode_method=encode_method,
                                impute_strategy=impute_strategy,
                                handle_imbalance=False,
                                imbalance_strategy=None,
                                task_type='regression'
                            )
                            st.session_state.distribution = None
                        
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
                        st.session_state.y_test = y_test

                        if hasattr(model, 'model'):
                            st.session_state.trained_model = model.model
                        elif hasattr(model, 'best_model'):
                            st.session_state.trained_model = model.best_model
                        else:
                            # If the structure is different, you may need to adjust this
                            st.session_state.trained_model = model
                        
                        st.success("✅ Training Complete!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Error during training: {str(e)}")
                        st.exception(e)
        
        # Display results if trained
        if st.session_state.trained and st.session_state.results:
            st.markdown("---")
            st.markdown("### 📈 Training Results")
            
            # Show imbalance handling results if applied
            if st.session_state.task_type == "Classification" and st.session_state.distribution:
                dist = st.session_state.distribution
                
                # Only show distribution if it's a valid classification distribution
                if 'train_before' in dist and 'train_after' in dist:
                    # Check if distributions are different (meaning resampling was applied)
                    if dist['train_before'] != dist['train_after']:
                        st.markdown("#### ⚖️ Class Distribution (After Resampling)")
                        
                        col_d1, col_d2 = st.columns(2)
                        
                        with col_d1:
                            st.write("**Before Resampling:**")
                            for class_label, count in dist['train_before'].items():
                                st.write(f"- Class {class_label}: {count}")
                        
                        with col_d2:
                            st.write("**After Resampling:**")
                            for class_label, count in dist['train_after'].items():
                                st.write(f"- Class {class_label}: {count}")
            
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
                    st.dataframe(comparison_df, width="stretch")
                
                # Confusion Matrix
                st.markdown("#### 🎯 Confusion Matrix")
                conf_matrix = model.get_confusion_matrix()
                st.write(conf_matrix)
                
                # Classification Report
                with st.expander("📋 Detailed Classification Report"):
                    st.text(model.get_classification_report())
                
                # Visualization AFTER model training
                st.markdown("---")
                st.markdown("### 📊 Model Performance Visualization")
                
                col_viz1, col_viz2 = st.columns(2)
                
                with col_viz1:
                    st.markdown("#### Confusion Matrix Heatmap")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title('Confusion Matrix')
                    st.pyplot(fig)
                    plt.close()
                
                with col_viz2:
                    st.markdown("#### Prediction Distribution")
                    y_test = st.session_state.y_test
                    predictions = metrics['predictions']
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    x_pos = range(len(Counter(y_test)))
                    actual_counts = [count for _, count in sorted(Counter(y_test).items())]
                    pred_counts = [count for _, count in sorted(Counter(predictions).items())]
                    
                    width = 0.35
                    ax.bar([x - width/2 for x in x_pos], actual_counts, width, label='Actual', color='steelblue')
                    ax.bar([x + width/2 for x in x_pos], pred_counts, width, label='Predicted', color='coral')
                    
                    ax.set_xlabel('Classes')
                    ax.set_ylabel('Count')
                    ax.set_title('Actual vs Predicted Distribution')
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(sorted(Counter(y_test).keys()))
                    ax.legend()
                    st.pyplot(fig)
                    plt.close()
            
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
                    st.dataframe(comparison_df, width="stretch")
                
                # Visualization AFTER model training
                st.markdown("---")
                st.markdown("### 📊 Model Performance Visualization")
                
                y_test = st.session_state.y_test
                predictions = metrics['predictions']
                
                col_viz1, col_viz2 = st.columns(2)
                
                with col_viz1:
                    st.markdown("#### Actual vs Predicted")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(y_test, predictions, alpha=0.6, color='steelblue')
                    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                    ax.set_xlabel('Actual Values')
                    ax.set_ylabel('Predicted Values')
                    ax.set_title('Actual vs Predicted Values')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
                
                with col_viz2:
                    st.markdown("#### Residuals Distribution")
                    residuals = y_test - predictions
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.hist(residuals, bins=30, color='coral', edgecolor='black')
                    ax.set_xlabel('Residuals')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Distribution of Residuals')
                    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
                    st.pyplot(fig)
                    plt.close()

                    
            st.markdown('---')
            st.markdown('### 🗃️ Download Trained Model')
            
            if st.session_state.trained_model is not None:
                # Serialize the model to bytes using pickle
                model_bytes = io.BytesIO()
                pickle.dump(st.session_state.trained_model, model_bytes)
                model_bytes.seek(0)
                
                st.download_button(
                    label="📦 Download Model as PKL",
                    data=model_bytes,
                    file_name=f"{st.session_state.task_type.lower()}_model.pkl",
                    mime="application/octet-stream",
                    help="Download the trained model as a pickle file for later use"
                )
            else:
                st.warning("⚠️ Model not available for download. Please train a model first.")
            
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
    Built with 🧠 by Ayush | Machine Learning Model Evaluator
    </div>
    """,
    unsafe_allow_html=True
)
