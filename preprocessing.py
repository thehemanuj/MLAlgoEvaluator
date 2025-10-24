import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from collections import Counter


class DataPreprocessor:
    def __init__(self, dataset_path):
        self.dataset = dataset_path
        self.preprocessor = None
        
    def preprocess(self, features, target, scale_method='standard', 
                   encode_method='onehot', impute_strategy='mean',
                   handle_imbalance=None, imbalance_strategy='smote',
                   sampling_ratio='auto'):
        """
        Preprocess dataset using scikit-learn pipelines with imbalance handling.
        
        Parameters:
        -----------
        features : list
            List of feature column names
        target : str or list
            Target column name (or list with single element)
        scale_method : str, default='standard'
            'standard' for StandardScaler, 'minmax' for MinMaxScaler
        encode_method : str, default='onehot'
            'onehot' for OneHotEncoder, 'label' for LabelEncoder
        impute_strategy : str, default='mean'
            Strategy for imputing numeric values ('mean', 'median', 'most_frequent')
        handle_imbalance : bool or None, default=None
            Whether to handle class imbalance. If None, auto-detects imbalance
        imbalance_strategy : str, default='smote'
            Strategy for handling imbalance:
            - 'smote': Synthetic Minority Over-sampling Technique
            - 'adasyn': Adaptive Synthetic Sampling
            - 'random_oversample': Random oversampling of minority class
            - 'random_undersample': Random undersampling of majority class
            - 'smote_tomek': SMOTE + Tomek links cleaning
            - 'smote_enn': SMOTE + Edited Nearest Neighbors cleaning
        sampling_ratio : float, str, or dict, default='auto'
            Sampling ratio for resampling. 'auto' handles automatically
            
        Returns:
        --------
        X_train, X_test, y_train, y_test : arrays
            Split and preprocessed data
        class_distribution : dict
            Distribution of classes before and after resampling
        """
        # Load dataset
        df = pd.read_csv(self.dataset)
        
        # Handle target as string
        target_col = target[0] if isinstance(target, list) else target
        
        # Select relevant columns
        selected_columns = features + [target_col]
        df = df[selected_columns]
        
        # Separate features and target
        X = df[features]
        y = df[target_col]
        
        # Check class distribution
        original_distribution = Counter(y)
        print(f"Original class distribution: {dict(original_distribution)}")
        
        # Auto-detect imbalance if not specified
        if handle_imbalance is None:
            handle_imbalance = self._detect_imbalance(y)
            if handle_imbalance:
                print("⚠️  Class imbalance detected! Applying resampling strategy.")
        
        # Split data first (avoid data leakage)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=39, train_size=0.8, stratify=y
        )
        
        train_distribution_before = Counter(y_train)
        print(f"Training set distribution before resampling: {dict(train_distribution_before)}")
        
        # Identify column types
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Build preprocessing pipelines
        if scale_method == 'standard':
            scaler = StandardScaler()
        elif scale_method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("scale_method must be 'standard' or 'minmax'")
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=impute_strategy)),
            ('scaler', scaler)
        ])
        
        # Categorical pipeline
        if encode_method == 'onehot':
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
            ])
        elif encode_method == 'label':
            from sklearn.preprocessing import OrdinalEncoder
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])
        else:
            raise ValueError("encode_method must be 'onehot' or 'label'")
        
        # Combine pipelines
        transformers = []
        if numeric_cols:
            transformers.append(('num', numeric_transformer, numeric_cols))
        if categorical_cols:
            transformers.append(('cat', categorical_transformer, categorical_cols))
        
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        )
        
        # Fit on training data and transform both sets
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Handle class imbalance AFTER preprocessing
        if handle_imbalance:
            X_train_processed, y_train = self._apply_resampling(
                X_train_processed, y_train, 
                strategy=imbalance_strategy,
                sampling_ratio=sampling_ratio
            )
            train_distribution_after = Counter(y_train)
            print(f"Training set distribution after resampling: {dict(train_distribution_after)}")
        
        # Get feature names after transformation
        feature_names = self.get_feature_names()
        
        # Convert back to DataFrame
        X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names)
        X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)
        
        # Prepare class distribution summary
        class_distribution = {
            'original': dict(original_distribution),
            'train_before': dict(train_distribution_before),
            'train_after': dict(Counter(y_train)) if handle_imbalance else dict(train_distribution_before),
            'test': dict(Counter(y_test))
        }
        
        return X_train_processed, X_test_processed, y_train, y_test, class_distribution
    
    def _detect_imbalance(self, y, threshold=0.3):
        """
        Detect if classes are imbalanced.
        
        Parameters:
        -----------
        y : Series
            Target variable
        threshold : float
            Ratio threshold below which imbalance is detected
            
        Returns:
        --------
        bool : True if imbalanced
        """
        class_counts = Counter(y)
        if len(class_counts) < 2:
            return False
        
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        ratio = min_count / max_count
        
        return ratio < threshold
    
    def _apply_resampling(self, X, y, strategy='smote', sampling_ratio='auto'):
        """
        Apply resampling strategy to handle imbalance.
        
        Parameters:
        -----------
        X : array
            Feature matrix
        y : array
            Target variable
        strategy : str
            Resampling strategy
        sampling_ratio : float, str, or dict
            Sampling ratio
            
        Returns:
        --------
        X_resampled, y_resampled : arrays
        """
        # Select resampling method
        if strategy == 'smote':
            sampler = SMOTE(sampling_strategy=sampling_ratio, random_state=42)
        elif strategy == 'adasyn':
            sampler = ADASYN(sampling_strategy=sampling_ratio, random_state=42)
        elif strategy == 'random_oversample':
            sampler = RandomOverSampler(sampling_strategy=sampling_ratio, random_state=42)
        elif strategy == 'random_undersample':
            sampler = RandomUnderSampler(sampling_strategy=sampling_ratio, random_state=42)
        elif strategy == 'smote_tomek':
            sampler = SMOTETomek(sampling_strategy=sampling_ratio, random_state=42)
        elif strategy == 'smote_enn':
            sampler = SMOTEENN(sampling_strategy=sampling_ratio, random_state=42)
        else:
            raise ValueError(f"Unknown imbalance strategy: {strategy}")
        
        try:
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            return X_resampled, y_resampled
        except Exception as e:
            print(f"⚠️  Resampling failed: {e}")
            print("Returning original data without resampling.")
            return X, y
    
    def get_feature_names(self):
        """Get feature names after transformation"""
        if self.preprocessor is None:
            return []
        
        feature_names = []
        for name, transformer, columns in self.preprocessor.transformers_:
            if name == 'remainder':
                continue
            
            if hasattr(transformer, 'get_feature_names_out'):
                names = transformer.get_feature_names_out(columns)
                feature_names.extend(names)
            else:
                feature_names.extend(columns)
        
        return feature_names
    
    def transform_new_data(self, new_data):
        """
        Transform new data using the fitted preprocessor.
        
        Parameters:
        -----------
        new_data : DataFrame
            New data to transform
            
        Returns:
        --------
        transformed_data : DataFrame
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call preprocess() first.")
        
        transformed = self.preprocessor.transform(new_data)
        feature_names = self.get_feature_names()
        
        return pd.DataFrame(transformed, columns=feature_names, index=new_data.index)


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DataPreprocessor('your_dataset.csv')
    
    # Define features and target
    features = ['age', 'income', 'gender', 'city', 'score']
    target = ['purchased']
    
    # Preprocess with imbalance handling
    X_train, X_test, y_train, y_test, distribution = preprocessor.preprocess(
        features=features,
        target=target,
        scale_method='standard',
        encode_method='onehot',
        impute_strategy='mean',
        handle_imbalance=True,  # Set to True to handle imbalance, None for auto-detect
        imbalance_strategy='smote',  # Options: 'smote', 'adasyn', 'random_oversample', etc.
        sampling_ratio='auto'  # 'auto' or specific ratio like 0.5
    )
    
    print("\n" + "="*60)
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
    print("\nClass Distribution Summary:")
    for key, value in distribution.items():
        print(f"  {key}: {value}")
    print("="*60)