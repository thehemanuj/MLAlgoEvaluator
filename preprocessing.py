import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class DataPreprocessor:
    def __init__(self, dataset_path):
        self.dataset = dataset_path
        self.preprocessor = None
        
    def preprocess(self, features, target, scale_method='standard', 
                   encode_method='onehot', impute_strategy='mean'):
        """
        Preprocess dataset using scikit-learn pipelines.
        
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
            
        Returns:
        --------
        X_train, X_test, y_train, y_test : arrays
            Split and preprocessed data
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
        
        # Split data first (avoid data leakage)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=39, train_size=0.8
        )
        
        # Identify column types
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Build preprocessing pipelines
        
        # Numeric pipeline
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
            # Note: LabelEncoder doesn't work well with pipelines for multiple columns
            # Using OrdinalEncoder instead
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
            remainder='passthrough'  # Keep any other columns as-is
        )
        
        # Fit on training data and transform both sets
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Get feature names after transformation
        feature_names = self.get_feature_names()
        
        # Convert back to DataFrame (optional, for better readability)
        X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
        X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)
        
        return X_train_processed, X_test_processed, y_train, y_test
    
    def get_feature_names(self):
        """Get feature names after transformation"""
        if self.preprocessor is None:
            return []
        
        feature_names = []
        for name, transformer, columns in self.preprocessor.transformers_:
            if name == 'remainder':
                continue
            
            if hasattr(transformer, 'get_feature_names_out'):
                # For transformers that support get_feature_names_out
                names = transformer.get_feature_names_out(columns)
                feature_names.extend(names)
            else:
                # Fallback for transformers without this method
                feature_names.extend(columns)
        
        return feature_names
    
    def transform_new_data(self, new_data):
        """
        Transform new data using the fitted preprocessor.
        
        Parameters:
        -----------
        new_data : DataFrame
            New data to transform (must have same columns as training data)
            
        Returns:
        --------
        transformed_data : DataFrame
            Preprocessed data
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
    target = ['purchased']  # or just 'purchased'
    
    # Preprocess with different options
    X_train, X_test, y_train, y_test = preprocessor.preprocess(
        features=features,
        target=target,
        scale_method='standard',  # or 'minmax'
        encode_method='onehot',   # or 'label'
        impute_strategy='mean'    # or 'median', 'most_frequent'
    )
    
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
    print("\nFeature names after transformation:")
    print(X_train.columns.tolist())
    
    # Transform new data (e.g., for predictions)
    # new_data = pd.DataFrame({'age': [25], 'income': [50000], ...})
    # new_data_processed = preprocessor.transform_new_data(new_data)