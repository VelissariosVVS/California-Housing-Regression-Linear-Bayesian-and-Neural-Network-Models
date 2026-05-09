import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


### ------------------------------ ###
### Loading the dataset            ###   
### ------------------------------ ###
def load_california_housing() -> pd.DataFrame:
    housing = fetch_california_housing()

    #  Put data in a pandas DataFrame
    df = pd.DataFrame(
        housing.data,
        columns=housing.feature_names
    )

    df['MedHouseValue'] = housing.target

    return df

### ------------------------------ ###
### Removing outliers              ###   
### ------------------------------ ###

def remove_extreme_occupancy_outliers(
    df: pd.DataFrame,
    threshold: float = 10.0
) -> pd.DataFrame:
    
    df_clean = df[df['AveOccup'] <= threshold].copy()

    return df_clean

### ------------------------------------- ###
### Split the features and target columns ###   
### ------------------------------------- ###

def split_features_target(
        df: pd.DataFrame,
        target_column: str = 'MedHouseValue'
):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y

### -------------------------------- ###
### Train, validation and test split ###
### -------------------------------- ###

def train_validation_test_split(
    X,
    y,
    train_size: float = 0.7,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
):
    """
    Split data into train, validation, and test sets.
    """

    # First split: train vs temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(validation_size + test_size),
        random_state=random_state
    )

    # Compute relative validation size inside temp set
    validation_ratio = validation_size / (
        validation_size + test_size
    )

    # Second split: validation vs test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1 - validation_ratio),
        random_state=random_state
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )

### ------------------------------ ###
### Standardization of the data    ###
### ------------------------------ ###

def scale_feature(X_train, X_val, X_test):
    """
    Standardize features by removing mean and scaling to unit variance.
    
    Args:
        X_train: Training features
        X_test: Testing features
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

### ------------------------------ ###
### Final preparation of the data  ###  
### ------------------------------ ###

def prepare_data(
    occupancy_threshold: float = 10.0,
    train_size: float = 0.70,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
):
    """
    Full preprocessing pipeline:
    1. Load dataset
    2. Remove extreme AveOccup outliers
    3. Split features and target
    4. Split into train, validation, and test sets
    5. Scale features using only the training data
    """

    df = load_california_housing()

    df_clean = remove_extreme_occupancy_outliers(
        df,
        threshold=occupancy_threshold
    )

    X, y = split_features_target(df_clean)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = train_validation_test_split(
        X,
        y,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        random_state=random_state
    )

    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_feature(
        X_train, 
        X_val, 
        X_test
    )

    return {
        "df": df,
        "df_clean": df_clean,

        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,

        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,

        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,

        "scaler": scaler,
        "feature_names": X.columns.tolist()
    }