import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y, title="Learning Curve", cv=5, scoring='balanced_accuracy'):
    """
    Plot learning curve for a given estimator and dataset.

    Parameters
    ----------
    estimator : sklearn estimator
        The estimator to plot the learning curve for.
    X : array-like of shape (n_samples, n_features)
        The input data.
    y : array-like of shape (n_samples,)
        The target values.
    title : str, default="Learning Curve"
        The title of the plot.
    cv : int, default=5
        The number of folds for cross-validation.
    scoring : str, default='balanced_accuracy'
        The scoring metric to use.

    Returns
    -------
    None
    """
        # Generate learning curve data

    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    # Compute mean and standard deviation for training and validation scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training Set Size")
    plt.ylabel(scoring)
    plt.grid()

    # Plot training score
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")

    # Plot validation score
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()

def plot_error_curve(estimator, X, y, title="Error Curve", cv=5, scoring='balanced_accuracy'):
    """
    Plot error curve for a given estimator and dataset.

    Parameters
    ----------
    estimator : sklearn estimator
        The estimator to plot the error curve for.
    X : array-like of shape (n_samples, n_features)
        The input data.
    y : array-like of shape (n_samples,)
        The target values.
    title : str, default="Error Curve"
        The title of the plot.
    cv : int, default=5
        The number of folds for cross-validation.
    scoring : str, default='balanced_accuracy'
        The scoring metric to use.

    Returns
    -------
    None
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    # Compute errors
    train_errors = 1 - np.mean(train_scores, axis=1)
    val_errors = 1 - np.mean(val_scores, axis=1)

    # Plot error curve
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training Set Size")
    plt.ylabel("Error")
    plt.grid()

    plt.plot(train_sizes, train_errors, 'o-', color="r", label="Training error")
    plt.plot(train_sizes, val_errors, 'o-', color="g", label="Cross-validation error")
    
    plt.legend(loc="best")
    plt.show()

def get_feature_names(preprocessor, feature_selector, k):
    """
    Get feature names from a ColumnTransformer and SelectKBest.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        The preprocessor to get feature names from.
    feature_selector : SelectKBest
        The feature selector to get feature names from.
    k : int
        The number of features to select.

    Returns
    -------
    list of str
        The selected feature names.
    """

        # Extract numerical feature names

    num_features = preprocessor.named_transformers_['num']['scaler'].get_feature_names_out()
    
    # Extract categorical feature names after one-hot encoding
    cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out()
    
    # Combine all feature names
    all_features = np.concatenate([num_features, cat_features])
    
    # Get the indices of features selected by SelectKBest
    selected_indices = feature_selector.get_support(indices=True)
    
    # Map the selected indices to the actual feature names
    selected_features = all_features[selected_indices]
    
    return selected_features



def get_feature_importances(model, feature_names):
    """
    Extracts feature importances from a model and returns them in a sorted DataFrame.

    Parameters
    ----------
    model : estimator object
        A trained model object that supports extraction of feature importances,
        such as having `feature_importances_` or `coef_` attributes.
    feature_names : list of str
        The list of feature names corresponding to the model's input features.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing feature names and their corresponding importances,
        sorted in descending order of importance.

    Raises
    ------
    ValueError
        If the model does not support feature importance extraction or if there is
        a mismatch between the number of feature names and importances.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).ravel()
    else:
        raise ValueError("The model does not support feature importance extraction.")
    
    if len(feature_names) != len(importances):
        raise ValueError(f"Mismatch in feature names ({len(feature_names)}) and importances ({len(importances)}).")
    
    return pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)




def plot_feature_importance(feature_importance_df, top_n=10):
    """
    Plots top N feature importances in a bar chart.

    Parameters
    ----------
    feature_importance_df : pandas.DataFrame
        A DataFrame containing feature names and their corresponding importances,
        sorted in descending order of importance.
    top_n : int, default=10
        The number of top features to plot.

    Returns
    -------
    None
    """
        # Select top N features

    top_features = feature_importance_df.head(top_n)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=top_features, 
        x='Importance', 
        y='Feature', 
        palette='viridis'
    )
    plt.title('Top Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()



# Assuming `data` is your DataFrame
# Step 3: Split Data into Features and Target
X = data.drop(columns=['Feedback_Score'])
y = data['Feedback_Score']  # Ensure `y` is categorical
y = y.astype('category')  # Explicitly treat as categorical


# Step 4: Preprocessing Pipeline
numeric_features = ['Age', 'Visit_Frequency', 'Average_Spend_Per_Visit', 'Total_Spend']
categorical_features = ['Gender', 'Service_Type', 'Loyalty_Program', 'Promotional_Response', 
                        'Age_Group', 'Visit_Frequency_Group', 'Customer_Value']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Step 5: Combined Feature Selection and Modeling
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42)
}

# Cross-validation results
best_model = None
best_score = float('-inf')

k = min(24, X.shape[1])  # Ensure k doesn't exceed the number of features


for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(score_func=f_classif, k=k)),
        ('model', model)
    ])
    
    # Perform cross-validation directly on the pipeline
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='balanced_accuracy')
    mean_score = np.mean(scores)
    
    print(f"Model: {name}, Cross-Validation Accuracy: {mean_score:.3f}")
    
    if mean_score > best_score:
        best_score = mean_score
        best_model = (name, pipeline)

# Step 6: Train Final Model on Entire Training Data
final_model_name, final_pipeline = best_model


# Plot the learning curve
plot_learning_curve(final_pipeline, X, y, title=f"Learning Curve for {final_model_name}")
# Plot error curve
plot_error_curve(final_pipeline, X, y, title=f"Error Curve for {final_model_name}")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
final_pipeline.fit(X_train, y_train)
# Get preprocessor and feature selector from the final pipeline
#preprocessor = final_pipeline.named_steps['preprocessor']
#feature_selector = final_pipeline.named_steps['feature_selection']
#model = final_pipeline.named_steps['model']

# Extract feature names
#selected_features = get_feature_names(preprocessor, feature_selector, k=k)

# Get feature importances
#feature_importance_df = get_feature_importances(model, selected_features)

# Plot top 10 features
#plot_feature_importance(feature_importance_df, top_n=10)

print(classification_report(y_test, final_pipeline.predict(X_test)))

# Save the model
joblib.dump(final_pipeline, f'{final_model_name}_Feedback_Score_model.pkl')

print(f"Best Model: {final_model_name} with Cross-Validation Accuracy: {best_score:.3f}")
