import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(['filename', 'length'], axis=1)
    X = df.drop('label', axis=1)
    y = df['label']
    return X, y


def explore_data(X, y):
    print("Dataset shape:", X.shape)
    print("Number of genres:", y.nunique())
    print("Genre distribution:")
    print(y.value_counts())
    print("\nMissing values:", X.isnull().sum().sum())
    print("\nFeature statistics:")
    print(X.describe())
    return X, y


def preprocess_data(X, y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, label_encoder, scaler


def analyze_features(X, y):
    plt.figure(figsize=(12, 10))
    correlation_matrix = X.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=pd.Categorical(y).codes,
                          cmap='tab10', alpha=0.6)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('PCA Visualization of Feature Space')
    plt.colorbar(scatter)
    plt.show()
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    return pca


def train_knn(X_train, y_train, X_test, y_test):
    param_grid = {
        'n_neighbors': [1, 3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_knn = grid_search.best_estimator_
    y_pred = best_knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"k-NN Best Parameters: {grid_search.best_params_}")
    print(f"k-NN Accuracy: {accuracy:.4f}")
    return best_knn, y_pred, accuracy


def train_decision_tree(X_train, y_train, X_test, y_test):
    param_grid = {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }
    dt = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_dt = grid_search.best_estimator_
    y_pred = best_dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Decision Tree Best Parameters: {grid_search.best_params_}")
    print(f"Decision Tree Accuracy: {accuracy:.4f}")
    return best_dt, y_pred, accuracy


def train_logistic_regression(X_train, y_train, X_test, y_test):
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs'],
        'max_iter': [1000, 2000]
    }
    lr = LogisticRegression(random_state=42)
    grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_lr = grid_search.best_estimator_
    y_pred = best_lr.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Best Parameters: {grid_search.best_params_}")
    print(f"Logistic Regression Accuracy: {accuracy:.4f}")
    return best_lr, y_pred, accuracy


def evaluate_model(y_test, y_pred, model_name, label_encoder):
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=label_encoder.classes_))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    for i, genre in enumerate(label_encoder.classes_):
        print(f"{genre}: {per_class_accuracy[i]:.4f}")


def cross_validate_models(X_train, y_train):
    models = {
        'k-NN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    results = {}
    for name, model in models.items():
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring='accuracy')
        results[name] = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores
        }
        print(f"{name}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    return results


def main():
    X, y = load_data('Data/features_3_sec.csv')  # or 'features_30_sec.csv'
    X, y = explore_data(X, y)
    pca = analyze_features(X, y)
    X_train, X_test, y_train, y_test, label_encoder, scaler = preprocess_data(
        X, y)
    print("Cross-validation results:")
    cv_results = cross_validate_models(X_train, y_train)

    print("\nTraining and evaluating models:")
    knn_model, knn_pred, knn_acc = train_knn(X_train, y_train, X_test, y_test)
    evaluate_model(y_test, knn_pred, "k-NN", label_encoder)
    dt_model, dt_pred, dt_acc = train_decision_tree(
        X_train, y_train, X_test, y_test)
    evaluate_model(y_test, dt_pred, "Decision Tree", label_encoder)

    lr_model, lr_pred, lr_acc = train_logistic_regression(
        X_train, y_train, X_test, y_test)
    evaluate_model(y_test, lr_pred, "Logistic Regression", label_encoder)

    results_df = pd.DataFrame({
        'Model': ['k-NN', 'Decision Tree', 'Logistic Regression'],
        'Accuracy': [knn_acc, dt_acc, lr_acc]
    })
    print("\nFinal Results Summary:")
    print(results_df.to_string(index=False))
    return {
        'models': {'knn': knn_model, 'dt': dt_model, 'lr': lr_model},
        'results': results_df,
        'preprocessors': {'scaler': scaler, 'label_encoder': label_encoder}
    }


if __name__ == "__main__":
    results = main()
