import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from typing import List, Tuple, Dict


class ModelTrainer:
    """
    A class to train and evaluate occupancy prediction models.

    Attributes:
        df (pd.DataFrame): The loaded and preprocessed dataset.
        features_with_light (List[str]): Features including the 'Light' sensor.
        features_without_light (List[str]): Features excluding the 'Light' sensor.
    """

    def __init__(self):
        """Initialize the ModelTrainer with empty dataset and predefined feature lists."""
        self.df: pd.DataFrame = pd.DataFrame()
        self.features_with_light: List[str] = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio', 'hour',
                                               'day_of_week']
        self.features_without_light: List[str] = ['Temperature', 'Humidity', 'CO2', 'HumidityRatio', 'hour',
                                                  'day_of_week']

    def load_and_preprocess_data(self, file_paths: List[str]) -> None:
        """
        Load data from CSV files and preprocess it.

        Args:
            file_paths (List[str]): List of paths to the CSV files.

        Raises:
            FileNotFoundError: If any of the specified files are not found.
            ValueError: If the loaded data is empty or invalid.
        """
        try:
            dfs = [pd.read_csv(file, sep=",", parse_dates=['date']) for file in file_paths]
            self.df = pd.concat(dfs, ignore_index=True)
            if self.df.empty:
                raise ValueError("Loaded data is empty.")
            self.df['hour'] = self.df['date'].dt.hour
            self.df['day_of_week'] = self.df['date'].dt.dayofweek
        except FileNotFoundError as e:
            raise FileNotFoundError(f"One or more data files not found. Error: {e}")
        except Exception as e:
            raise ValueError(f"Error in loading or preprocessing data: {e}")

    def plot_correlation_matrix(self) -> None:
        """
        Plot the correlation matrix for all features including occupancy.

        Raises:
            ValueError: If the dataset has not been loaded.
        """
        if self.df.empty:
            raise ValueError("Dataset not loaded. Call load_and_preprocess_data first.")

        corr = self.df[self.features_with_light + ['Occupancy']].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()

    def train_and_evaluate_model(self, features: List[str]) -> Tuple[RandomForestClassifier, StandardScaler]:
        """
        Train and evaluate a Random Forest model using the specified features.

        Args:
            features (List[str]): List of features to use for training.

        Returns:
            Tuple[RandomForestClassifier, StandardScaler]: Trained model and fitted scaler.

        Raises:
            ValueError: If the dataset has not been loaded.
        """
        if self.df.empty:
            raise ValueError("Dataset not loaded. Call load_and_preprocess_data first.")

        X = self.df[features]
        y = self.df['Occupancy']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print("\nClassification Report (Test Data):")
        print(classification_report(y_test, y_test_pred))

        self.plot_feature_importance(model, features)
        self.plot_confusion_matrix(y_test, y_test_pred)
        self.assess_model_fit(train_accuracy, test_accuracy)

        return model, scaler

    def plot_feature_importance(self, model: RandomForestClassifier, features: List[str]) -> None:
        """
        Plot feature importance for the trained model.

        Args:
            model (RandomForestClassifier): Trained Random Forest model.
            features (List[str]): List of features used in the model.
        """
        feature_importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
        feature_importance = feature_importance.sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray) -> None:
        """
        Plot the confusion matrix for the model predictions.

        Args:
            y_true (pd.Series): True labels.
            y_pred (np.ndarray): Predicted labels.
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix (Test Data)')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()

    def assess_model_fit(self, train_accuracy: float, test_accuracy: float) -> None:
        """
        Assess the model fit based on training and test accuracies.

        Args:
            train_accuracy (float): Accuracy on the training set.
            test_accuracy (float): Accuracy on the test set.
        """
        accuracy_diff = train_accuracy - test_accuracy
        print("\nModel Fit Assessment:")
        if accuracy_diff > 0.05:
            print(f"The model shows signs of overfitting. (Difference: {accuracy_diff:.4f})")
        elif accuracy_diff < -0.05:
            print(f"The model shows signs of underfitting. (Difference: {accuracy_diff:.4f})")
        else:
            print(f"The model seems to be well-fitted. (Difference: {accuracy_diff:.4f})")

    def plot_hourly_occupancy(self) -> None:
        """
        Plot the average occupancy by hour of the day.

        Raises:
            ValueError: If the dataset has not been loaded.
        """
        if self.df.empty:
            raise ValueError("Dataset not loaded. Call load_and_preprocess_data first.")

        hourly_occupancy = self.df.groupby('hour')['Occupancy'].mean()
        plt.figure(figsize=(12, 6))
        hourly_occupancy.plot(kind='bar')
        plt.title('Average Occupancy by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Occupancy')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def train_models(self) -> Dict[str, Tuple[RandomForestClassifier, StandardScaler]]:
        """
        Train models with and without the 'Light' feature.

        Returns:
            Dict[str, Tuple[RandomForestClassifier, StandardScaler]]: Dictionary containing trained models and scalers.

        Raises:
            ValueError: If the dataset has not been loaded.
        """
        if self.df.empty:
            raise ValueError("Dataset not loaded. Call load_and_preprocess_data first.")

        models = {}
        print("Model with Light feature:")
        models['with_light'] = self.train_and_evaluate_model(self.features_with_light)

        print("\nModel without Light feature:")
        models['without_light'] = self.train_and_evaluate_model(self.features_without_light)

        return models

    def save_models(self, models: Dict[str, Tuple[RandomForestClassifier, StandardScaler]],
                    models_dir: str = 'models') -> None:
        """
        Save trained models and scalers to files.

        Args:
            models (Dict[str, Tuple[RandomForestClassifier, StandardScaler]]): Dictionary of models and scalers to save.
            models_dir (str): Directory to save the model files.

        Raises:
            IOError: If there's an error saving the model files.
        """
        try:
            os.makedirs(models_dir, exist_ok=True)
            joblib.dump(models['with_light'][0], os.path.join(models_dir, 'model_with_light.joblib'))
            joblib.dump(models['without_light'][0], os.path.join(models_dir, 'model_without_light.joblib'))
            joblib.dump(models['with_light'][1], os.path.join(models_dir, 'scaler_with_light.joblib'))
            joblib.dump(models['without_light'][1], os.path.join(models_dir, 'scaler_without_light.joblib'))
            print(f"Models and scalers saved in {models_dir}")
        except IOError as e:
            raise IOError(f"Error saving model files: {e}")


if __name__ == "__main__":
    trainer = ModelTrainer()

    # Load and preprocess data
    file_paths = [
        'path/to/datatraining.txt',
        'path/to/datatest.txt',
        'path/to/datatest2.txt'
    ]
    trainer.load_and_preprocess_data(file_paths)

    # Plot correlation matrix
    trainer.plot_correlation_matrix()

    # Train models
    models = trainer.train_models()

    # Plot hourly occupancy
    trainer.plot_hourly_occupancy()

    # Save models
    trainer.save_models(models)