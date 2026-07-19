import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix



def magnitude_distribution(df):
    plt.figure(figsize=(10,6))

    sns.histplot(
        df['Magnitude'],
        bins=30,
        kde=True
    )

    plt.title("Magnitude Distribution")
    plt.xlabel("Magnitude")
    plt.ylabel("Frequency")

    plt.show()


def earthquake_locations(df):
    plt.figure(figsize=(10,6))

    plt.scatter(
        df['Longitude'],
        df['Latitude'],
        c=df['Magnitude'],
        cmap='hot',
        alpha=0.5
    )

    plt.colorbar(label='Magnitude')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Earthquake Locations")

    plt.show()



def depth_vs_magnitude(df):
    plt.figure(figsize=(10,6))

    sns.scatterplot(
        x='Depth',
        y='Magnitude',
        data=df
    )

    plt.title("Depth vs Magnitude")
    plt.xlabel("Depth")
    plt.ylabel("Magnitude")

    plt.show()



def risk_distribution(df):
    plt.figure(figsize=(6,5))

    sns.countplot(
        x='Risk',
        data=df
    )

    plt.title("Risk Distribution")
    plt.xlabel("Risk")
    plt.ylabel("Count")

    plt.show()


def correlation_heatmap(df):
    plt.figure(figsize=(15,10))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap='coolwarm',
        fmt='.2f'
    )

    plt.title("Correlation Heatmap")

    plt.show()

def feature_importance(model, X):
    plt.figure(figsize=(10,6))

    sns.barplot(
        x=model.feature_importances_,
        y=X.columns
    )

    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")

    plt.show()




def confusion_matrix_plot(y_test, y_pred):

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.show()