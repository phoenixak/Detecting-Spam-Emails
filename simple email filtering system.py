import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import resample
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def load_and_preprocess_data(filepath):
    """
    Load and preprocess the SMS spam dataset.
    
    Parameters:
    - filepath: Path to the CSV file containing the dataset.
    
    Returns:
    - df: Preprocessed DataFrame ready for analysis.
    """
    # Load the dataset with proper encoding
    df = pd.read_csv(filepath, encoding='latin-1')
    
    # Drop unnecessary columns
    df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
    
    # Encode 'spam' as 1 and 'ham' as 0
    df['v1'] = df["v1"].map({'spam': 1, 'ham': 0})
    
    return df

def visualize_data_distribution(df):
    """
    Visualize the distribution of spam vs. ham messages.
    
    Parameters:
    - df: DataFrame containing the preprocessed data.
    """
    sns.countplot(x='v1', data=df)
    plt.show()

def upsample_minority_class(df):
    """
    Upsample the minority class to balance the dataset.
    
    Parameters:
    - df: DataFrame containing the preprocessed data.
    
    Returns:
    - df_upsampled: DataFrame with balanced classes.
    """
    df_majority = df[df['v1'] == 0]
    df_minority = df[df['v1'] == 1]
    
    df_minority_upsampled = resample(df_minority,
                                     replace=True,
                                     n_samples=len(df_majority),
                                     random_state=42)
    
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    
    return df_upsampled

def vectorize_text(df):
    """
    Convert SMS texts into numerical vectors using CountVectorizer.
    
    Parameters:
    - df: DataFrame containing the preprocessed data.
    
    Returns:
    - X: Vectorized input variables.
    """
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df["v2"])
    
    return X

def train_and_evaluate_model(X, y):
    """
    Train a Gaussian Naive Bayes classifier and evaluate its performance.
    
    Parameters:
    - X: Vectorized input variables.
    - y: Target variable (spam or ham).
    
    Returns:
    - score: Model accuracy score on the test set.
    """
    X_train, X_test, y_train, y_test = train_test_split(X.toarray(), y, test_size=0.3, random_state=0)
    
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    
    score = clf.score(X_test, y_test)
    
    return score

# Main execution flow
filepath = 'Detecting Spam Emails\dataset\spam.csv'
df = load_and_preprocess_data(filepath)
visualize_data_distribution(df)
df_upsampled = upsample_minority_class(df)
X = vectorize_text(df_upsampled)
y = df_upsampled["v1"]
score = train_and_evaluate_model(X, y)

print(f"Model accuracy score: {score}")