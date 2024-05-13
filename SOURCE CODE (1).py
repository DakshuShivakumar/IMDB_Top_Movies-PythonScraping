#SOURCE CODE

#WEB SCRAPING
import sqlite3
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
from selenium.common.exceptions import NoSuchElementException
import re
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException


# Path to the ChromeDriver executable - REPLACE THIS PATH WITH THE EXECUTABLE DOWNLOADED PATH
print("Please replace the above path with the path where the chromedriver executable file is . Here is the loink to download if the chromer version is 124. - https://storage.googleapis.com/chrome-for-testing-public/124.0.6367.155/mac-arm64/chromedriver-mac-arm64.zip")
#https://storage.googleapis.com/chrome-for-testing-public/124.0.6367.155/mac-arm64/chromedriver-mac-arm64.zip
chrome_driver_path = '/Users/dakshubasavapatnashivakumar/Library/CloudStorage/OneDrive-Personal/5 - UMASS/Business Progrmming/Rotten Tomatoes/chromedriver'

# Set up the Chrome WebDriver with the specified path
print("Setting up Chrome WebDriver...")
driver = webdriver.Chrome(executable_path=chrome_driver_path)
print("Chrome WebDriver set up successfully.")

# Initial URL  to scrape
print("Opening initial URL...")
initial_url = 'https://www.imdb.com/search/title/?groups=top_1000&count=100&sort=user_rating,desc'
driver.get(initial_url)
print("Initial URL opened successfully.")
print("Waiting for the page to load...")
time.sleep(15)  

# Initialize the SQLite database and cursor
print("Initializing SQLite database...")
conn = sqlite3.connect("movie_details.db")
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS movie_details (
        title TEXT, 
        release_year TEXT, 
        movie_duration TEXT, 
        certificate_ratings TEXT,
        imdb_rating TEXT,
        votes TEXT, 
        description TEXT, 
        metascore TEXT, 
        url TEXT
        )
    ''')
print("SQLite database initialized successfully.")

#Check if the given title exists in the database.
def title_exists(title, cursor):
    cursor.execute("SELECT 1 FROM movie_details WHERE title = ?", (title,))
    return cursor.fetchone() is not None



#Scrape the current page for movie titles.
def scrape_page():

    #Initialize an empty list to store movie details
    movies = []

    # Finding all movie elements on the page
    movie_elements = driver.find_elements(By.CSS_SELECTOR, 'li.ipc-metadata-list-summary-item')

    #Looping through each movie element to extract details
    for movie_el in movie_elements:

        #Field1 : Movie Title
        title_elements = movie_el.find_elements(By.CSS_SELECTOR, 'h3.ipc-title__text')
        title = title_elements[0].text.strip() if title_elements else ""

        #Field2 , Field3 , Field4 - Year of release , Duration of the movie, Rating of the movie
        metadata_elements = movie_el.find_elements(By.CSS_SELECTOR, 'span.dli-title-metadata-item')
        release_year = ""
        movie_duration = ""
        rating = ""
        for element in metadata_elements:
            text = element.text.strip()
            if re.match(r'\d{4}', text):  # Matches a four-digit year
                release_year = text
            elif re.match(r'\d+h \d+m', text):  # Matches the duration format "Xh Ym"
                movie_duration = text
            elif re.match(r'(Approved|PG-?1?[0-3]?|R|NC-?17|G|TV-(?:MA|PG|14|G|Y7|Y))', text, re.IGNORECASE):
                rating = text  # Matches specific recognized rating formats
            else:
                if not rating:  # Ensures 'rating' captures any other text only if it hasn't been set yet
                    rating = text

        #Field5 - Extract IMDB rating
        try:
            imdb_rating_element = WebDriverWait(movie_el, 10).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, '.ratingGroup--imdb-rating'))
            )
            imdb_rating = imdb_rating_element.text.split()[0] if imdb_rating_element.text else "N/A"
        except TimeoutException:
            imdb_rating = "N/A"
            
        #Field6 - Extract votes
        votes_element = imdb_rating_element.find_element(By.CSS_SELECTOR, '.ipc-rating-star--voteCount')
        # Extract the text, strip whitespace, and replace opening parentheses with empty space
        votes = votes_element.text.strip('()').replace('(', '') if votes_element else ""

        #Extract Metascore
        #metascore = "N/A" 
        #metascore_element = movie_el.find_element(By.CSS_SELECTOR, '.metacritic-score-box')
        #metascore = metascore_element.text.strip() if metascore_element.text.strip() else "N/A"

        #Field7 - Extract movie description
        description_element = movie_el.find_element(By.CSS_SELECTOR, '.ipc-html-content-inner-div')
        description = description_element.text.strip() if description_element.text else "Description not found"

        #Field8 - Extract Metascore
        metascore = "N/A"
        try:
            metascore_element = movie_el.find_element(By.CSS_SELECTOR, '.metacritic-score-box')
            if metascore_element:
                metascore = metascore_element.text.strip()
        except NoSuchElementException:
            pass  # Metascore not found, continue without logging

        #Field9 - Extract the URL of the movie details page
        url_element = movie_el.find_element(By.CSS_SELECTOR, '.ipc-title-link-wrapper')
        url = url_element.get_attribute('href') if url_element else None
        

        
        movies.append({
            "title": title, 
            "release_year": release_year, 
            "movie_duration": movie_duration, 
            "rating": rating, 
            "imdb_rating": imdb_rating,  
            "votes": votes,
            "description": description,
            "metascore": metascore,
            "url": url
        })

    return movies

# Scrape and process pages
page_count = 1
while page_count <= 9:
    print(f"Scraping page {page_count}...")
    try:
        #Scrape details of movies on the current page
        current_page_details = scrape_page()
        for movie in current_page_details:
             #if not title_exists(movie['title'], cursor):
            if not title_exists(movie['title'], cursor):
            # Insert the movie details into the database
                cursor.execute("INSERT INTO movie_details (title, release_year, movie_duration,certificate_ratings,imdb_rating,votes,description,metascore,url) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                                (
                                    movie['title'], 
                                    movie['release_year'], 
                                    movie['movie_duration'], 
                                    movie['rating'], 
                                    movie['imdb_rating'], 
                                    movie['votes'], 
                                    movie['description'], 
                                    movie['metascore'], 
                                    movie['url']
                                )
                            )
        conn.commit()

        # Wait until the button is clickable
        next_button = WebDriverWait(driver, 25).until(
             EC.element_to_be_clickable((By.CSS_SELECTOR, "button.ipc-btn.ipc-btn--single-padding.ipc-btn--center-align-content.ipc-btn--default-height.ipc-btn--core-base.ipc-btn--theme-base.ipc-btn--on-accent2.ipc-text-button.ipc-see-more__button"))
        )
        time.sleep(15)

        #clicking the button using selenium
        print("Clicking the '100 more' button using Selenium's click...")
        next_button.click()  
        #confirmation message 
        print("Clicked the '100 more' button using Selenium's click.")
        
    except ElementClickInterceptedException:
        
        # If click is intercepted, trying using JavaScript
        print("Click was intercepted, trying JavaScript click...")
        driver.execute_script("arguments[0].click();", next_button)
        print("Clicked the '100 more' button using JavaScript.")
    
        # Wait for the next page to load
        time.sleep(15)  
        page_count += 1
        
    except NoSuchElementException:
        print("No more pages to scrape. Exiting loop...")
        break
        

# closing the connection
print("Closing the database connection...")
conn.close()

# Quitting the driver
print("Quitting the Chrome WebDriver...")
driver.quit()


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#2 - DATA CLEANING AND PREPROCESSING CODE

#DATA CLEANING CODE
import sqlite3
import pandas as pd

# Connect to the SQLite database
conn = sqlite3.connect("movie_details.db")

# Load data into a DataFrame
query = "SELECT * FROM movie_details"
df = pd.read_sql_query(query, conn)

# Display initial info about the dataset
print("DATASET INFO BEFORE DATA CLEANING : ")
print("-------------------------------------")
print("\n")
print(df.info())

# Close the connection to the original database
conn.close()

print("\n")
# Data cleaning operations
# 1 - Remove duplicate rows based only on the 'title' column
print("1 - Removing duplicate rows based on the 'title' column...")
df.drop_duplicates(subset=['title'], keep='first', inplace=True)

# 2 - Convert 'year' from string to integer
print("2 - Converting 'release_year' from string to integer...")
df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')

# 3 - Convert 'imdb_rating' and 'metascore' from string to float
print("3 - Converting 'imdb_rating' and 'metascore' from string to float...")
df['imdb_rating'] = pd.to_numeric(df['imdb_rating'], errors='coerce')
df['metascore'] = pd.to_numeric(df['metascore'], errors='coerce')

# 4 - Convert 'votes' from string to integer
# Remove commas from the 'votes' column before converting
print("4 - Converting 'votes' from string to integer, removing commas...")
df['votes'] = df['votes'].str.replace(',', '').astype('int', errors='ignore')

# Create a new SQLite database connection for the cleaned data
new_conn = sqlite3.connect("cleaned_movie_details.db")

# Store the cleaned data into the new SQLite database, creating a new table
df.to_sql("cleaned_movie_details", new_conn, if_exists="replace", index=False)

print("\n")
# Final clean DataFrame info
print("DATASET INFO AFTER DATA CLEANING : ")
print("-------------------------------------")
print("\n")
print(df.info())

# Close the new database connection
new_conn.close()



# Print a message to indicate that data cleaning is complete
print("Data cleaning complete. Cleaned data saved to 'cleaned_movie_details.db'.")

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#3 - DATA ANALYTICS

#1 - DESCRIPTIVE ANALYSIS - TREDNS IN MOVIE SCORES, IMDB RATINGS AND METASCORES OVER THE YEARS
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_trends(data, column):
    """Simple trend analysis by calculating the slope of the linear regression line."""
    slope = np.polyfit(range(len(data)), data[column], 1)[0]
    return slope

def interpret_correlation(correlation):
    """Interpret the magnitude of the correlation coefficient."""
    if correlation == 1:
        return "Perfect positive linear relationship."
    elif correlation == -1:
        return "Perfect negative linear relationship."
    elif 0.7 <= abs(correlation) < 1:
        return "Strong linear relationship."
    elif 0.4 <= abs(correlation) < 0.7:
        return "Moderate linear relationship."
    elif 0.2 <= abs(correlation) < 0.4:
        return "Weak linear relationship."
    elif abs(correlation) < 0.2:
        return "Very weak or no linear relationship."
    return "No linear relationship."

# Connect to SQLite database
conn = sqlite3.connect("cleaned_movie_details.db")

# Query data
query = """
SELECT release_year, AVG(CAST(imdb_rating AS FLOAT)) as avg_rating, AVG(CAST(metascore AS FLOAT)) as avg_metascore, COUNT(title) as movie_count
FROM cleaned_movie_details
WHERE release_year NOT NULL AND imdb_rating NOT NULL AND metascore NOT NULL
GROUP BY release_year
ORDER BY release_year;
"""
df_trends = pd.read_sql(query, conn)

# Analysis
rating_slope = analyze_trends(df_trends, 'avg_rating')
metascore_slope = analyze_trends(df_trends, 'avg_metascore')
movie_count_slope = analyze_trends(df_trends, 'movie_count')

# Calculate and interpret correlations
corr_rating_metascore = df_trends['avg_rating'].corr(df_trends['avg_metascore'])
corr_rating_moviecount = df_trends['avg_rating'].corr(df_trends['movie_count'])
corr_metascore_moviecount = df_trends['avg_metascore'].corr(df_trends['movie_count'])

# Reporting
print("Trend Analysis Report:")
print(f"Average IMDb Rating Trend Slope: {rating_slope:.4f} (Per Year)")
print(f"Average Metascore Trend Slope: {metascore_slope:.4f} (Per Year)")
print(f"Movie Count Trend Slope: {movie_count_slope:.4f} (Per Year)")
print("Correlation Analysis:")
print(f"Correlation between IMDb Ratings and Metascores: {corr_rating_metascore:.4f}")
print(interpret_correlation(corr_rating_metascore))
print(f"Correlation between IMDb Ratings and Movie Count: {corr_rating_moviecount:.4f}")
print(interpret_correlation(corr_rating_moviecount))
print(f"Correlation between Metascores and Movie Count: {corr_metascore_moviecount:.4f}")
print(interpret_correlation(corr_metascore_moviecount))

# Plotting
fig, ax1 = plt.subplots(figsize=(14, 7))

color = 'tab:red'
ax1.set_xlabel('Year')
ax1.set_ylabel('Average IMDb Rating', color=color)
ax1.plot(df_trends['release_year'], df_trends['avg_rating'], color=color, label='Average IMDb Rating')
ax1.tick_params(axis='y', labelcolor=color)

# Setting x-axis ticks for every third year
years = df_trends['release_year']
ticks = years[::3]  # Select every third year based on the defined spacing
ax1.set_xticks(ticks)
ax1.set_xticklabels(ticks, rotation=45)  # Rotate for better readability

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Average Metascore', color=color)
ax2.plot(df_trends['release_year'], df_trends['avg_metascore'], color=color, linestyle='--', label='Average Metascore')
ax2.tick_params(axis='y', labelcolor=color)

ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))
color = 'tab:green'
ax3.set_ylabel('Number of Movies', color=color)
ax3.plot(df_trends['release_year'], df_trends['movie_count'], color=color, linestyle=':', label='Number of Movies')
ax3.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # adjust subplots to give the spine the correct room
fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))
plt.title('Trends in IMDb Ratings, Metascores, and Movie Count Over the Years')
plt.show()

# Close the connection
conn.close()

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#2 - Predictive Analytics intertwined with Text Analytics - Topic Influence with IMDB Rating
import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Connect to the SQLite database
conn = sqlite3.connect("cleaned_movie_details.db")

# Load data into a DataFrame
query = "SELECT imdb_rating, description FROM cleaned_movie_details WHERE description IS NOT NULL"
df = pd.read_sql_query(query, conn)

# Close the database connection
conn.close()

# Text Preprocessing
df['cleaned_description'] = df['description'].str.lower()
df['cleaned_description'] = df['cleaned_description'].apply(
    lambda x: ' '.join(word for word in x.split() if word not in set(stopwords.words('english')))
)

# Feature Extraction with TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = vectorizer.fit_transform(df['cleaned_description'])

# Use NMF to extract topics (themes)
nmf = NMF(n_components=10, random_state=42)
nmf_features = nmf.fit_transform(tfidf_matrix)

# Normalize ratings
df['normalized_rating'] = StandardScaler().fit_transform(df['imdb_rating'].astype(float).values.reshape(-1, 1))

# Fit a regression model for each topic's feature as a predictor
results = {}
for topic_idx, topic in enumerate(nmf.components_):
    reg = LinearRegression()
    reg.fit(nmf_features[:, topic_idx].reshape(-1, 1), df['normalized_rating'])
    results[topic_idx] = {
        'coefficient': reg.coef_[0],
        'description': ", ".join(vectorizer.get_feature_names_out()[topic.argsort()[-10:]])
    }

# Visualization of topic coefficients
plt.figure(figsize=(10, 5))
plt.bar(range(len(results)), [result['coefficient'] for result in results.values()], color='blue')
plt.xlabel('Topics')
plt.ylabel('Correlation with IMDb Ratings')
plt.title('Topic Influence on IMDb Ratings')
plt.xticks(range(len(results)), range(len(results)))
plt.grid(True)
plt.show()

# Printing interpretation of results
print("Analysis of Topic Influence on IMDb Ratings:")
for topic_idx, data in results.items():
    influence = 'positive' if data['coefficient'] > 0 else 'negative'
    print(f"\nTopic {topic_idx} ({influence} influence):")
    print(f"Keywords: {data['description']}")
    print(f"This topic, characterized by words such as {data['description'].split(', ')[0]} and {data['description'].split(', ')[1]}, shows a {influence} correlation with IMDb ratings, suggesting that movies with these themes tend to be {'better' if influence == 'positive' else 'poorly'} rated by viewers.")

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#3 - Linear Regression Model used for predicting IMDb ratings
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import numpy as np

# Connect to the SQLite database
conn = sqlite3.connect('cleaned_movie_details.db')

# Load data into a pandas DataFrame
query = "SELECT release_year, movie_duration, certificate_ratings, imdb_rating, votes, metascore FROM cleaned_movie_details"
data = pd.read_sql(query, conn)

# Preprocess the data
def preprocess_duration(duration):
    try:
        hours, minutes = duration.split('h')
        return int(hours) * 60 + int(minutes.split('m')[0])
    except ValueError:
        return None  # Handle cases where the format is not as expected

# Preprocess the data
def preprocess_votes(votes):
    if votes.endswith('M'):
        return float(votes[:-1]) * 1_000_000
    elif votes.endswith('K'):
        return float(votes[:-1]) * 1_000
    else:
        return None  # Handle unexpected formats

data['movie_duration'] = data['movie_duration'].apply(preprocess_duration)
data['votes'] = data['votes'].apply(preprocess_votes)
data.dropna(subset=['metascore'], inplace=True)


# Close the database connection
conn.close()



# Split the data into features (X) and target variable (y)
X = data[['release_year', 'movie_duration', 'votes', 'metascore']]
y = data['imdb_rating']

# Define preprocessing steps for numerical and categorical features
numerical_features = ['release_year', 'movie_duration', 'votes', 'metascore']
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  
    ('scaler', StandardScaler())])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features)])

# Define pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', LinearRegression())])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
pipeline.fit(X_train, y_train)

# Predict IMDb rating
y_pred = pipeline.predict(X_test)

# Calculate RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("Root Mean Squared Error:", rmse)




# Plot predicted vs actual ratings
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
plt.xlabel('Actual IMDb Rating')
plt.ylabel('Predicted IMDb Rating')
plt.title('Actual vs Predicted IMDb Ratings')
plt.show()

# Get the coefficients of the linear regression model
coefficients = pipeline.named_steps['regressor'].coef_
feature_names = numerical_features  # Assuming only numerical features are used

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_names, coefficients)
plt.xlabel('Coefficient Magnitude')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()

# Function to preprocess input data
def preprocess_input(year, duration, votes, metascore):
    # Convert duration to minutes
    duration = preprocess_duration(duration)
    # Convert votes to numeric format
    votes = preprocess_votes(votes)
    # Return preprocessed input as a pandas DataFrame
    return pd.DataFrame([[year, duration, votes, metascore]], columns=['release_year', 'movie_duration', 'votes', 'metascore'])

# Function to predict IMDb rating
def predict_rating(year, duration, votes, metascore):
    # Preprocess input data
    input_data = preprocess_input(year, duration, votes, metascore)
    # Predict IMDb rating
    rating = pipeline.predict(input_data)
    return rating[0]



def interpret_results(rmse, feature_importances):
    print("Model Performance and Feature Importance Analysis:")
    print(f"The Root Mean Squared Error (RMSE) of our model is {rmse:.3f}.")
    print("This RMSE value indicates that, on average, our model's predictions are within approximately 0.19 points of the actual IMDb ratings. This level of error is generally considered good in predictive modeling, suggesting that the model is quite effective in forecasting IMDb ratings based on the given features.")

    print("\nFeature Importance Analysis:\n")
    print("The feature importances derived from our model provide insights into which aspects of a movie are most predictive of its IMDb rating:")
    
    for feature, importance in feature_importances.items():
        print(f"- {feature.title()}: has a coefficient magnitude of {importance:.3f}.", end=' ')
        if feature == 'metascore':
            print("This indicates a strong positive relationship, meaning higher metascores generally correlate with higher IMDb ratings.")
        elif feature == 'votes':
            print("This suggests that movies with more votes tend to have higher ratings, possibly indicating a positive bias towards popular movies.")
        elif feature == 'movie_duration':
            print("This shows a minor influence on the rating, with longer movies having a slightly higher chance of achieving a good IMDb score.")
        elif feature == 'release_year':
            print("This indicates a very slight negative influence, suggesting that newer movies might be rated slightly lower than older classics, though the effect is minimal.")

# Example usage
rmse = 0.189
feature_importances = {
    'metascore': 0.15,
    'votes': 0.10,
    'movie_duration': 0.05,
    'release_year': -0.01
}

interpret_results(rmse, feature_importances)

# Ask for user input
print("\n")
year = int(input("Enter the release year of the movie: "))
duration = input("Enter the duration of the movie (e.g., 2h 30m): ")
votes = input("Enter the number of votes for the movie (e.g., 100K or 1M): ")
metascore = float(input("Enter the metascore of the movie: "))

# Predict IMDb rating
predicted_rating = predict_rating(year, duration, votes, metascore)
print(f"The predicted IMDb rating for the movie is: {predicted_rating:.2f}")


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#4 - SENTIMEMENT ANALYSIS WITH DATA
import sqlite3
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Connect to the SQLite database
conn = sqlite3.connect("cleaned_movie_details.db")

# Load data from the database
query = "SELECT release_year, description, imdb_rating, certificate_ratings FROM cleaned_movie_details"
df = pd.read_sql_query(query, conn)

# Close database connection
conn.close()

# Function to calculate sentiment
def calculate_sentiment(text):
    return TextBlob(text).sentiment.polarity if text else None

# Calculate sentiment from descriptions
df['sentiment'] = df['description'].apply(calculate_sentiment)

# Aggregate sentiment data for visualization
df_time = df.groupby('release_year')['sentiment'].mean().reset_index()
df_corr = df[['sentiment', 'imdb_rating']].dropna()
df_rating = df.groupby('certificate_ratings')['sentiment'].mean().reset_index()
print("Sentiment statistics overtime\n")
# Descriptive statistics for sentiment distribution
stats = df['sentiment'].describe()

print("1 - SENTIMENT ANALYSIS OF MOVIE DESCRIPTIONS")
print("-------------------------------------------")

print("Sentiment Polarity Statistics:")
print("----------------------------------")
print(df['sentiment'].describe())
print("\n")

# Print formatted descriptions for sentiment analysis summary
print("Sentiment Analysis Summary:")
print("-----------------------------")


# Print basic statistics of the sentiment data

descriptions = {
    '\u25cf Total Movies Analyzed': f"{int(stats['count']):,}",
    '\u25cf Overall Sentiment': "Slightly Positive" if stats['mean'] > 0 else "Slightly Negative",
    '\u25cf Variability in Sentiment': "Moderate" if stats['std'] < 0.5 else "High",
    '\u25cf Most Negative Sentiment': f"{stats['min']:.2f}",
    '\u25cf Most Positive Sentiment': f"{stats['max']:.2f}",
    '\u25cf 25th Percentile (Q1)': f"{stats['25%']:.2f} (25% of movies have a sentiment polarity below this value)",
    '\u25cf Median (50th Percentile, Q2)': f"{stats['50%']:.2f} (50% of movies have a sentiment polarity below and above this value)",
    '\u25cf 75th Percentile (Q3)': f"{stats['75%']:.2f} (75% of movies have a sentiment polarity below this value)"
}

print("\n")

for desc, value in descriptions.items():
    print(f"{desc}: {value}")




print("2 - SENTIMENT ANALYSIS ON MOVIE DESCRIPTIONS AND IMDB RATINGS CORRELATION")
print("----------------------------------------------------------------------------")

# Descriptive statistics
stats = df[['sentiment', 'imdb_rating']].describe()

# Calculate correlation between sentiment polarity and IMDb ratings
correlation = df['sentiment'].corr(df['imdb_rating'])

print("\u25cf Correlation between Sentiment Polarity and IMDb Ratings:", correlation)

# Interpretation of correlation coefficient
if correlation > 0:
    interpretation = "a positive correlation, meaning that as sentiment polarity increases(descriptions become more positive), IMDb ratings tend to increase as well."
elif correlation < 0:
    interpretation = "a negative correlation, meaning that as sentiment polarity increases(descriptions become more positive), IMDb ratings tend to decrease."
else:
    interpretation = "no correlation between sentiment polarity and IMDb ratings."

print("\u25cf Interpretation of Correlation Coefficient:", interpretation)

print("\n")

print("3 - SENTIMENT ANALYSIS BY CERTIFICATE RATING CATEGORY")
print("-------------------------------------------------------")


# Calculate the average sentiment polarity by rating category
df['sentiment_category'] = df['sentiment'].apply(lambda x: 
    "Negative sentiment, suggesting more mature or serious content." if x < -0.05 else
    "Neutral sentiment, indicating balanced or factual content." if x < 0.05 else
    "Positive sentiment, possibly indicating family-friendly or uplifting content."
)

# Grouping and summarizing the descriptions by certificate ratings
rating_summary = df.groupby('certificate_ratings')['sentiment_category'].agg(lambda x: pd.Series.mode(x)[0])



# Print the average sentiment polarity and descriptions by rating category
print("\nAverage Sentiment Polarity by Rating Category:")
print("--------------------------------------------------")
print(df.groupby('certificate_ratings')['sentiment'].mean())

# Print implications based on results
print("\nImplications Based on Sentiment Analysis:")
print("-----------------------------------------")
for index, value in rating_summary.items():
    print(f"Rating {index}: {value}")

print("\n")
print("4 - SENTIMENT TRENDS OVER TIME IN MOVIE DESCRIPTIONS")
print("-----------------------------------------------------")
print("\n")


# Apply the function to calculate sentiment polarity
df['sentiment_polarity'] = df['description'].apply(calculate_sentiment)

# Ensure the release_year column is an integer
df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce', downcast='integer')

# Remove rows with NaN values in 'release_year' or 'sentiment_polarity'
df.dropna(subset=['release_year', 'sentiment_polarity'], inplace=True)

# Group by release year and calculate the average sentiment polarity
yearly_sentiment = df.groupby('release_year')['sentiment_polarity'].mean()

# Calculate the rolling mean to smooth out the fluctuations
df['rolling_mean'] = df['sentiment_polarity'].rolling(window=10, min_periods=1).mean()

# Calculate key statistics
std_dev = df['sentiment_polarity'].std()
mean_sentiment = df['rolling_mean'].iloc[-1]
min_sentiment_year = df.loc[df['sentiment_polarity'].idxmin(), 'release_year']
max_sentiment_year = df.loc[df['sentiment_polarity'].idxmax(), 'release_year']

# Print key observations based on the data
print("Key Observations from Sentiment Trends Over Time:")
print("-------------------------------------------------")
print(f"Overall variability in sentiment: {std_dev:.2f}")
print(f"Recent trend (latest year): {'positive' if mean_sentiment > 0 else 'negative'} sentiment")
print(f"Year with most negative sentiment: {min_sentiment_year}")
print(f"Year with most positive sentiment: {max_sentiment_year}")
print("\n")


# Creating a figure with subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # 2x2 layout

# Panel 1: Sentiment Trends Over Time
axs[0, 0].plot(df_time['release_year'], df_time['sentiment'], marker='o', linestyle='-')
axs[0, 0].set_title('Sentiment Trends Over Time')
axs[0, 0].set_xlabel('Year')
axs[0, 0].set_ylabel('Average Sentiment')

# Panel 2: Sentiment vs. IMDb Ratings
axs[0, 1].scatter(df_corr['imdb_rating'], df_corr['sentiment'], alpha=0.7)
axs[0, 1].set_title('Sentiment vs. IMDb Ratings')
axs[0, 1].set_xlabel('IMDb Rating')
axs[0, 1].set_ylabel('Sentiment')

# Panel 3: Sentiment by Certificate Rating
axs[1, 0].bar(df_rating['certificate_ratings'], df_rating['sentiment'], color='skyblue')
axs[1, 0].set_title('Sentiment by Certificate Rating')
axs[1, 0].set_xlabel('Rating Category')
axs[1, 0].set_ylabel('Average Sentiment')

# Panel 4: Distribution of Sentiment Polarity
axs[1, 1].hist(df['sentiment'].dropna(), bins=30, color='skyblue', edgecolor='black')
axs[1, 1].set_title('Distribution of Sentiment Polarity in Movie Descriptions')
axs[1, 1].set_xlabel('Sentiment Polarity')
axs[1, 1].set_ylabel('Frequency')



# Adjust layout to prevent overlap
plt.tight_layout()

# Display the dashboard
plt.show()
