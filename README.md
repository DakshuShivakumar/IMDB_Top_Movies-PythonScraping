# IMDB_Top_Movies-PythonScraping
Scrapes IMDB's top 1000 movies of all the times with 6 other fields

**PROJECT : IMDB MOVIE MOSAIC – DETAILS SCRAPER**
---------------------------------------------------

**OBJECTIVE** : 
---------------

        -The project "IMDB Movie Mosaic - Details Scraper" involves collecting comprehensive data about the top movies of all time from the IMDb website.
        -This data includes essential details such as movie titles, release years, durations, certificate ratings, IMDb ratings, vote counts, descriptions, metascores, and URLs. 
        -The objective is to create a rich database that facilitates detailed analyses and visualizations of trends in movie ratings, popularity, quality metrics, and correlations between various movie attributes. This will enable a deeper understanding of cinematic trends and industry standards over time.


**STEP 1 : DATASET**
-----------------------

        1. Title: This is the official name of the movie as listed on IMDb. It's crucial for identifying the film and linking it to other data sources or references.

        2. Release Year: Indicates the year when the movie was first released to the public, helping to categorize films chronologically and analyze trends over different periods.

        3. Duration: Measured in minutes, this field represents the total length of the movie. Duration is useful for exploring correlations between the length of films and their ratings or popularity.

        4. Certificate Ratings: The classification provided by movie rating boards that suggest the suitability of the film for different audiences based on content (e.g., G, PG, PG-13, R, NC-17 in the U.S.). This helps in analyzing viewership demographics and preferences.

        5. IMDb Rating: The average rating given by IMDb users, on a scale from 1 to 10. This is a critical measure of a movie's popularity and critical reception and is often used in predictive modeling and recommendation systems.

        6. Votes: Reflects the total number of IMDb users who have rated the movie. A higher number of votes typically indicates greater viewer engagement and can lend more credibility to the IMDb rating.

        7. Description: A concise synopsis of the movie’s plot, highlighting major themes and key events without giving away spoilers. This textual data is useful for sentiment analysis and natural language processing tasks.

        8. Metascore: A numeric score out of 100, derived from the critical reviews aggregated by Metacritic. This score provides a consensus evaluation from professional critics and is useful for comparisons between audience and critic perceptions.

        9. URL: The web address of the movie’s page on IMDb. It allows direct access to more detailed information and additional data like cast, crew, and user reviews.

STEP 2 : SUMMARY OF THE CODE
--------------------------------

1. Web Scraping Script Overview
        -Imports and Setup: Import necessary libraries (selenium, sqlite3, time, re) and configure Selenium Chrome WebDriver with the specified path to the chromedriver.
        -WebDriver Initialization: Initialize and open IMDb URL for top-rated movies to start data extraction.
2. Database Configuration
        -SQLite Database Setup: Initialize movie_details.db and create a table named movie_details to store movie attributes including title, release year, duration, ratings, and more.
3. Helper Functions
        -title_exists Function: Checks if a movie title already exists in the database to avoid duplications.
4. Data Extraction Function: scrape_page
        -Data Extraction: Extract movie details such as title, year, duration, ratings, votes, description, metascore, and URL using CSS selectors.
5. Error Handling: Manage exceptions for timeouts or missing elements to ensure smooth data extraction.
6. Data Insertion and Navigation
        -Insert Data into Database: Insert movie details into the database after ensuring no duplicates exist.
7. Page Navigation: Navigate through multiple pages by clicking the "100 more" button, using both direct clicks and JavaScript clicks to handle potential intercepts.
8. Script Termination
        -Close Database and WebDriver: Ensure the database connection is closed and WebDriver is quit after completing the scraping.
9. Exception Handling
        -Robust Error Management: Includes handling for common issues like element click interceptions and timeouts, ensuring continued operation even when exceptions occur.


STEP 3 - Data Cleaning
-----------------------

1. Overview
        -Imports and Setup: Imports necessary libraries (sqlite3 for database access and pandas for data manipulation) and establishes a connection to the SQLite database movie_details.db.
2. Data Loading: Executes a SQL query to load movie details into a pandas DataFrame and prints initial dataset information.
3. Data Cleaning Steps
        -Remove Duplicates: Removes duplicate entries based on the 'title' column.
4. Data Type Conversion:
        -Converts 'release_year' to integer.
        -Converts 'imdb_rating' and 'metascore' to float.
        -Cleans and converts 'votes' from string to integer, removing commas.
5. Saving Cleaned Data
        -Export to New Database: Opens a new database connection (cleaned_movie_details.db) and saves the cleaned DataFrame, replacing any existing table.
6. Final Overview: Prints information about the cleaned dataset to verify changes.
7. Finalization
Closure: Closes both the initial and new database connections.

STEP 4 - DATA ANALYTICS AND visualization
------------------------------------------

1 - DESCRIPTIVE ANALYSIS - TREDNS IN MOVIE SCORES, IMDB RATINGS AND METASCORES OVER THE YEARS
2 -  Predictive Analytics intertwined with Text Analytics - Topic Influence with IMDB Rating
3 - Linear Regression Model used for predicting IMDb ratings
4. Sentiment Analysis

LIBRARIES USED IN MAIN SOURCE CODE
-----------------------------------

sqlite3
selenium
webdriver_manager
time
re

IN DATA CLEANING CODE
----------------------

pandas

DATA ANALYTICS WITH visualization
-----------------------------------

numpy 
matplotlib
sklearn
nltk
textblob

CHROMEDRIVER INSTALLATION
-----------------------------
#Replace this with your path where the ChromeDriver executable is present
chromedriver executable download link - https://storage.googleapis.com/chrome-for-testing-public/124.0.6367.155/mac-arm64/chromedriver-mac-arm64.zip
