import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import ast

# In this preprocessing model, we give unknown artists the average popularity, and we average out popularity predictions
# for known artists

###############################     Reading Initial Data    ###################################

# Read in data
data = pd.read_csv("../Datasets/Original/train.csv")
test = pd.read_csv("../Datasets/Original/test.csv")


# Currently, artists column is stored as a string representation of list of artists, convert this to actual list
data['artists'] = data['artists'].map(lambda x: ast.literal_eval(x))
test['artists'] = test['artists'].map(lambda x: ast.literal_eval(x))

##############################      Getting feature and target variables      #########################################

# Get target variables
y = data[["popularity"]]

# Get relevant feature variables
X = data.drop(["id", "release_date", "name"], axis=1)

test = test.drop(["release_date", "name"], axis=1)

###############################             Preprocessing             ########################################

# Dictionaries to store average popularity values for each artist and each year
artistVals = dict()
yearVals = dict()

# Go through each sample
for _, row in X.iterrows():

    # Go through each artist in the current sample
    for artist in row['artists']:

        # If we have come across this artist before, simply update our total and count values
        if artist in artistVals.keys():
            total, count = artistVals[artist]
            artistVals[artist] = (total + int(row['popularity']), count + 1)

        # If we haven't come across this artist before, initalise the total and count values
        else:
            artistVals[artist] = (int(row['popularity']), 1)

    # If we have come across this year before, simply update our total and count values
    if row['year'] in yearVals.keys():
        (total, count) = yearVals[row['year']]
        yearVals[row['year']] = (total + row['popularity'], count + 1)

    # If we haven't come across this year before, initalise the total and count values
    else:
        yearVals[row['year']] = (row['popularity'], 1)

print("Got total popularity points and number of popularity points for each artist and year...")

# To get average popularity of a song
total = 0
countTotal = 0

# Iterate through each artist
for key in artistVals.keys():
    # Calculate the average popularity of that artist
    (tot, count) = artistVals[key]
    artistVals[key] = tot / count

    # Add count of popularity points and count of total number of songs
    total += tot
    countTotal += count

# Get average popularity for a song in the training test
avg = total / countTotal

# Get average popularity for each year
for key in yearVals.keys():
    (tot, count) = yearVals[key]
    yearVals[key] = tot / count

print("Calculated average popularity for artists and total average popularity...")

avgPopularitiesArtist = []
avgPopularitiesYear = []

# Go through each row in training dataset
for _, row in X.iterrows():

    # Initialise total average popularity of artists
    total = 0

    # Get number of artists
    numArtists = len(row['artists'])

    # Add average popularity of each artist to total
    for artist in row['artists']:
        total += artistVals[artist]

    # Calculate average popularity of artists, but if the song has more artists then give it a slight boost
    avgPopularitiesArtist.append((total / numArtists)) # * (1 + (numArtists - 1) / 100))

    # Get average popularity of the year
    avgPopularitiesYear.append(yearVals[row['year']])

# Add average popularity of the artists as a column
X.insert(1, "avgPopularityArtists", avgPopularitiesArtist, None)

# Add average popularity of the year as a column
X.insert(1, "avgPopularityYear", avgPopularitiesYear, None)

print("Created a new column for each song's average popularity by artist and year in train data...")

print("Created a new column for each song's average popularity by artist and year in test data...")

# Get rid of the categorical and target columns
X = X.drop(["artists", "year", "popularity"], axis=1)

avgPopularitiesArtistTest = []
avgPopularitiesYearTest = []

# Go through each row in test dataset
for _, row in test.iterrows():

    # Initialise total average popularity of artists
    total = 0

    # Get number of artists
    numArtists = len(row['artists'])

    # Add average popularity of each artist to total, since this is the test set, this may have artists we haven't come
    # across, in which case just add half the average popularity, since an unknown artist probably isn't very popular
    for artist in row['artists']:
        if artist in artistVals.keys():
            total += artistVals[artist]
        else:
            total += avg

    # Calculate average popularity of artists, but if the song has more artists then give it a slight boost
    avgPopularitiesArtistTest.append((total / numArtists) * (1 + (numArtists - 1) / 50))

    # Get average popularity of the year, as this is a test set this also may have years we haven't come across
    if row['year'] in yearVals.keys():
        avgPopularitiesYearTest.append(yearVals[row['year']])
    else:
        avgPopularitiesYearTest.append(avg * (1 + (row['year'] - 2000) / 200))

# Add average popularity of the artists as a column
test.insert(1, "avgPopularityArtists", avgPopularitiesArtistTest, None)

# Add average popularity of the year as a column
test.insert(1, "avgPopularityYear", avgPopularitiesYearTest, None)

ids = test["id"]

test = test.drop(["artists", "year", "id"], axis=1)


# Standardize our training and test data
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X))

test = pd.DataFrame(scaler.transform(test))
test.insert(0, "id", ids, None)

scalerTarget = StandardScaler()
y = pd.DataFrame(scalerTarget.fit_transform(y))

# VarMean = pd.DataFrame({"var": [scalerTarget.var_], "mean": [scalerTarget.mean_]})
# VarMean.to_csv("varMean.csv")

with open("../Datasets/Processed/FinalDatasets/PreprocessingModel1/varMean.txt", "w+") as f:
    f.write(f"{scalerTarget.var_}, {scalerTarget.mean_}" )

# Save our dataframe for later inspection...
X.to_csv("../Datasets/Processed/FinalDatasets/PreprocessingModel1/X.csv", index=False)
y.to_csv("../Datasets/Processed/FinalDatasets/PreprocessingModel1/y.csv", index=False)

test.to_csv("../Datasets/Processed/FinalDatasets/PreprocessingModel1/test.csv", index=False)