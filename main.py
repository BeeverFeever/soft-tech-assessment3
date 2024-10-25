# Author: u3280803 + u3285396
# Assessment 3
# Due: 25/10/2024

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import r2_score

import joblib

@st.cache_resource
def train_models(X_train, Y_train):
    models = {
        "linear regression": LinearRegression(),
        "decision tree": DecisionTreeRegressor(),
        "random forest": RandomForestRegressor(),
        "adaboost": AdaBoostRegressor(),
        "k-nearest": KNeighborsRegressor(),
    }
    already_trained = {}
    for name, model in models.items():
        model.fit(X_train, Y_train)
        st.write(f"Model {name} has been trained.")
        already_trained[name] = model
    return already_trained


@st.cache_data
def transform_scaler(train, test):
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(train))
    test_scaled = pd.DataFrame(scaler.transform(test))
    return scaler, train_scaled, test_scaled

@st.cache_data
def transform_ohe(test, train):
    ohe_transformer = OneHotEncoder(sparse_output = False, dtype = "float")
    train_encoded = pd.DataFrame(ohe_transformer.fit_transform(train))
    test_encoded = pd.DataFrame(ohe_transformer.transform(test))
    train_encoded.columns = ohe_transformer.get_feature_names_out()
    test_encoded.columns = ohe_transformer.get_feature_names_out()
    return ohe_transformer, test_encoded, train_encoded


def main():
    st.title("Assessment 3 - Predicting flight prices")
    # ======================================================================
    # Step 1 - read in the data and get rid of duplicates
    df = pd.read_csv('Clean_Dataset.csv')
    df = pd.DataFrame.drop_duplicates(df)
    df = df.drop(columns = ["Unnamed: 0"])

    st.header("Step 1 - Read in raw data")
    st.dataframe(df)

    st.write(f"number of flights: {df["flight"].value_counts()}")

    st.subheader("Observations")
    st.write("Uhhhh what is there to observe? I mean there wasn't any duplicates, it was a clean dataset so yeah there's that.")

    # ======================================================================
    # Step 2 - problem statement
    st.divider()
    # Prediction - flight prices
    # independent variables - airline, flight, departure_time, stops, class, days_left
    # dependent variable - price
    st.header("Step 2 - Problem Statement")
    # df.drop(df[df["class"] == "Business"].index, inplace = True)
    st.write(f"**Shape**: {df.shape}")
    st.write(f"**Size**: {df.size}")
    st.write("The distribution of the data is generally balanced, and there is a steady, slight positive skew. It displays a consistent baseline with sporadic higher outliers and regular upward spikes, keeping this pattern consistent throughout the timeline. The dataset contained both business and economy class flights, we decided to get rid of the business class flights and focus only on economy. The business flights were significantly more expensive which would have caused false outcomes and wrong predictions later on.")

    st.dataframe(df)

    st.subheader("Observations")
    st.write("The data set is pretty big, with just over 300,000 data rows. Our dependent or predictor variable is going to be price while the rest are independent variables.")

    # ======================================================================
    # Step 3 - Visualising distribution of target variable
    st.divider()
    st.header("Step 3 - Visualising target variable")

    fig, ax = plt.subplots()
    ax.hist(df["price"], bins = "auto")
    ax.set_title("Distribution of Price")
    ax.set_yscale("log")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Price")
    st.pyplot(fig)

    st.subheader("Observations")
    st.write("The frequency of price has a large positive skew. There is so many entries for the lower prices that a log scale for the frequency is required to visualise how many entries there are in the higher prices.")

    # ======================================================================
    # Step 4 - Basic data exploration
    st.divider()
    st.header("Step 4 - Basic data exploration")

    st.subheader("Dataset summary")
    st.dataframe(df.describe())


    st.subheader("Dataset after removing unwanted columns")
    st.write("**Unwanted columns**")
    st.markdown("""
                - flight
                - arrival_time
                - class (we are only looking at economy)""")
    df = df.drop(columns = ["flight", "arrival_time", "class"])
    st.write(f"Shape: {df.shape}")
    st.dataframe(df)

    st.dataframe(pd.DataFrame(df.dtypes, columns = ["Data Type"]))

    st.subheader("Observations")
    st.write("Lots of continuous data columns, we observed that the shear number of different flights were useless and arrival time/duration were dependent after the fact.")

    # ======================================================================
    # Step 5 - EDA of data
    st.divider()
    st.header("Step 5 - Visual EDA - Histograms and Bar graphs")

    # Categorical variables
    st.subheader("Number of flights per Airline")
    st.bar_chart(df["airline"].value_counts(), horizontal = False, x_label = "Airline", y_label = "Amount of flights")

    st.subheader("Number of flights per Departure Time")
    st.bar_chart(df["departure_time"].value_counts(), horizontal = False, x_label = "Departure Time", y_label = "Amount of flights")

    st.subheader("Number of flights per number of Stops")
    st.bar_chart(df["stops"].value_counts(), horizontal = False, x_label = "Stops", y_label = "Amount of flights")

    st.subheader("Number of flights per Source")
    st.bar_chart(df["source_city"].value_counts(), horizontal = False, x_label = "source_city", y_label = "Amount of flights")

    st.subheader("Number of flights per Destination")
    st.bar_chart(df["destination_city"].value_counts(), horizontal = False, x_label = "destination_city", y_label = "Amount of flights")

    # Continuous variables
    fig, ax = plt.subplots(nrows = 2)
    ax[0].hist(df["duration"], bins = "auto")
    ax[0].set_title("Duration of flights")
    ax[0].set_ylabel("Amount")
    ax[1].set_xlabel("Duration")
    sns.boxplot(x = df["duration"], ax = ax[1])
    st.pyplot(fig)

    fig, ax = plt.subplots(nrows = 2)
    ax[0].hist(df["days_left"], bins = "auto")
    ax[0].set_title("Number of days left to buy flights")
    ax[0].set_ylabel("Amount")
    ax[1].set_xlabel("Days Left")
    sns.boxplot(x = df["days_left"], ax = ax[1])
    st.pyplot(fig)

    fig, ax = plt.subplots(nrows = 2)
    ax[0].hist(df["price"], bins = "auto")
    ax[0].set_title("Number of flights per Price")
    ax[0].set_ylabel("Amount")
    ax[1].set_xlabel("Price")
    sns.boxplot(x = df['price'], ax = ax[1])
    st.pyplot(fig)

    st.subheader("Observations")
    st.markdown("""
                - The two most popular airlines also had the most expensive flights.
                - Each city has roughly the same amount of mentions for destination and source. This makes sense as a plane has to leave an airport that it arrives at so there is roughly a 1 for 1 source and destination for each airport.
                - There was a really high spike in frequency for days left to buy tickets once the value got to about 48. I'm assuming this is people buying tickets as soon as they get released.
             """)

    # ======================================================================
    # Step 6 - Removal of outliers
    st.divider()
    st.header("Step 6 - Outlier Analysis")

    continuous_attributes = df[["duration", "days_left", "price"]]

    st.write("### Number of outliers")
    outliers_count = pd.DataFrame()

    for attr in continuous_attributes:
        Q1 = continuous_attributes[attr].quantile(0.25)
        Q3 = continuous_attributes[attr].quantile(0.75)
        IQR = Q3 - Q1

        outliers = (continuous_attributes[attr] < (Q1 - 1.5 * IQR)) | (continuous_attributes[attr] > (Q3 + 1.5 * IQR))
        outliers_count.at[0, attr] = outliers.sum()

        df = df[(continuous_attributes[attr] >= (Q1 - 1.5 * IQR)) & (continuous_attributes[attr] <= (Q3 + 1.5 * IQR))]

    outliers_count = outliers_count.T
    outliers_count = outliers_count.rename(columns = {0: "Count"})
    st.dataframe(outliers_count)

    st.write("### Dataset after removing outliers")
    st.write(df)

    st.subheader("Observations")
    st.write("There was only a small amount of outliers as the frequency of values for each of the variables drops off significantly.")

    # ======================================================================
    # Step 7 - Missing Value Analysis
    st.divider()
    st.header("Step 7 - Missing Value Analysis")

    missing_value_count = df.isna().sum()
    st.dataframe(missing_value_count, column_config={"0": "Missing values count"})

    st.subheader("Observations")
    st.write("There is no missing values.")

    # ======================================================================
    # Step 8 - Feature Selection
    st.divider()
    st.header("Step 8 - Feature Selection")

    # continuous correlations
    st.write("### Continuous Correlation")
    fig, ax = plt.subplots()
    corr, _ = stats.pearsonr(df["duration"], df["price"])
    st.write(f"Correlation: {corr}")
    ax.scatter(df["duration"], df["price"])
    ax.set_xlabel("Duration")
    ax.set_ylabel("Price")
    ax.set_title("Correlation between duration and price")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    corr, _ = stats.pearsonr(df["days_left"], df["price"])
    st.write(f"Correlation: {corr}")
    ax.scatter(df["days_left"], df["price"])
    ax.set_xlabel("Days Left")
    ax.set_ylabel("Price")
    ax.set_title("Correlation between days_left and price")
    st.pyplot(fig)

    # categorical correlations
    st.write("### Categorical correlations")
    fig, ax = plt.subplots()
    sns.boxplot(data = df, x = "airline", y = "price", ax = ax)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.boxplot(data = df, x = "departure_time", y = "price", ax = ax)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.boxplot(data = df, x = "stops", y = "price", ax = ax)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.boxplot(data = df, x = "destination_city", y = "price", ax = ax)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.boxplot(data = df, x = "source_city", y = "price", ax = ax)
    st.pyplot(fig)

    st.subheader("Observations")
    st.write("We observed that everything was correlated pretty closely.")

    # ======================================================================
    # Step 9 - Relationship Analysis Between Categorical and Continuous Variables
    st.divider()
    st.header('Step 9 - Relationship Analysis Between Categorical and Continuous Variables')

    categorical_df = pd.DataFrame({
        'airline': df['airline'],
        'source_city': df['source_city'],
        'destination_city': df['destination_city'],
        'departure_time': df['departure_time'],
        'stops': df['stops'],
    })

    # Iterate through selected categorical and continuous variables
    for cat_col in categorical_df:
        st.write(f"### Analyzing price by {cat_col}")

        # Perform ANOVA
        groups = [df[df[cat_col] == group]["price"] for group in df[cat_col].unique()]
        f_stat, p_value = stats.f_oneway(*groups)

        # Display ANOVA results
        st.write(f"ANOVA F-statistic: {f_stat}")
        st.write(f"ANOVA P-value: {p_value}")

        if p_value < 0.05:
            st.write(f"Significant relationship between {cat_col} and Price (p-value < 0.05)")
        else:
            st.write(f"No significant relationship between {cat_col} and Price (p-value >= 0.05)")

        # Create histograms for each category
        fig, ax = plt.subplots(figsize=(8, 6))
        for group in df[cat_col].unique():
            sns.histplot(df[df[cat_col] == group]["price"], bins=10, label=f"Group {group}", kde=True, ax=ax)

        # Add titles and labels
        ax.set_title(f'Histogram of price by {cat_col}')
        ax.set_xlabel("Price")
        ax.set_ylabel('Frequency')
        ax.legend(title=cat_col)

        st.pyplot(fig)

    st.subheader("Observations")
    st.write("We observed that the relationship between categorical and continours data were also closely correlated with the P value being really small or even 0, while the F value changing, was in a good range.")

    # ======================================================================
    # Step 10 - Selecting final variables for AI model
    st.divider()
    st.header('Step 10 - Selecting final variables for AI model')
    st.markdown("""
                - Days left
                - Stops
                - Airline
                - Departure Time
                - Source City
                - Destination City
                """)

    st.subheader("Observations")
    st.write("The variables chosen were based on step 4 and the removal of some data columns such as flight number, arrival time and duration.")

    # ======================================================================
    # Step 11 - Converting categorical data for AI model
    st.divider()
    st.header('Step 11 - Convert categorical data for AI model')

    features = df[["airline", "source_city", "destination_city", "departure_time", "stops"]]

    # features
    # drop all categorical columns as well as target column and duration then concatenate the ohencoded categorical data on
    # NOTE: we drop duration as you don't know the duration of a flight before that flight actually happens
    X = df.drop(columns = ["duration", "airline", "source_city", "destination_city", "departure_time", "stops", "price"])
    st.dataframe(X)
    # target
    Y = df["price"]

    st.subheader("Observations")
    st.write("This is the step that caused us to drop the flight number as a feature. This step caused there to be over 1000 columns in the dataset due to One-Hot encoding.")

    # ======================================================================
    # Step 12 - Data split and transformation
    st.divider()
    st.header("Step 12 - Data split and transformation")

    test_size = 0.2
    random_state = 98
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size, random_state = random_state)
    X_ohe_train, X_ohe_test, Y_train, Y_test = train_test_split(features, Y, test_size = test_size, random_state = random_state)

    scaler, X_train_scaler, X_test_scaler = transform_scaler(X_train, X_test)
    ohe_transformer, X_train_encoded, X_test_encoded = transform_ohe(X_ohe_train, X_ohe_test)

    X_train_scaled = pd.concat([X_train_scaler, X_train_encoded], axis = 1)
    X_test_scaled = pd.concat([X_test_scaler, X_test_encoded], axis = 1)

    X_train_scaled = X_train_scaled.rename(columns = {0: "days_left"})
    X_test_scaled = X_test_scaled.rename(columns = {0: "days_left"})

    st.write("X train scaled")
    st.dataframe(X_train_scaled)

    st.write("X test scaled")
    st.dataframe(X_test_scaled)

    st.subheader("Test and train data")
    st.markdown(f"""
                **Features**
                > Train: {X_train_scaled.shape}
                > Test: {X_test_scaled.shape}

                **Target**
                > Train: {Y_train.shape}
                > Test: {Y_test.shape}
                """)

    st.subeader("Observations")
    st.write("We observed that data training and testing took a very long time, and in the end, ended up being pretty close together.")

    # ======================================================================
    # Step 13 - Running multiple models
    st.divider()
    st.header("Step 13 - Running multiple models")

    if "already_trained" not in st.session_state:
        st.session_state.already_trained = train_models(X_train_scaled, Y_train)
    else:
        st.write("All models already trained.")

    r2_scores = {} 
    for name, model in st.session_state.already_trained.items():
        prediction = model.predict(X_test_scaled)
        st.subheader(name)

        r2 = r2_score(Y_test, prediction)
        r2_scores[name] = r2
        st.write(f"r2: {r2}")

    st.subheader("Observations")
    st.write("We observed that training and running the different models, took less time than the first time.")

    # ======================================================================
    # Step 14 - Best model
    st.divider()
    st.header("Step 14 - Best model")

    st.dataframe(r2_scores)
    fig, ax = plt.subplots()
    sns.barplot(r2_scores, ax = ax)
    ax.set_xlabel("model")
    ax.set_ylabel("r2 score")
    st.pyplot(fig)

    best_model_name = max(r2_scores, key = r2_scores.get)
    best_model = st.session_state.already_trained[best_model_name]
    st.write(f"The best model is {best_model_name}")

    # retrain on whole dataset
    days_left_scaled = pd.DataFrame(scaler.fit_transform(pd.DataFrame(df["days_left"])))
    categorical_values = features
    categorical_encoded = pd.DataFrame(ohe_transformer.fit_transform(categorical_values))
    categorical_encoded.columns = ohe_transformer.get_feature_names_out()

    transformed_data = pd.concat([days_left_scaled, categorical_encoded], axis = 1)
    transformed_data.rename(columns = {0: "days_left"}, inplace = True)

    best_model.fit(transformed_data, Y)

    best_model_filename = "best_model.sav"
    joblib.dump(best_model, best_model_filename)

    st.subheader("Observations")
    st.write("We obseverd that the best model was random forest, however knn/k-ne.")

    # ======================================================================
    # Step 15 - Model deployment and showcase
    st.divider()
    st.header("Step 15 - Model deployment and showcase")

    model = joblib.load(best_model_filename)

    days_left = st.number_input(f"Number of days until flight", value = 1, min_value = 1)
    categorical_values = {}
    categorical_values["airline"] = st.selectbox(f"Airline", options = df["airline"].unique())
    categorical_values["source_city"] = st.selectbox(f"Source city", options = df["source_city"].unique())
    categorical_values["destination_city"] = st.selectbox(f"Destination", options = df["destination_city"].unique())
    categorical_values["departure_time"] = st.selectbox(f"Departure time", options = df["departure_time"].unique())
    categorical_values["stops"] = st.selectbox(f"Number of stops", options = df["stops"].unique())

    days_left_scaled = pd.DataFrame(scaler.transform(pd.DataFrame([days_left])))

    categorical_values = pd.DataFrame([categorical_values])
    categorical_encoded = pd.DataFrame(ohe_transformer.transform(categorical_values))
    categorical_encoded.columns = ohe_transformer.get_feature_names_out()

    transformed_data = pd.concat([days_left_scaled, categorical_encoded], axis = 1)
    transformed_data.rename(columns = {0: "days_left"}, inplace = True)

    if st.button("Predict"):
        prediction = model.predict(transformed_data)
        st.write(f"Predicted price: ${prediction[0]:.2f}")

if __name__ == "__main__":
    main()
