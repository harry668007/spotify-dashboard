import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

# Set page configuration for wide layout
st.set_page_config(
    page_title="Spotify Listener Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize RoBERTa QA Model
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

# App Title
st.title("Spotify Listener Dashboard")
st.write("Upload your Spotify JSON file(s) to explore your listening habits and ask questions.")
# Instructions for Users
st.subheader("How to Get Your Spotify Streaming History")
st.write("""
To analyze your Spotify listening habits, you need your 'StreamingHistory' JSON files from Spotify. 
Follow these steps to request and download your data:
1. Go to the [Spotify Privacy Settings](https://www.spotify.com/account/privacy/).
2. Under **Download your data**, click **Request**.
3. Spotify will email you a ZIP file containing your data within a few days.
4. Extract the ZIP file and locate the `StreamingHistory` JSON files (e.g., `StreamingHistory0.json`, `StreamingHistory1.json`).
5. Upload these files below to analyze your listening habits.
""")

def detect_file_format(data):
    """
    Detect whether the uploaded JSON file is in the old or new format.
    Returns 'old' or 'new' based on the detected format.
    """
    if "ts" in data[0] and "master_metadata_track_name" in data[0]:
        return "new"
    elif "endTime" in data[0] and "trackName" in data[0]:
        return "old"
    else:
        return "unknown"

# def convert_old_format(data):
#     """
#     Converts old Spotify data format to DataFrame.
#     """
#     return pd.DataFrame(data)

def convert_new_format_to_old_format(new_data):
    """
    Converts new Spotify data format to the old format by extracting relevant columns
    and reformatting the fields.
    """
    converted_data = []
    for record in new_data:
        converted_data.append({
            "endTime": pd.to_datetime(record["ts"]).strftime("%Y-%m-%d %H:%M"),  # Convert 'ts' to desired format
            "artistName": record.get("master_metadata_album_artist_name", None),  # Artist name
            "trackName": record.get("master_metadata_track_name", None),  # Track name
            "msPlayed": record.get("ms_played", 0)  # Playtime in milliseconds
        })
    return pd.DataFrame(converted_data)



# File Upload Section
uploaded_files = st.file_uploader("Upload JSON File(s)", type="json", accept_multiple_files=True)

# Ensure session state for the query response
if "llm_response" not in st.session_state:
    st.session_state.llm_response = ""

if uploaded_files:
    combined_data = []
    for file in uploaded_files:
        # Load the JSON data
        file_data = pd.read_json(file).to_dict(orient="records")
        
        # Detect the format
        file_format = detect_file_format(file_data)
        if file_format == "old":
            converted_df = pd.DataFrame(file_data)
        elif file_format == "new":
            converted_df = convert_new_format_to_old_format(file_data)
        else:
            st.error("Unknown file format. Please upload valid Spotify data files.")
            continue
        
        combined_data.append(converted_df)

    # Combine all converted data
    if combined_data:
        df = pd.concat(combined_data, ignore_index=True)
        # st.write("Data successfully loaded and converted!")
        # st.write(df.head())  # Preview the converted DataFrame
    else:
        st.error("No valid data to process. File conversion failed. Check the file formats")

    # Data Cleaning
    df['endTime'] = pd.to_datetime(df['endTime'])
    df = df[df['msPlayed'] > 0]
    df['hour'] = df['endTime'].dt.hour
    df['day_of_week'] = df['endTime'].dt.day_name()
    df['duration_minutes'] = df['msPlayed'] / 60000
    df['month'] = df['endTime'].dt.month

    # KPI Calculations
    total_hours = df['duration_minutes'].sum() / 60
    unique_artists = df['artistName'].nunique()
    unique_songs = df['trackName'].nunique()
    most_active_hour = df.groupby('hour')['duration_minutes'].sum().idxmax()
    avg_listens_per_day = df.groupby(df['endTime'].dt.date)['duration_minutes'].sum().mean()
    avg_listens_per_month = df.groupby('month')['duration_minutes'].sum().mean()
    avg_listens_per_hour = df.groupby('hour')['duration_minutes'].sum().mean()
    top_artist = df.groupby('artistName')['duration_minutes'].sum().idxmax()
    top_song = df.groupby('trackName')['duration_minutes'].sum().idxmax()


    # Display KPIs
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Total Hours Listened", value=f"{total_hours:.2f} hours")
    col2.metric(label="Unique Artists", value=unique_artists)
    col3.metric(label="Unique Songs", value=unique_songs)

    col4, col5, col6 = st.columns(3)
    col4.metric(label="Most Active Hour", value=f"{most_active_hour}:00")
    col5.metric(label="Avg Listens per Day", value=f"{avg_listens_per_day:.2f} minutes")
    col6.metric(label="Avg Listens per Month", value=f"{avg_listens_per_month:.2f} minutes")

    # Visualizations
    st.subheader("Visualizations")

    # First Row of Charts
    col1, col2, col3 = st.columns(3)
    with col1:
        # Top 5 Artists by Playtime
        top_artists = df.groupby('artistName')['duration_minutes'].sum().nlargest(5).reset_index()
        fig1 = px.bar(top_artists, x='artistName', y='duration_minutes', title="Top 5 Artists by Playtime")
        st.plotly_chart(fig1)

    with col2:
        # Top 5 Songs by Playtime
        top_songs = df.groupby('trackName')['duration_minutes'].sum().nlargest(5).reset_index()
        fig2 = px.bar(top_songs, x='trackName', y='duration_minutes', title="Top 5 Songs by Playtime")
        st.plotly_chart(fig2)

    # Second Row of Charts
    # col4, col5, col6 = st.columns(3)
    with col3:
        # Listening Trends by Hour
        hourly_playtime = df.groupby('hour')['duration_minutes'].sum().reset_index()
        fig3 = px.line(hourly_playtime, x='hour', y='duration_minutes', title="Listening Trends by Hour")
        st.plotly_chart(fig3)
    col4, col5, col6 = st.columns(3)
    with col4:
        # Weekly Listening Heatmap
        active_day_hour = df.groupby(['day_of_week', 'hour'])['duration_minutes'].sum().reset_index()
        heatmap_data = active_day_hour.pivot(index='day_of_week', columns='hour', values='duration_minutes').fillna(0)
        fig4 = px.imshow(heatmap_data, labels=dict(x="Hour", y="Day", color="Minutes"), title="Weekly Listening Heatmap")
        st.plotly_chart(fig4)

    # Third Row of Charts
    # col5, col6 = st.columns(2)
    with col5:
        # Monthly Playtime Trends
        monthly_playtime = df.groupby('month')['duration_minutes'].sum()
        fig5 = px.line(monthly_playtime, x=monthly_playtime.index, y=monthly_playtime.values, title="Monthly Playtime Trends")
        st.plotly_chart(fig5)

    with col6:
        # Playtime Distribution (Fully vs Partially Played)
        fully_played = len(df[df['msPlayed'] >= (3 * 60 * 1000) * 0.8])  # Assuming 3 min avg song length
        partially_played = len(df[df['msPlayed'] < (3 * 60 * 1000) * 0.8])
        play_distribution = pd.DataFrame({
            'Category': ['Fully Played', 'Partially Played'],
            'Count': [fully_played, partially_played]
        })
        fig6 = px.pie(play_distribution, values='Count', names='Category', title="Playtime Distribution")
        st.plotly_chart(fig6)

    # Calculate Metrics
    total_hours = df['duration_minutes'].sum() / 60
    unique_artists = df['artistName'].nunique()
    unique_songs = df['trackName'].nunique()
    most_active_hour = df.groupby('hour')['duration_minutes'].sum().idxmax()
    most_active_day = df.groupby('day_of_week')['duration_minutes'].sum().idxmax()
    date_range_start = df['endTime'].min().date()
    date_range_end = df['endTime'].max().date()

    # Listening Trends
    avg_listens_per_day = df.groupby(df['endTime'].dt.date)['duration_minutes'].sum().mean()
    avg_listens_per_month = df.groupby(df['month'])['duration_minutes'].sum().mean()
    avg_listens_per_hour = df.groupby(df['hour'])['duration_minutes'].sum().mean()
    highest_month = df.groupby('month')['duration_minutes'].sum().idxmax()
    lowest_month = df.groupby('month')['duration_minutes'].sum().idxmin()

    # Comparisons
    weekday_playtime = df[df['day_of_week'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])]['duration_minutes'].sum()
    weekend_playtime = df[df['day_of_week'].isin(['Saturday', 'Sunday'])]['duration_minutes'].sum()

    # Playtime Distribution
    fully_played = len(df[df['msPlayed'] >= (3 * 60 * 1000) * 0.8])
    partially_played = len(df[df['msPlayed'] < (3 * 60 * 1000) * 0.8])
    fully_played_percentage = fully_played / len(df) * 100

    # Longest/Shortest Sessions
    df['session'] = (df['endTime'] - df['endTime'].shift(1)).dt.total_seconds() > 1800
    session_lengths = df.groupby(df['session'].cumsum())['duration_minutes'].sum()
    longest_session = session_lengths.max()
    shortest_session = session_lengths.min()

    # Top Artists and Songs
    top_artist = df.groupby('artistName')['duration_minutes'].sum().idxmax()
    top_song = df.groupby('trackName')['duration_minutes'].sum().idxmax()
    
    weekday_playtime = df[df['day_of_week'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])]['duration_minutes'].sum()
    weekend_playtime = df[df['day_of_week'].isin(['Saturday', 'Sunday'])]['duration_minutes'].sum()

    monthly_top_artist = df.groupby(['month', 'artistName'])['duration_minutes'].sum().idxmax()[1]
    monthly_top_song = df.groupby(['month', 'trackName'])['duration_minutes'].sum().idxmax()[1]

    top_5_artist_playtime = df.groupby('artistName')['duration_minutes'].sum().nlargest(5).sum()

    session_deltas = (df['endTime'] - df['endTime'].shift()).dt.total_seconds().fillna(0)
    df['session'] = (session_deltas > 1800).cumsum()
    session_lengths = df.groupby('session')['duration_minutes'].sum()
    avg_session_duration = session_lengths.mean()
    song_repeatability_count = df['trackName'].duplicated().sum()

    monthly_trend_variance = df.groupby('month')['duration_minutes'].sum().pct_change().mean() * 100
    consistency_rate = (len(df.groupby('day_of_week')) / 7) * 100
    # Calculate the total listening time per month
    monthly_activity = df.groupby('month')['duration_minutes'].sum()

    # Get the most active month
    most_active_month = monthly_activity.idxmax()
    most_active_month_total = monthly_activity.max()

    # LLM Section
    

    st.subheader("Ask a Question About Your Data")
    with st.container():
        user_query = st.text_input("Type your question here:")

        if user_query:
            # Context for the LLM
            context = f"""
            Your Spotify listening data (data range) spans from {date_range_start} to {date_range_end}.
            You have listened for a total of {total_hours:.2f} hours.
            You streamed {unique_songs} unique songs and {unique_artists} unique artists.
            Your most active hour of the day is {most_active_hour}:00.
            The most active day of the week is {most_active_day}.
            On average, you listen to {avg_listens_per_day:.2f} minutes per day and {avg_listens_per_month:.2f} minutes per month.
            The month with the highest listening time is {highest_month}, and the lowest is {lowest_month}.
            Fully played songs make up {fully_played_percentage:.2f}% of your total listening activity.
            Your longest listening session lasted {longest_session:.2f} minutes, and the shortest lasted {shortest_session:.2f} minutes.
            Your top artist overall is {top_artist}, and your most played song is {top_song}.
            You have listened to {weekday_playtime:.2f} minutes on weekdays and {weekend_playtime:.2f} minutes on weekends.
            Your top artist by month is {monthly_top_artist}.
            Your top song by month is {monthly_top_song}.
            You listen to {top_artist} the most during late-night hours.
            Your top 5 artists account for {top_5_artist_playtime:.2f} minutes of your listening time.
            Your playtime distribution is {fully_played_percentage:.2f}% fully played songs and {100 - fully_played_percentage:.2f}% partially played songs.
            Your average listening session duration is {avg_session_duration:.2f} minutes.
            You have repeated songs {song_repeatability_count} times in your dataset.
            Your listening activity has been consistent at {consistency_rate:.2f}% across the week.
            Your listening trends have changed month-to-month with {monthly_trend_variance:.2f}% variance in playtime.
            """

            # Generate LLM Response
            try:
                response = qa_model(question=user_query, context=context)
                if response['score'] > 0.3:
                    st.session_state.llm_response = response['answer']
                else:
                    st.session_state.llm_response = "Sorry, I couldn't find an answer to your question."
            except Exception as e:
                st.session_state.llm_response = f"Error processing your request: {e}"

        # Display LLM Response
        if st.session_state.llm_response:
            st.markdown(f"**LLM Response:** {st.session_state.llm_response}")
 
            
            
            # response = qa_model(question=user_query, context=context)
            # if response['score'] > 0.5:
            #     st.write(f"**Answer:** {response['answer']}")
            # else:
            #     st.write("Sorry, I couldn't find an answer to your question. Please try rephrasing.")
else:
    st.write("Please upload your Spotify Streaming History JSON files to begin.")
