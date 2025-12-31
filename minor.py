
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time 

# --- 1. Data Loading and Preprocessing ---
try:
    # Using the local EXCEL file we created earlier
    df =   pd.read_excel(r"C:\Users\ssp14\OneDrive\Desktop\tourdata.xlsx")
except FileNotFoundError:
    st.error("Error: 'india_tourism_with_budget.xlsx' not found. Please ensure it's in the directory.")
    st.stop()

def extract(text):
    """Extracts the first element of a semi-colon separated string."""
    if pd.isna(text):
        return ""
    return text.split(';')[0].strip()

# Apply feature engineering (Same as original)
df["wonders"] = df["tags"].apply(extract)
df["enjoy"] = df["activities"].apply(extract)
df["category"] = df["tags"].str.split(';').str[1].fillna("General")
df["activity"] = df["activities"].str.split(';').str[1].fillna("Sightseeing")
df["budget"] = df["estimated_budget"]


# Select and create the new DataFrame (Removed Location/Map columns)
new_columns = ["place_name", "state_ut", "description", "best_time_to_visit", 
               "nearby_attractions", "wonders", "enjoy", "category", "activity", 
               "budget"]
new_data = pd.DataFrame(df, columns=new_columns).fillna('')

# Combine features into a single string for TF-IDF
new_data["combined_features"] = (
    new_data["place_name"].str.lower() + ' ' +
    new_data["description"].str.lower() + ' ' +
    new_data["category"].str.lower() + ' '  +  
    new_data["activity"].str.lower() + ' '  +  
    new_data["best_time_to_visit"].str.lower() + ' ' +
    new_data["nearby_attractions"].str.lower()
)

# --- 2. TF-IDF Vectorization and Cosine Similarity ---
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tf_matrix = tfidf.fit_transform(new_data["combined_features"])
cosine_sim = cosine_similarity(tf_matrix, tf_matrix)

# --- 3. Recommendation Logic ---
def recommend_system(budget, preferred_activity, top_n=5):
    filtered_data = new_data[new_data["budget"] <= budget].copy()
    if filtered_data.empty:
        return pd.DataFrame()

    seed_index = -1
    if preferred_activity and preferred_activity != "No Preference":
         activity_filtered = filtered_data[filtered_data['activity'].str.contains(preferred_activity, case=False, na=False)]
         if not activity_filtered.empty:
             seed_index = activity_filtered.index[0]
         else:
             seed_index = filtered_data.sort_values(by='budget').index[0]
    else:
         seed_index = filtered_data.sort_values(by='budget').index[0]
    
    sim_scores = list(enumerate(cosine_sim[seed_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[1:]] 
    
    final_recommendation_indices = []
    count = 0 
    for index in top_indices:
        if new_data.loc[index, 'budget'] <= budget: 
            final_recommendation_indices.append(index)
            count += 1
            if count >= top_n:
                break

    if not final_recommendation_indices:
        return filtered_data.head(top_n).sort_values(by='budget', ascending=True)

    recommendation_df = new_data.iloc[final_recommendation_indices]
    return recommendation_df.sort_values(by=['budget'], ascending=True)

# --- 4. Streamlit Application Interface ---
st.set_page_config(
    page_title="Tourist Spot Recommender",
    page_icon="🧭",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main { background-color: #f0f2f6; }
div.stButton > button { border-radius: 8px; padding: 10px 24px; font-size: 16px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #008080;'>🌍 India Tourist Recommendation System 🏛️</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1em;'>Plan your perfect trip based on your preferences and budget.</p>", unsafe_allow_html=True)

st.markdown("---")

# --- Recommendation Form (Removed Trip Duration/Date) ---
st.header("🎯 Set Your Travel Parameters")
with st.form("recommendation_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        user_budget = st.number_input(
            "1. Enter your amount between 3000 to 40000 (in ₹):", 
            step = 1000.0,
            format="%.2f"
        )

    with col2:
        unique_activities = ["No Preference"] + sorted(new_data['activity'].unique().tolist())
        preferred_activity = st.selectbox(
            "2. Preferred Activity/Theme:", 
            options=unique_activities,
            index=0
        )
        
    submitted = st.form_submit_button("🔍 Show Recommendations", type="primary")

# --- Results Display Section ---
if submitted:
    if user_budget <= 0:
        st.error("Please enter a budget greater than 0.")
    else:
        results = recommend_system(user_budget, preferred_activity)
        st.markdown("---")
        
        if isinstance(results, pd.DataFrame) and not results.empty:
            st.success(f"🥳 Top Destinations Recommended for your budget of ₹{user_budget:,.2f}:")
            st.markdown("---")

            # --- Keyword Search Filter ---
            st.subheader("🔎 Filter Recommendations by Keyword")
            keyword = st.text_input("Search by Place, Activity, or Description:", placeholder="e.g., 'beach'")
            filtered_results = results.copy()
            if keyword:
                filtered_results = filtered_results[
                    filtered_results["place_name"].str.lower().str.contains(keyword.lower(), na=False) |
                    filtered_results["description"].str.lower().str.contains(keyword.lower(), na=False)
                ]
                
            if filtered_results.empty:
                st.warning("No matches found.")
            else:
                st.subheader("Detailed Recommendations")
                num_results = len(filtered_results)
                # Show results in a clean grid
                cols = st.columns(3)
                for i, (index, row) in enumerate(filtered_results.iterrows()):
                    with cols[i % 3]: 
                        st.markdown(
                            f"""
                            <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin-bottom: 20px; background-color: white; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);">
                                <h4 style="color: #FF4B4B;">{row['place_name']}</h4>
                                <p><strong>📍 State:</strong> {row['state_ut']}</p>
                                <p><strong>💵 Budget:</strong> ₹{row['budget']:,.2f}</p>
                                <p><strong>🗓️ Best Time:</strong> {row['best_time_to_visit']}</p>
                                <p><strong>✨ Theme:</strong> {row['category']}</p>
                                <details><summary><strong>📝 Description</strong></summary><p>{row['description']}</p></details>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
        else:
            st.warning("No destinations found within this budget.")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ App Utilities")
    st.subheader("🆘 Send Feedback")
    with st.form("help_form"):
        message = st.text_area("Feedback", height=100)
        if st.form_submit_button("✉️ Submit"):
            st.success("Thank you!")
    
    st.markdown("---")
    st.subheader("🔍 Explore All")
    all_states = sorted(new_data['state_ut'].unique().tolist())
    selected_state = st.selectbox("Filter by State:", options=["All India"] + all_states)
    
    explore_data = new_data if selected_state == "All India" else new_data[new_data['state_ut'] == selected_state]
    st.dataframe(explore_data[['place_name', 'budget', 'activity']], hide_index=True)

st.markdown("---")

st.markdown("<p style='text-align: center; font-size: 0.8em; color: gray;'>Tourist Recommendation System | Built with Streamlit, TF-IDF, and Cosine Similarity.</p>", unsafe_allow_html=True)
