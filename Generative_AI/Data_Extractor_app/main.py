import streamlit as st
from data_extractor import extract_movie_data
import json

# Set page config for a wider layout and better title
st.set_page_config(page_title="Movie Data Extractor", page_icon="🎬", layout="centered")

st.title("🎬 Movie Data Extractor")
st.markdown("Enter a movie article below to extract key information instantly.")

# Text area input
article_text = st.text_area("Enter movie article:", height=200)

if st.button("Extract", type="primary"):
    if article_text.strip():
        try:
            with st.spinner("Analyzing article and extracting data..."):
                extracted_data = extract_movie_data(article_text)
            
            st.success("Extraction Complete!")
            st.subheader("Extracted Movie Information")
            
            # Define the fields we want to display in the table
            fields_to_display = [
                'Actor Name', 'Director', 'Movie Title', 'Release Year', 
                'Genre', 'Budget', 'Revenue', 'Studio', 'IMDb Rating'
            ]
            
            # Create a clean list of dictionaries for the Streamlit table
            table_data = []
            for field in fields_to_display:
                value = extracted_data.get(field, "N/A")
                # Convert None/Null to "N/A" for display
                if value is None or value == "":
                    value = "N/A"
                    
                table_data.append({"Field": field, "Extracted Value": value})
            
            # Display the table using st.dataframe (looks cleaner than st.table)
            st.dataframe(
                table_data, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Field": st.column_config.TextColumn("Field", width="medium"),
                    "Extracted Value": st.column_config.TextColumn("Extracted Value", width="large")
                }
            )
            
            # --- ENHANCEMENTS ---
            st.divider()
            
            # Option to download the data as JSON
            json_string = json.dumps(extracted_data, indent=2)
            st.download_button(
                label="📥 Download Data as JSON",
                file_name="movie_data.json",
                mime="application/json",
                data=json_string,
            )
            
            # Expander to view the raw JSON
            with st.expander("View Raw JSON Output"):
                st.json(extracted_data)
                
        except Exception as e:
            st.error(f"Extraction failed: {str(e)}")
    else:
        st.warning("Please enter some text in the text area first.")