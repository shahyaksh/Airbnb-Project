import streamlit as st
import pandas as pd
from PropertySearch import search
from langchain_core.messages import AIMessage, HumanMessage

property_listings_df = pd.read_csv('Cleaned_Listings_final.csv',low_memory=False)


def display_results(properties):
    """Display recommended properties with details."""
    for recommended_property in properties:
        property_id = recommended_property.metadata['property_id']
        image_url = property_listings_df[property_listings_df['id']==property_id]['picture_url'].values[0]
        print(image_url)
        st.image(image_url, use_column_width=True)
        st.markdown(recommended_property.metadata["property_url"])
        st.markdown(f"**Description:** {recommended_property.page_content}")
        st.markdown(f"**Accomodates:** {recommended_property.metadata['accommodates']}")
        st.markdown(f"**City:** {recommended_property.metadata['City_name']}")
        st.markdown(f"**Neighbourhood:** {recommended_property.metadata['neighborhood_name']}")
        st.markdown(f"**Amenities:** {recommended_property.metadata['imp_amenities']}")
        st.markdown(f"**Price:** {recommended_property.metadata['price']}")
        st.markdown("---")  # Divider for visual clarity


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am an Airbnb Property Search bot. How can I help you?"),
    ]
    st.session_state.property_metadata = {}
    st.session_state.prev_description = ""
    st.session_state.results = []

# Streamlit app configuration
st.set_page_config(page_title='Airbnb Property Search Bot', page_icon='🤖')
st.title(':blue[Airbnb Property Search Bot] 🤖')
st.subheader('Your :green[AI Assistant] for finding ideal properties', divider='rainbow')
st.caption("Note: This bot can make mistakes. Please verify important information.")

# Display chat history

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            try:
                st.markdown(message.content)
                display_results(message.metadata['properties'])
            except:
                continue


# Accept user input
user_input = st.chat_input("Write a short description of what you are looking for in an Airbnb property.")

if user_input and user_input.strip():
    # Add user input to chat history
    st.session_state.chat_history.append(HumanMessage(user_input))
    with st.chat_message("Human"):
        st.markdown(user_input)

    try:
        # Determine if it's an initial query or a refined query
        if len(st.session_state.chat_history) == 1:  # Initial query
            st.session_state.property_metadata = search.call_gemini_api(user_input)
            docs = search.search_similar_properties(user_input,st.session_state.property_metadata)
            st.session_state.prev_description = user_input
        else:  # Refined query
            query_dictionary = {
                "prev_query": st.session_state.prev_description,
                "previous_metadata": st.session_state.property_metadata,
                "current_query": user_input
            }
            query_prompt = search.generate_refined_prompt(query_dictionary)
            refined_data = search.call_gemini_api(query_prompt)
            st.session_state.property_metadata = refined_data.get("updated_metadata", {})
            refined_query = refined_data.get("refined_query", user_input)
            docs = search.search_similar_properties(refined_query,st.session_state.property_metadata)
            st.session_state.prev_description = refined_query

        if docs:
            # Save results to session state
            st.session_state.results.append(docs)

            # Display the current results
            with st.chat_message("AI"):
                st.markdown("Here are the recommended properties based on your input:")
                display_results(docs)
            st.session_state.chat_history.append(
                AIMessage(content="Here are the recommended properties based on your input:",
                          metadata={"properties": docs}))
        else:
            with st.chat_message("AI"):
                st.markdown("No matching properties found. Please refine your query.")
            st.session_state.chat_history.append(
                AIMessage(content="No matching properties found. Please refine your query."))
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")