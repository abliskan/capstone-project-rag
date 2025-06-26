import streamlit as st
import uuid
import plotly.express as px
import time
from datetime import datetime
from chains.rag_chains import get_memory_rag_answer, observe

# Try to import Langfuse for additional tracking
try:
    from langfuse import Langfuse
    from langfuse.decorators import observe
    LANGFUSE_ENABLED = True
except ImportError:
    LANGFUSE_ENABLED = False
    def observe(name=None):
        def decorator(func):
            return func
        return decorator

@observe(name="streamlit_chat_query")
def render_chat_interface():
        st.subheader("🧠 Chat with SourchefBot")
        query = st.chat_input("Ask: 'Suggest a vegan dinner under 600 kcal'")
        if query:
            start_time = time.time()
            with st.spinner("🤔 SourchefBot is Thinking..."):
                response = get_memory_rag_answer(query, user_id=st.session_state.user_id)
                response_time = time.time() - start_time
                
                # Update analytics
                st.session_state.analytics["total_queries"] += 1
                
                # Update average response time
                current_avg = st.session_state.analytics["avg_response_time"]
                total_queries = st.session_state.analytics["total_queries"]
                st.session_state.analytics["avg_response_time"] = \
                    (current_avg * (total_queries - 1) + response_time) / total_queries
                
                # Append to chat messages
                st.session_state.chat_messages.append({
                    "user": query,
                    "bot": response["answer"],
                    "title": response["title"],
                    "nutrition": response["nutrition"],
                    "videos": response["videos"],
                    "timestamp": datetime.now(),
                    "response_time": response_time
                })
                st.session_state.chat_history.append(response)
                st.session_state.analytics["prompt_counts"][query] = st.session_state.analytics["prompt_counts"].get(query, 0) + 1
                st.success(f"Response generated in {response_time:.2f}s")

        for msg in st.session_state.chat_messages:
            st.markdown(f"**🧑 You:** {msg['user']}")
            st.markdown(f"**🤖 SourchefBot:** {msg['bot']}")
            st.markdown(f"**📊 Nutrition:**")
            nutrition = msg['nutrition']

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("Calories", f"{nutrition['calories']}")

            with col2:
                st.metric("Protein", f"{nutrition['protein']}g")

            with col3:
                st.metric("Fat", f"{nutrition['fat']}g")
                
            with col4:
                st.metric("Fat", f"{nutrition['fat']}g")
            
            with col5:
                st.metric("Carbs", f"{nutrition['carbs']}g")
            
            st.markdown("---")
            
            st.markdown("🎥 **Suggested Videos**")
            for vid in msg["videos"]:
                st.markdown(f"**{vid['title']}**")
                if "embed" in vid and vid["embed"].startswith("https://www.youtube.com/embed/"):
                    st.video(vid["embed"])
                else:
                    st.warning("⚠️ Video embed URL missing or malformed.")
                if "link" in vid:
                    st.markdown(f"[▶️ Watch on YouTube]({vid['link']})", unsafe_allow_html=True)

        if st.session_state.chat_history and st.button("❤️ Save to Favorites"):
            latest = st.session_state.chat_history[-1]
            st.session_state.favorites.append(latest)
            st.success("Saved to favorites.")
            st.session_state.analytics["favorite_counts"][latest["title"]] = st.session_state.analytics["favorite_counts"].get(latest["title"], 0) + 1

def render_meal_plan_interface():
        st.subheader("📅 Weekly Meal Planner")
        if not st.session_state.favorites:
            st.info("Save some favorite recipes first!")
        else:
            options = [f["title"] for f in st.session_state.favorites]
            
            # Create meal plan grid
            col1, col2 = st.columns(2)
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
            for i, day in enumerate(days):
                with col1 if i % 2 == 0 else col2:
                    current_meal = st.session_state.weekly_plan[day]
                    current_index = 0
                    if current_meal in options:
                        current_index = options.index(current_meal)
            
                    st.session_state.weekly_plan[day] = st.selectbox(
                        f"**{day}**", 
                        options, 
                        index=current_index,
                        key=f"meal_{day}"
                    )
            
            # Action buttons
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("✅ Save Plan"):
                    st.success("🎉 Meal plan saved successfully!")
            
            with col_btn2:
                if st.button("🗑️ Clear Plan"):
                    for day in st.session_state.weekly_plan:
                        st.session_state.weekly_plan[day] = ""
                    st.success("🧹 Meal plan cleared!")
                    st.rerun()

def render_favorites_interface():
        st.subheader("❤️ Your Favorite Recipes")
        if not st.session_state.favorites:
            st.info("No favorites saved yet.")
        else:
            # Search and filter
            search_term = st.text_input("🔍 Search favorites", placeholder="Enter recipe name...")
            
            filtered_favorites = st.session_state.favorites
            if search_term:
                filtered_favorites = [
                    fav for fav in st.session_state.favorites 
                    if search_term.lower() in fav["title"].lower()
                ]
            
            st.markdown(f"### 📋 {len(filtered_favorites)} Recipe(s)")
            for i, fav in enumerate(st.session_state.favorites):
                st.markdown(f"### 🍽️ {fav['title']}")
                st.markdown(fav["answer"])
                if st.button(f"🗑️ Remove", key=f"remove_{i}"):
                    st.session_state.favorites.pop(i)
                    st.success("Removed.")
                    st.experimental_rerun()

def render_analytics_interface():
        st.subheader("📈 Analytics Dashboard")
        analytics = st.session_state.analytics
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Queries", analytics["total_queries"])
        
        with col2:
            st.metric("Avg Response Time", f"{analytics['avg_response_time']:.2f}s")
        
        with col3:
            st.metric("Favorite Recipes", len(st.session_state.favorites))
        
        with col4:
            session_duration = datetime.now() - analytics["session_start"]
            hours = session_duration.total_seconds() / 3600
            st.metric("Session Duration", f"{hours:.1f}h")
        
        st.subheader("📊 Prompt Frequency")
        data = st.session_state.analytics["prompt_counts"]
        if data:
            df = {"Prompt": list(data.keys()), "Count": list(data.values())}
            with st.container():
                st.write("This is plotting the frequency of prompts used in the chat.")
                st.plotly_chart(px.bar(df, x="Prompt", y="Count"), use_container_width=True)

        st.subheader("❤️ Favorite Recipes")
        favs = st.session_state.analytics["favorite_counts"]
        if favs:
            df2 = {"Recipe": list(favs.keys()), "Count": list(favs.values())}
            with st.container():
                st.write("This is plotting the frequency of favorite recipes.")
                st.plotly_chart(px.bar(df2, x="Recipe", y="Count"), use_container_width=True)
        
        st.subheader("📝 Recent Activity")
        if st.session_state.chat_messages:
            recent_messages = st.session_state.chat_messages[-5:]  # Last 5 messages
            for msg in reversed(recent_messages):
                st.markdown(f"**{msg['timestamp'].strftime('%H:%M:%S')}** - {msg['title']}")
        else:
            st.info("No recent activity.")

def main():
    st.set_page_config(page_title="AI-Sourchef-Bot - Food Recommender + Recipes", page_icon="🥗")
    st.title("🥗 AI-Sourchef-Bot - Food Recommender + Recipes")

    # Session state initialization
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "favorites" not in st.session_state:
        st.session_state.favorites = []
    if "weekly_plan" not in st.session_state:
        st.session_state.weekly_plan = {d: "" for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]}
    if "analytics" not in st.session_state:
        st.session_state.analytics = {
            "prompt_counts": {}, 
            "favorite_counts": {},
            "session_start": datetime.now(),
            "total_queries": 0,
            "avg_response_time": 0
        }

    # Sidebar menu
    menu = st.sidebar.radio("📚 Sourchef Menu", ["🧠 Chat", "📅 Meal Plan", "❤️ Favorites", "📈 Analytics"], label_visibility="collapsed")
    # User info in sidebar
    st.sidebar.caption(f"**User ID:** `{st.session_state.user_id[:8]}...`")
    st.sidebar.caption(f"**Session:** {st.session_state.analytics['total_queries']} queries")
    st.sidebar.success("🟢 Langfuse Enabled")
    
    # Main content based on menu selection
    if menu == "🧠 Chat":
        render_chat_interface()
    elif menu == "📅 Meal Plan":
        render_meal_plan_interface()
    elif menu == "❤️ Favorites":
        render_favorites_interface()
    elif menu == "📈 Analytics":
        render_analytics_interface()

if __name__ == "__main__":
    main()