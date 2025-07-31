import streamlit as st
import pandas as pd
import joblib
import datetime
from datetime import date
import os
import smtplib
from email.message import EmailMessage

# Email credentials
EMAIL_ADDRESS = "kumarmilan577@gmail.com"       # 🔐 Replace this
EMAIL_PASSWORD = "smte fcrq aglu xmve"   # 🔐 Replace this

# Load ML components
model = joblib.load("D:/infotact/NEW/priority_predictor.joblib")
tfidf = joblib.load("D:/infotact/NEW/tfidf_vectorizer.joblib")
le = joblib.load("D:/infotact/NEW/label_encoder.joblib")

# Mappings
icon_dict = {
    "High": "🔴",
    "Medium": "🟡",
    "Low": "🟢"
}
suggestion_dict = {
    "High": "🚨 High priority. Do this ASAP.",
    "Medium": "⚠️ Medium priority. Schedule soon.",
    "Low": "✅ Low priority. Defer if needed."
}

CSV_PATH = "tasks.csv"

# Set Streamlit UI
st.set_page_config(page_title="AI Task Manager", layout="centered")
st.title("📌 AI Task Management System")
st.subheader("Predict task priority, manage tasks, and get email reminders")

# Initialize session state for result display
if 'last_result' not in st.session_state:
    st.session_state.last_result = None

# --- Task Input Form ---
with st.form("task_form"):
    task_text = st.text_area("📝 Task Description", placeholder="e.g., Complete AI assignment by tomorrow")
    due_date = st.date_input("📅 Due Date", min_value=date.today())
    email = st.text_input("📧 Email for Reminder (optional)")
    submitted = st.form_submit_button("🔍 Analyze & Save Task")

if submitted:
    if not task_text.strip():
        st.warning("Please enter a task description.")
    else:
        X_input = tfidf.transform([task_text])
        pred = model.predict(X_input)[0]
        pred_label = le.inverse_transform([pred])[0]
        icon = icon_dict[pred_label]
        suggestion = suggestion_dict[pred_label]
        due_in_days = (due_date - date.today()).days

        # Save to session
        st.session_state.last_result = {
            "Description": task_text,
            "Due Date": due_date.strftime("%Y-%m-%d"),
            "Priority": pred_label,
            "Icon": icon,
            "Suggestion": suggestion,
            "Due In": due_in_days,
            "Email": email
        }

        # Save to CSV
        task_data = {
            "Description": task_text,
            "Due Date": due_date.strftime("%Y-%m-%d"),
            "Priority": pred_label,
            "Suggestion": suggestion,
            "Email": email
        }
        df_new = pd.DataFrame([task_data])

        if os.path.exists(CSV_PATH):
            df_old = pd.read_csv(CSV_PATH)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_combined = df_new
        df_combined.to_csv(CSV_PATH, index=False)
        st.success("✅ Task saved successfully!")

        # Send Email
        if email.strip():
            try:
                msg = EmailMessage()
                msg['Subject'] = f"[Task Reminder] {pred_label} Priority Task Due on {due_date}"
                msg['From'] = EMAIL_ADDRESS
                msg['To'] = email
                msg.set_content(
                    f"Task: {task_text}\nPriority: {pred_label}\nDue Date: {due_date}\n\nSuggestion: {suggestion}\n\nSent by AI Task Manager"
                )
                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                    smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                    smtp.send_message(msg)
                st.success(f"📧 Reminder sent to {email}")
            except Exception as e:
                st.error(f"Failed to send email: {e}")

# Show prediction result if exists
if st.session_state.last_result:
    st.markdown("### 🔍 Last Task Analysis")
    st.write(f"**Task:** {st.session_state.last_result['Description']}")
    st.write(f"**Predicted Priority:** {st.session_state.last_result['Icon']} {st.session_state.last_result['Priority']}")
    st.write(f"**Due in:** `{st.session_state.last_result['Due In']}` day(s)")
    st.write(f"**Suggestion:** {st.session_state.last_result['Suggestion']}")
    if st.session_state.last_result['Email']:
        st.write(f"**Reminder Email:** {st.session_state.last_result['Email']}")

# --- Task Dashboard ---
st.markdown("---")
st.header("📋 Task Dashboard")

if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    if not df.empty:
        priority_filter = st.selectbox("🔎 Filter by Priority", ["All"] + df["Priority"].unique().tolist())
        if priority_filter != "All":
            df = df[df["Priority"] == priority_filter]

        st.markdown("### 🛠️ Edit or Delete Tasks Below")
        edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True, key="editor")
        if st.button("💾 Save Changes"):
            edited_df.to_csv(CSV_PATH, index=False)
            st.success("Changes saved to task list.")

        if st.button("🗑️ Delete All Tasks"):
            os.remove(CSV_PATH)
            st.warning("All tasks deleted. Reload the app.")
    else:
        st.info("No tasks found.")
else:
    st.info("No tasks added yet.")
