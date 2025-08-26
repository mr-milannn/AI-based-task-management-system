import streamlit as st
import pandas as pd
import joblib
import datetime
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import date

# --- Load ML components ---

base_path = os.path.dirname(__file__)  # folder where app_new.py is located

model = joblib.load(os.path.join(base_path, "priority_predictor.joblib"))
tfidf = joblib.load(os.path.join(base_path, "tfidf_vectorizer.joblib"))
le = joblib.load(os.path.join(base_path, "label_encoder.joblib"))


# --- Setup Streamlit UI ---
st.set_page_config(page_title="AI Task Manager", layout="centered")
st.title("📌 AI Task Management System")
st.subheader("Predict task priority & get intelligent suggestions")

# --- Helper Maps ---
icon_dict = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
suggestion_dict = {
    "High": "🚨 High priority. Do this ASAP.",
    "Medium": "⚠️ Medium priority. Schedule soon.",
    "Low": "✅ Low priority. Defer if needed."
}

# --- Load/Initialize Task Data ---
csv_path = "task_data.csv"
if "df" not in st.session_state:
    try:
        st.session_state.df = pd.read_csv(csv_path)
    except:
        st.session_state.df = pd.DataFrame(columns=["Title", "Description", "Due Date", "Priority", "Suggestion", "Days Left", "Email"])

# --- Task Input Form ---
with st.form("task_form"):
    task_title = st.text_input("📌 Task Title")
    task_text = st.text_area("📝 Task Description")
    due_date = st.date_input("📅 Due Date", min_value=date.today())
    email = st.text_input("📧 Email to notify (optional)")
    submit = st.form_submit_button("🔍 Analyze Task")

# --- Predict and Save ---
if submit and task_text:
    vec = tfidf.transform([task_text])
    pred = model.predict(vec)[0]
    pred_label = le.inverse_transform([pred])[0]
    icon = icon_dict[pred_label]
    suggestion = suggestion_dict[pred_label]
    days_left = (due_date - date.today()).days

    st.markdown("### 🔍 Task Analysis")
    st.write(f"**Predicted Priority:** {icon} {pred_label}")
    st.write(f"**Due in:** {days_left} day(s)")
    st.write(f"**Suggestion:** {suggestion}")

    # Save to dataframe
    new_row = pd.DataFrame([{
        "Title": task_title,
        "Description": task_text,
        "Due Date": due_date,
        "Priority": pred_label,
        "Suggestion": suggestion,
        "Days Left": days_left,
        "Email": email
    }])
    st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
    st.session_state.df.to_csv(csv_path, index=False)

    # Optional: Email Reminder
    if email:
        try:
            gmail_user = 'kumarmilan577@mail.com'
            gmail_pass = 'bnhn jlmy oasv sfpe'  # Use App Password, NOT regular password
            msg = MIMEMultipart()
            msg['From'] = gmail_user
            msg['To'] = email
            msg['Subject'] = f"Task Reminder: {task_title}"
            body = f"""
            Task: {task_title}
            Description: {task_text}
            Priority: {pred_label}
            Suggestion: {suggestion}
            Due in: {days_left} day(s)
            """
            msg.attach(MIMEText(body, 'plain'))
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(gmail_user, gmail_pass)
            server.send_message(msg)
            server.quit()
            st.success("📧 Email reminder sent!")
        except Exception as e:
            st.error(f"❌ Failed to send email: {e}")

elif submit:
    st.warning("Please enter a task description.")

# --- Task Dashboard ---
st.markdown("## 📋 Task Dashboard")
if not st.session_state.df.empty:
    # Filtering
    priority_filter = st.selectbox("Filter by Priority", options=["All", "High", "Medium", "Low"])
    df_display = st.session_state.df
    if priority_filter != "All":
        df_display = df_display[df_display["Priority"] == priority_filter]

    st.dataframe(df_display)

    # Edit/Delete Section
    st.markdown("### ✏️ Edit or ❌ Delete a Task")
    index_list = df_display.index.tolist()
    if index_list:
        selected_index = st.selectbox("Select task index", index_list)
        selected_row = df_display.loc[selected_index]

        # Use .get with default to avoid KeyErrors
        new_title = st.text_input("Edit Title", selected_row.get("Title", ""), key='edit_title')
        new_desc = st.text_area("Edit Description", selected_row.get("Description", ""), key='edit_desc')
        new_due = st.date_input("Edit Due Date", datetime.datetime.strptime(str(selected_row.get("Due Date", date.today())), '%Y-%m-%d').date(), key='edit_due')
        new_email = st.text_input("Edit Email", selected_row.get("Email", ""), key='edit_email')
        action = st.radio("Action", ["Edit", "Delete"], horizontal=True)
        if st.button("✅ Apply"):
            if action == "Edit":
                # Recalculate
                vec = tfidf.transform([new_desc])
                pred = model.predict(vec)[0]
                pred_label = le.inverse_transform([pred])[0]
                icon = icon_dict[pred_label]
                suggestion = suggestion_dict[pred_label]
                days_left = (new_due - date.today()).days

                # Update values
                st.session_state.df.loc[selected_index] = [new_title, new_desc, new_due, pred_label, suggestion, days_left, new_email]
                st.success("✅ Task updated successfully!")
            elif action == "Delete":
                st.session_state.df = st.session_state.df.drop(index=selected_index).reset_index(drop=True)
                st.success("🗑️ Task deleted.")
            # Save to CSV
            st.session_state.df.to_csv(csv_path, index=False)
else:
    st.info("No tasks added yet.")


