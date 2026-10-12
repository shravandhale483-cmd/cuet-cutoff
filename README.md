# University Cutoff Prediction App 🎓

A Streamlit application to deploy a trained machine learning model for university cutoff predictions.

## 🛠️ Local Installation & Running
1. Ensure you have Python 3.9+ installed.
2. Install the necessary packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## 🚀 Deployment to Streamlit Community Cloud
1. Create a public repository on [GitHub](https://github.com).
2. Push these files to your repository:
   - `app.py`
   - `requirements.txt`
   - `model.joblib`
   - `.gitignore`
   - `README.md`
3. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
4. Click **New app**, select your GitHub repository, specify the branch (usually `main`), set the Main file path to `app.py`, and click **Deploy**!