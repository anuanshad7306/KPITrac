# KPITrac  

**KPITrac** is a business KPI (Key Performance Indicator) monitoring and analysis tool that helps organizations track their performance in real-time as well as from uploaded datasets. The system provides forecasting, anomaly detection, and insightful recommendations to support decision-making.  

---

## 🚀 Features  

- **User Roles**  
  - **Admin**: Manage users and reset passwords via email authentication.  
  - **Analyst**: Access KPIs and insights.  

- **KPI Tracking**  
  - Real-time KPI monitoring  
  - Support for uploaded datasets  
  - Daily, weekly, monthly, and yearly views  

- **Analytics**  
  - Revenue forecasting using Prophet  
  - Anomaly detection on KPIs  
  - Automated recommendations  

- **UI & Theme**  
  - Built with **Streamlit**  
  - Dark orange/brown theme with **Space Grotesk** font  

---

## 🛠️ Tech Stack  

- **Frontend**: Streamlit  
- **Backend**: Flask  
- **Database**: MongoDB  
- **Forecasting**: Prophet  
- **Visualization**: Streamlit & custom plots  

---

## 📂 Project Structure  

```
KPITrac/
│── backend/        # Flask backend APIs
│── frontend/       # Streamlit app
│── models/         # Forecasting & anomaly detection models
│── data/           # Sample datasets
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation & Setup  

1. **Clone the repository**  
   ```bash
   git clone https://github.com/aabhiiih/KPITrac.git
   cd KPITrac
   ```

2. **Create a virtual environment & activate it**  
   ```bash
   python -m venv venv
   source venv/bin/activate   # For Linux/Mac
   venv\Scripts\activate      # For Windows
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the backend**  
   ```bash
   cd backend
   python app.py
   ```

5. **Run the frontend**  
   ```bash
   cd frontend
   streamlit run app.py
   ```

---

## 📊 Example Use Cases  

- Monitor revenue and expense KPIs in real-time  
- Forecast future revenue trends  
- Detect anomalies in daily sales or traffic  
- Provide automated recommendations for business improvement  

---

## 🤝 Contributing  

Contributions are welcome! Please fork this repository and submit a pull request for improvements or new features.  

---

## 📜 License  

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.  

---

## 👥 Authors  

- **Abhinand**  
- **Anshad**  
