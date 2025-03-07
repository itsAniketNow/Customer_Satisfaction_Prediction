# Customer Satisfaction Prediction

## 📌 Project Overview

This project aims to predict **customer satisfaction ratings** based on various features extracted from customer support tickets, purchase history, and demographic details. The dataset contains information about **support ticket resolutions, ticket priority, purchase details, and customer interactions** with the support system.

By applying **Machine Learning (ML)** techniques, we analyze the impact of different factors on customer satisfaction and develop a model that can help businesses improve their service quality and customer experience.

---

## 📂 Dataset Description

The dataset includes the following key columns:

### 🎯 **Target Variable**

- `customer_satisfaction_rating`: The rating provided by the customer after their issue is resolved (numerical: 1-5 scale).

### 📊 **Feature Variables**

- **Customer Information:** `customer_id`, `customer_age`, `customer_gender`, `location`
- **Purchase Details:** `product_purchased`, `date_of_purchase`
- **Support Ticket Information:** `ticket_id`, `ticket_type`, `ticket_subject`, `ticket_description`, `ticket_channel`, `ticket_status`, `ticket_priority`, `first_response_time`, `time_to_resolution`, `resolution`
- **Customer Interaction Data:** `number_of_tickets_raised`, `customer_engagement_score`, `previous_satisfaction_score`

---

## ⚙️ Data Preprocessing & Feature Engineering

1. **Handling Missing Values**
   - Imputed missing values using **mean/mode imputation** for numerical and categorical data.

2. **Handling Categorical Data**
   - `product_purchased` is **label-encoded**.
   - `ticket_type`, `ticket_subject`, and `ticket_channel` are **one-hot encoded**.
   - `ticket_status` and `ticket_priority` are **ordinal encoded** with predefined mappings.

3. **DateTime Feature Extraction**
   - `date_of_purchase` is converted into `year, month, day, and weekday`.
   - `first_response_time` and `time_to_resolution` are transformed similarly.

4. **Feature Selection**
   - Dropped columns: `ticket_id` (irrelevant), `ticket_description`, `resolution` (text-heavy, unstructured data).

5. **Train-Test Split**
   - The dataset is split into **80% training** and **20% testing** using `train_test_split()`.

---

## 🚀 Model Training & Evaluation

### **Machine Learning Algorithm Used**

- **Random Forest Regressor** (`RandomForestRegressor(n_estimators=100, random_state=42)`) is used due to its ability to handle numerical and categorical features effectively.
- **Support Vector Regression (SVR)** was also tested but performed slightly worse.
- **Linear Regression** was used as a baseline model.

### **Feature Importance Analysis**

Using **feature importance** from Random Forest, the most impactful variables on customer satisfaction are:

✅ `customer_age`
✅ `product_purchased`
✅ `purchase_day`, `purchase_month`, `purchase_weekday`
✅ `ticket_priority`
✅ `customer_gender`
✅ `ticket_status`
✅ `ticket_channel`
✅ `ticket_subject`
✅ `customer_engagement_score`
✅ `number_of_tickets_raised`

These insights can help businesses prioritize support strategies for **specific customer segments and ticket types**.

---

## 📈 Model Performance Metrics

The model is evaluated using:

| Model | MAE | MSE | RMSE | R² Score |
|--------|------|------|------|----------|
| Random Forest Regressor | 0.58 | 0.47 | 0.68 | 0.82 |
| Support Vector Regression | 0.72 | 0.61 | 0.78 | 0.76 |
| Linear Regression | 0.91 | 0.82 | 0.90 | 0.68 |

Results indicate that the **Random Forest Regressor** performed best with an **R² score of 0.82**, meaning it explains 82% of the variance in customer satisfaction ratings.

---

## 🛠️ Tools & Technologies Used

- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Jupyter Notebook / Google Colab** for execution
- **GitHub** for version control

---

## 🛠️ How to Run the Project (Google Colab)

### 1️⃣ **Clone the Repository**

```bash
git clone https://github.com/your-username/customer-satisfaction-prediction.git
cd customer-satisfaction-prediction
```

### 2️⃣ **Open the Jupyter Notebook in Google Colab**

- Upload the `.ipynb` notebook to [Google Colab](https://colab.research.google.com/).
- Ensure all necessary Python packages are installed:

```python
!pip install pandas numpy scikit-learn seaborn matplotlib
```

### 3️⃣ **Run the Notebook Cells**

- Follow the step-by-step sections in the notebook to process data, train the model, and evaluate results.

---

## 📌 Future Enhancements

🔹 Hyperparameter tuning using **GridSearchCV** or **RandomizedSearchCV**
🔹 Experimenting with **XGBoost, LightGBM** for better accuracy
🔹 Deploying the model as a **REST API using Flask/Django**
🔹 Creating a dashboard in **Streamlit or Power BI** for visualization
🔹 Incorporating **sentiment analysis on customer feedback** for better predictions

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork, improve, and submit a **Pull Request**.

---

## 🏷️ Contact

👤 **Aniket Surwade**\
📧 [mailbox.as.aniketsurwade@gmail.com]()\
🔗 [[LinkedIn](https://www.linkedin.com/in/aniket-surwade/)]


---

🔥 **If you found this useful, don't forget to ⭐ the repository!** 🚀

