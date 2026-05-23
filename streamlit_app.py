"""Streamlit интерфейс для предсказания задержки доставки."""

import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Delivery Delay Prediction", page_icon="📦")
st.title("Предсказание задержки доставки")
st.write("Введите параметры заказа и узнайте, будет ли доставка задержана.")

model = joblib.load("models/xgb_tuned.pkl")
with open("models/columns.json") as f:
    MODEL_COLUMNS = json.load(f)

st.sidebar.header("Параметры заказа")

shipping_mode = st.sidebar.selectbox("Способ доставки", ["Standard Class", "First Class", "Second Class", "Same Day"])
days_scheduled = st.sidebar.slider("Плановых дней доставки", 0, 4, 3)
order_status = st.sidebar.selectbox("Статус заказа", ["PENDING", "PROCESSING", "COMPLETE", "CLOSED", "CANCELED", "SUSPECTED_FRAUD", "ON_HOLD", "PAYMENT_REVIEW", "PENDING_PAYMENT"])
payment_type = st.sidebar.selectbox("Тип оплаты", ["DEBIT", "TRANSFER", "CASH", "PAYMENT"])
customer_segment = st.sidebar.selectbox("Сегмент клиента", ["Consumer", "Corporate", "Home Office"])
market = st.sidebar.selectbox("Рынок", ["Europe", "LATAM", "Pacific Asia", "USCA", "Africa"])
order_region = st.sidebar.selectbox("Регион", ["Western Europe", "Central America", "South America", "Eastern Asia", "Southern Asia", "Oceania", "Eastern Europe", "West Africa", "Central Africa", "North Africa", "West Asia", "Southern Europe", "South of USA", "Canada", "Northern Europe", "Caribbean", "Southeast Asia", "Eastern Africa", "Central Asia", "US Center", "West of USA", "North Asia", "East of USA"])
department = st.sidebar.selectbox("Отдел", ["Fan Shop", "Apparel", "Golf", "Outdoors", "Technology", "Fitness", "Footwear", "Fanshop", "Book Shop", "Pet Shop", "Health and Beauty"])
category = st.sidebar.selectbox("Категория", ["Sporting Goods", "Cleats", "Electronics", "Cameras ", "Computers", "Accessories", "Women's Clothing", "Men's Clothing", "Children's Clothing", "Water Sports", "Indoor/Outdoor Games", "Cardio Equipment", "Fishing", "Camping & Hiking"])

st.sidebar.header("Параметры товара")
product_price = st.sidebar.number_input("Цена товара ($)", min_value=10, max_value=2000, value=100)
quantity = st.sidebar.slider("Количество", 1, 5, 1)
discount = st.sidebar.number_input("Скидка ($)", min_value=0, max_value=500, value=10)
discount_rate = st.sidebar.slider("Ставка скидки", 0.0, 0.25, 0.05)
profit_ratio = st.sidebar.slider("Маржинальность", -2.75, 0.50, 0.10)
sales = product_price * quantity
profit = sales * profit_ratio

st.sidebar.header("Дата и место")
order_month = st.sidebar.slider("Месяц заказа", 1, 12, 6)
order_dayofweek = st.sidebar.slider("День недели (0=Пн)", 0, 6, 2)
order_hour = st.sidebar.slider("Час заказа", 0, 23, 14)
latitude = st.sidebar.number_input("Широта", min_value=-34.0, max_value=49.0, value=48.0)
longitude = st.sidebar.number_input("Долгота", min_value=-158.0, max_value=115.0, value=2.0)

if st.button("🔍 Предсказать", type="primary"):
    data = {
        "Days for shipment (scheduled)": days_scheduled,
        "Sales per customer": sales,
        "Latitude": latitude,
        "Longitude": longitude,
        "Order Item Discount": discount,
        "Order Item Discount Rate": discount_rate,
        "Order Item Product Price": product_price,
        "Order Item Profit Ratio": profit_ratio,
        "Order Item Quantity": quantity,
        "Sales": sales,
        "Order Profit Per Order": profit,
        "order_month": order_month,
        "order_dayofweek": order_dayofweek,
        "order_hour": order_hour,
    }

    row = pd.DataFrame([data])

    cat_features = {
        "Shipping Mode": shipping_mode,
        "Order Status": order_status,
        "Type": payment_type,
        "Customer Segment": customer_segment,
        "Market": market,
        "Order Region": order_region,
        "Department Name": department,
        "Category Name": category,
    }

    for col_name in MODEL_COLUMNS:
        if col_name not in row.columns:
            row[col_name] = 0

    for feature_name, value in cat_features.items():
        col = f"{feature_name}_{value}"
        if col in MODEL_COLUMNS:
            row[col] = 1

    row = row[MODEL_COLUMNS]

    prediction = model.predict(row)[0]
    probability = model.predict_proba(row)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"Задержка доставки. (вероятность: {probability:.1%})")
    else:
        st.success(f"Доставка вовремя (вероятность задержки: {probability:.1%})")

    st.metric("Сумма заказа", f"${sales:.2f}")
    st.metric("Прибыль", f"${profit:.2f}")