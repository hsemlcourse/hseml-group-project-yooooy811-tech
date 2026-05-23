"""FastAPI сервис для предсказания задержки доставки."""

import json

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Delivery Delay Prediction")

model = joblib.load("models/xgb_tuned.pkl")
with open("models/columns.json") as f:
    MODEL_COLUMNS = json.load(f)


class OrderInput(BaseModel):
    """Входные данные заказа."""

    shipping_mode: str = "Standard Class"
    days_for_shipment_scheduled: int = 4
    order_status: str = "PENDING"
    payment_type: str = "DEBIT"
    customer_segment: str = "Consumer"
    market: str = "Europe"
    order_region: str = "Western Europe"
    department_name: str = "Fan Shop"
    category_name: str = "Sporting Goods"
    order_item_discount: float = 10.0
    order_item_discount_rate: float = 0.05
    order_item_product_price: float = 100.0
    order_item_profit_ratio: float = 0.2
    order_item_quantity: int = 2
    sales: float = 200.0
    sales_per_customer: float = 200.0
    order_profit_per_order: float = 40.0
    latitude: float = 48.0
    longitude: float = 2.0
    order_month: int = 6
    order_dayofweek: int = 2
    order_hour: int = 14


@app.post("/predict")
def predict(order: OrderInput):
    """Предсказание задержки доставки."""
    data = {
        "Days for shipment (scheduled)": order.days_for_shipment_scheduled,
        "Sales per customer": order.sales_per_customer,
        "Latitude": order.latitude,
        "Longitude": order.longitude,
        "Order Item Discount": order.order_item_discount,
        "Order Item Discount Rate": order.order_item_discount_rate,
        "Order Item Product Price": order.order_item_product_price,
        "Order Item Profit Ratio": order.order_item_profit_ratio,
        "Order Item Quantity": order.order_item_quantity,
        "Sales": order.sales,
        "Order Profit Per Order": order.order_profit_per_order,
        "order_month": order.order_month,
        "order_dayofweek": order.order_dayofweek,
        "order_hour": order.order_hour,
    }

    row = pd.DataFrame([data])

    cat_features = {
        "Shipping Mode": order.shipping_mode,
        "Order Status": order.order_status,
        "Type": order.payment_type,
        "Customer Segment": order.customer_segment,
        "Market": order.market,
        "Order Region": order.order_region,
        "Department Name": order.department_name,
        "Category Name": order.category_name,
    }

    for col_name in MODEL_COLUMNS:
        if col_name not in row.columns:
            row[col_name] = 0

    for feature_name, value in cat_features.items():
        col = f"{feature_name}_{value}"
        if col in MODEL_COLUMNS:
            row[col] = 1

    row = row[MODEL_COLUMNS]

    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1])

    return {
        "prediction": prediction,
        "label": "Задержка" if prediction == 1 else "Вовремя",
        "probability": round(probability, 4),
    }


@app.get("/health")
def health():
    return {"status": "ok"}