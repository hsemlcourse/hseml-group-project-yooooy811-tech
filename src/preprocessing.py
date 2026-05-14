"""Модуль предобработки данных для задачи предсказания задержки доставки."""

import pandas as pd


SEED = 42

# Колонки для удаления
LEAK_COLS = ["Delivery Status", "Days for shipping (real)"]
PERSONAL_COLS = [
    "Customer Email", "Customer Password",
    "Customer Fname", "Customer Lname", "Customer Street",
]
USELESS_COLS = [
    "Product Image", "Product Description",
    "Order Zipcode", "Customer Zipcode", "Product Status",
]
ID_COLS = [
    "Order Id", "Order Item Id", "Customer Id",
    "Order Customer Id", "Product Card Id",
    "Order Item Cardprod Id", "Category Id",
    "Department Id", "Product Category Id",
]
MULTICOLLINEAR_COLS = ["Product Price", "Benefit per order", "Order Item Total"]
HIGH_CARD_COLS = [
    "Customer City", "Customer State", "Order City",
    "Order Country", "Order State", "Product Name", "Customer Country",
]


def load_data(path: str) -> pd.DataFrame:
    """Загрузка датасета."""
    return pd.read_csv(path, encoding="latin-1")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Очистка: удаление ненужных колонок, создание фичей из дат."""
    drop_cols = LEAK_COLS + PERSONAL_COLS + USELESS_COLS + ID_COLS
    df = df.drop(columns=drop_cols)
    df = df.drop(columns=MULTICOLLINEAR_COLS)

    # Фичи из дат
    df["order_date"] = pd.to_datetime(df["order date (DateOrders)"])
    df["order_month"] = df["order_date"].dt.month
    df["order_dayofweek"] = df["order_date"].dt.dayofweek
    df["order_hour"] = df["order_date"].dt.hour
    df = df.drop(columns=["order date (DateOrders)", "shipping date (DateOrders)", "order_date"])

    # Высококардинальные
    df = df.drop(columns=HIGH_CARD_COLS)

    return df


def prepare_features(df: pd.DataFrame):
    """One-hot encoding и разделение на X, y."""
    df_encoded = pd.get_dummies(df, drop_first=True)
    x = df_encoded.drop(columns="Late_delivery_risk")
    y = df_encoded["Late_delivery_risk"]
    return x, y