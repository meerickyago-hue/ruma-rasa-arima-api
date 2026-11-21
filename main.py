import io
import json
import os
import warnings
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

app = FastAPI(
    title="API Peramalan Penjualan Restoran Ruma Rasa (ARIMA)",
    description="Backend RESTful API menggunakan model ARIMA(1,1,0) tervalidasi untuk 3 menu andalan.",
    version="2.0.0",
)

# Izinkan akses lintas domain dari frontend Hostinger
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Konfigurasi resmi hasil eksperimen Google Colab
MENU_CONFIG = {
    "Nasi Goreng Kampung": {
        "order": (1, 1, 0),
        "harga_satuan": 25000,
        "model_file": "saved_models_arima/arima_model_nasi_goreng_kampung.pkl",
    },
    "Ice Lychee Tea": {
        "order": (1, 1, 0),
        "harga_satuan": 18000,
        "model_file": "saved_models_arima/arima_model_ice_lychee_tea.pkl",
    },
    "Spaghetti Aglio Olio": {
        "order": (1, 1, 0),
        "harga_satuan": 32000,
        "model_file": "saved_models_arima/arima_model_spaghetti_aglio_olio.pkl",
    },
}


def klasifikasi_mape(mape_val: float) -> str:
    if mape_val < 10:
        return "Sangat Baik (Highly Accurate)"
    elif mape_val <= 20:
        return "Baik (Good)"
    elif mape_val <= 50:
        return "Layak (Reasonable)"
    return "Buruk (Inaccurate)"


@app.get("/")
def health_check():
    metadata_loaded = os.path.exists("saved_models_arima/model_metadata.json")
    return {
        "status": "online",
        "service": "Ruma Rasa ARIMA Forecasting API",
        "supported_menus": list(MENU_CONFIG.keys()),
        "colab_artifacts_synced": metadata_loaded,
    }


@app.get("/api/model-info")
def get_model_info():
    meta_path = "saved_models_arima/model_metadata.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"menus": MENU_CONFIG}


@app.post("/api/predict-arima")
async def predict_arima(
    file: UploadFile = File(...),
    menu_name: str = Form("Nasi Goreng Kampung"),
    forecast_steps: int = Form(1),
):
    # 1. Validasi pilihan menu & horizon prediksi
    if menu_name not in MENU_CONFIG:
        raise HTTPException(
            status_code=400,
            detail=f"Menu '{menu_name}' tidak valid. Pilih salah satu: {list(MENU_CONFIG.keys())}",
        )

    if forecast_steps < 1 or forecast_steps > 24:
        raise HTTPException(
            status_code=400,
            detail="Periode peramalan harus berada di antara 1 hingga 24 bulan.",
        )

    filename = (file.filename or "").lower()
    if not (filename.endswith(".xlsx") or filename.endswith(".xls") or filename.endswith(".csv")):
        raise HTTPException(
            status_code=400,
            detail="Format berkas tidak didukung. Unggah berkas Excel (.xlsx/.xls) atau CSV (.csv).",
        )

    # 2. Pembacaan berkas dari memori
    try:
        contents = await file.read()
        buffer = io.BytesIO(contents)

        if filename.endswith(".csv"):
            df = pd.read_csv(buffer)
        else:
            xls = pd.ExcelFile(buffer)
            if menu_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=menu_name)
            else:
                df = pd.read_excel(xls, sheet_name=0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal membaca isi berkas: {str(e)}")

    # 3. Validasi struktur kolom & integritas data
    required_cols = {"Tanggal_Periode", "Qty_Terjual"}
    if not required_cols.issubset(set(df.columns)):
        raise HTTPException(
            status_code=400,
            detail=f"Kolom wajib tidak ditemukan. Pastikan terdapat kolom: {list(required_cols)}",
        )

    df = df.dropna(subset=["Tanggal_Periode", "Qty_Terjual"]).copy()
    try:
        df["Tanggal_Periode"] = pd.to_datetime(df["Tanggal_Periode"])
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Format kolom 'Tanggal_Periode' tidak valid. Gunakan format tanggal standar (YYYY-MM-DD).",
        )

    df["Qty_Terjual"] = pd.to_numeric(df["Qty_Terjual"], errors="coerce")
    df = df.dropna(subset=["Qty_Terjual"])

    if (df["Qty_Terjual"] < 0).any():
        raise HTTPException(
            status_code=400,
            detail="Ditemukan nilai kuantitas terjual negatif. Periksa kembali integritas data penjualan.",
        )

    df = df.sort_values("Tanggal_Periode").drop_duplicates(subset=["Tanggal_Periode"])
    df.set_index("Tanggal_Periode", inplace=True)
    df = df.asfreq("MS")
    df["Qty_Terjual"] = df["Qty_Terjual"].interpolate(method="linear")

    if len(df) < 12:
        raise HTTPException(
            status_code=400,
            detail=f"Jumlah data historis terlalu sedikit ({len(df)} bulan). Minimal dibutuhkan 12 periode bulanan.",
        )

    # 4. Ambil konfigurasi model & harga satuan spesifik menu
    cfg = MENU_CONFIG[menu_name]
    order = cfg["order"]
    harga_satuan = (
        float(df["Harga_Satuan_Rp"].iloc[0])
        if "Harga_Satuan_Rp" in df.columns and pd.notna(df["Harga_Satuan_Rp"].iloc[0])
        else float(cfg["harga_satuan"])
    )

    series_qty = df["Qty_Terjual"].astype("float64")

    # 5. Evaluasi Out-of-Sample Kronologis 80:20 (sama persis dengan eksperimen Colab)
    train_size = int(round(len(series_qty) * 0.8))
    train_s = series_qty.iloc[:train_size]
    test_s = series_qty.iloc[train_size:]

    try:
        eval_model = ARIMA(train_s, order=order).fit()
        fc_test = eval_model.forecast(steps=len(test_s))

        mape_val = float(mean_absolute_percentage_error(test_s, fc_test) * 100)
        mae_val = float(mean_absolute_error(test_s, fc_test))
        rmse_val = float(np.sqrt(mean_squared_error(test_s, fc_test)))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal mengevaluasi model ARIMA: {str(e)}")

    # 6. Fitting Ulang Penuh pada seluruh data historis untuk proyeksi ke depan
    try:
        full_fit = ARIMA(series_qty, order=order).fit()
        forecast_res = full_fit.get_forecast(steps=forecast_steps)
        pred_mean = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menghasilkan proyeksi ARIMA: {str(e)}")

    # Susun data historis untuk grafik
    historical_data = [
        {
            "periode": idx.strftime("%Y-%m-%d"),
            "label": idx.strftime("%b %Y"),
            "qty": int(round(val)),
            "omzet_rp": int(round(val) * harga_satuan),
        }
        for idx, val in series_qty.items()
    ]

    # Susun data proyeksi periode mendatang
    forecast_data = []
    for i in range(forecast_steps):
        fc_date = pred_mean.index[i]
        qty_pred = max(0, int(round(float(pred_mean.iloc[i]))))
        ci_low = max(0, int(round(float(conf_int.iloc[i, 0]))))
        ci_high = max(0, int(round(float(conf_int.iloc[i, 1]))))
        omzet_pred = int(qty_pred * harga_satuan)

        forecast_data.append(
            {
                "periode": fc_date.strftime("%Y-%m-%d"),
                "label": fc_date.strftime("%b %Y"),
                "prediksi_qty": qty_pred,
                "batas_bawah_95": ci_low,
                "batas_atas_95": ci_high,
                "estimasi_omzet_rp": omzet_pred,
            }
        )

    return {
        "status": "success",
        "menu": menu_name,
        "model_order": f"ARIMA{order}",
        "harga_satuan_rp": int(harga_satuan),
        "total_periode_historis": len(series_qty),
        "split_info": {
            "train_months": len(train_s),
            "test_months": len(test_s),
        },
        "metrics": {
            "mape": round(mape_val, 2),
            "mae": round(mae_val, 2),
            "rmse": round(rmse_val, 2),
            "kategori": klasifikasi_mape(mape_val),
            "aic": round(float(full_fit.aic), 2),
            "bic": round(float(full_fit.bic), 2),
        },
        "historical": historical_data,
        "forecast": forecast_data,
    }
