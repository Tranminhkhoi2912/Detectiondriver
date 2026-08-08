"""
scripts/train_model.py
------------------------
HUẤN LUYỆN MODEL MACHINE LEARNING (bước 2 của "tự train model").

Huấn luyện model phân loại trạng thái tài xế (AWAKE / EYE_CLOSED / YAWNING /
HEAD_DROP) từ 3 đặc trưng hình học (EAR, MAR, pitch_dev), dựa trên dữ liệu
đã thu thập THỦ CÔNG ở scripts/collect_data.py (nhãn do người dùng tự gán,
không phải nhãn sinh ra từ rule-based).

Model: RandomForestClassifier (scikit-learn). Lý do chọn:
    - Dữ liệu chỉ có 3 chiều đặc trưng, không cần mạng neural sâu
    - Ít nhạy cảm với việc chuẩn hóa dữ liệu, robust với nhiễu
    - Có thể diễn giải qua feature_importance_ - hữu ích khi viết báo cáo
    - Huấn luyện nhanh (vài giây) ngay cả trên máy yếu, không cần GPU

Sau khi chạy xong, script sẽ:
    1. In classification report (precision/recall/f1-score từng lớp)
    2. In điểm cross-validation 5-fold (đánh giá độ ổn định của model)
    3. Vẽ & lưu confusion matrix        -> models/confusion_matrix.png
    4. Vẽ & lưu biểu đồ feature importance -> models/feature_importance.png
    5. Lưu model đã train              -> models/drowsiness_classifier.pkl

Sau khi có file .pkl, chỉ cần chạy lại python main.py (với config.USE_ML_MODEL
= True, mặc định đã bật) là hệ thống sẽ tự động dùng model này.
"""

import os
import sys

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # chỉ lưu file ảnh, không cần mở cửa sổ hiển thị
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


FEATURE_NAMES = ["ear", "mar", "pitch_dev"]


def main():
    if not os.path.exists(config.TRAINING_DATA_PATH):
        print(f"[Lỗi] Chưa có dữ liệu huấn luyện tại: {config.TRAINING_DATA_PATH}")
        print("Hãy chạy trước: python scripts/collect_data.py")
        return

    df = pd.read_csv(config.TRAINING_DATA_PATH)
    print(f"[Dữ liệu] Đã tải {len(df)} dòng từ {config.TRAINING_DATA_PATH}")
    print("[Dữ liệu] Phân bố nhãn:")
    print(df["label"].value_counts().to_string())

    label_counts = df["label"].value_counts()
    if len(df) < 100:
        print("\n[Cảnh báo] Dữ liệu khá ít (<100 dòng) - model có thể không đáng tin cậy.")
    if (label_counts < 30).any():
        thin_labels = label_counts[label_counts < 30].index.tolist()
        print(f"[Cảnh báo] Các nhãn sau có quá ít mẫu (<30): {thin_labels}")
        print("           Nên thu thập thêm bằng scripts/collect_data.py trước khi tin dùng model.")

    if df["label"].nunique() < 2:
        print("[Lỗi] Cần ít nhất 2 nhãn khác nhau để huấn luyện classifier. Dừng lại.")
        return

    X = df[FEATURE_NAMES].values
    y_raw = df["label"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    # stratify=y giữ tỉ lệ các lớp đồng đều giữa tập train/test
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        # Trường hợp có lớp quá ít mẫu để stratify được
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    # ---------- Đánh giá ----------
    y_pred = model.predict(X_test)
    print("\n" + "=" * 70)
    print("KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST (20% dữ liệu, chưa từng thấy lúc train)")
    print("=" * 70)
    present_labels = sorted(set(y_test) | set(y_pred))
    print(classification_report(
        y_test, y_pred,
        labels=present_labels,
        target_names=encoder.inverse_transform(present_labels),
        zero_division=0,
    ))

    try:
        cv_scores = cross_val_score(model, X, y, cv=min(5, label_counts.min()))
        print(f"Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f} "
              f"(trên {len(cv_scores)} fold)")
    except ValueError as e:
        print(f"[Bỏ qua cross-validation] Không đủ mẫu cho một số lớp: {e}")

    # ---------- Confusion matrix ----------
    os.makedirs(os.path.dirname(config.ML_MODEL_PATH), exist_ok=True)
    cm = confusion_matrix(y_test, y_pred, labels=present_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.inverse_transform(present_labels))
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix - Drowsiness Classifier")
    plt.tight_layout()
    cm_path = os.path.join(os.path.dirname(config.ML_MODEL_PATH), "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"\n[Lưu] Confusion matrix -> {cm_path}")

    # ---------- Feature importance ----------
    importances = model.feature_importances_
    fig, ax = plt.subplots(figsize=(6, 4))
    order = np.argsort(importances)
    ax.barh(np.array(FEATURE_NAMES)[order], importances[order], color="steelblue")
    ax.set_xlabel("Mức độ quan trọng (feature importance)")
    ax.set_title("Feature Importance - Random Forest")
    plt.tight_layout()
    fi_path = os.path.join(os.path.dirname(config.ML_MODEL_PATH), "feature_importance.png")
    plt.savefig(fi_path, dpi=150)
    plt.close(fig)
    print(f"[Lưu] Feature importance -> {fi_path}")
    for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1]):
        print(f"    {name}: {imp:.3f}")

    # ---------- Lưu model ----------
    bundle = {
        "model": model,
        "label_encoder": encoder,
        "feature_names": FEATURE_NAMES,
    }
    joblib.dump(bundle, config.ML_MODEL_PATH)
    print(f"\n[Lưu] Model đã huấn luyện -> {config.ML_MODEL_PATH}")
    print("Bây giờ chạy 'python main.py' để dùng model này (config.USE_ML_MODEL = True mặc định).")


if __name__ == "__main__":
    main()
