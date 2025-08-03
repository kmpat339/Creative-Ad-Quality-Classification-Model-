
import argparse, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import tensorflow as tf
from utils import build_dataset, CLASS_NAMES

def main(args):
    model = tf.keras.models.load_model(args.model_path)
    ds = build_dataset(args.data_dir, shuffle=False, batch_size=args.batch_size)
    y_true = np.concatenate([y for _, y in ds], axis=0)
    y_pred_probs = model.predict(ds)
    y_pred = y_pred_probs.argmax(axis=1)

    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    auc = roc_auc_score(y_true, y_pred_probs[:,1])
    print("ROC‑AUC:", auc)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45)
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Saved confusion_matrix.png")

if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Trained model file')
    parser.add_argument('--data_dir', default='data', help='Dataset directory')
    parser.add_argument('--batch_size', type=int, default=32)
    main(parser.parse_args())
