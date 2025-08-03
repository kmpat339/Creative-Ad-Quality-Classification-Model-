
import argparse, os, pathlib
import tensorflow as tf
from utils import build_dataset, build_model, CLASS_NAMES

def main(args):
    train_ds = build_dataset(args.data_dir, batch_size=args.batch_size)
    val_ds = build_dataset(args.data_dir, shuffle=False, batch_size=args.batch_size)

    model = build_model(num_classes=len(CLASS_NAMES))
    ckpt_dir = pathlib.Path('models')
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / 'best_model.h5'

    cb = tf.keras.callbacks.ModelCheckpoint(str(ckpt_path), save_best_only=True,
                                            monitor='val_accuracy', mode='max')
    model.fit(train_ds, epochs=args.epochs, validation_data=val_ds, callbacks=[cb])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data', help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    main(parser.parse_args())
