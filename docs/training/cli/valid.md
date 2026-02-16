# Validating a Trained OCR Model

The `valid` command is used to **evaluate a trained license plate OCR model** on a labeled dataset, using the same
configuration and preprocessing setup used during training.

---

## Basic Usage

```shell
fast-plate-ocr valid \
  --model trained_models/2025-06-28_14-33-51/best.keras \
  --plate-config-file config/latin_plates.yaml \
  --annotations data/val.csv
```

---

## Output

- **Evaluation metrics** will be printed to the terminal (e.g., accuracy, loss).
- The script automatically compiles the model using the metrics **defined during training**.
- It **does not save new weights** or modify the model.
- If the model includes a **region head**, region metrics like `region_acc`, `region_top3_acc`, and `region_macro_f1` will also appear.

---

## Example Output

```
40/40 ━━━━━━━━━━━━━━━━━━━━ 10s 100ms/step - loss: 0.2185 - plate_acc: 0.921 - cat_acc: 0.983 - top_3_k: 0.998
```
