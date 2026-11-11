"""Evaluate labeled trials with participant-held-out predictions."""
import argparse
import csv
from collections import defaultdict
from statistics import mean, stdev

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from main import FEATURES


def load_trials(paths):
    trials = {}
    for path in paths:
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) < 10:
            raise ValueError(f"{path}: need at least 10 valid samples.")
        first = rows[0]
        key = (first["participant"], first["trial"])
        label = first["label"]
        if not all(key) or label not in ("truth", "lie") or key in trials:
            raise ValueError(f"{path}: invalid or duplicate participant/trial.")
        if any((row["participant"], row["trial"], row["label"]) != (*key, label)
               for row in rows):
            raise ValueError(f"{path}: each file must contain one labeled trial.")
        columns = [[float(row[feature]) for row in rows] for feature in FEATURES]
        if not np.isfinite(columns).all():
            raise ValueError(f"{path}: non-finite measurements.")
        trials[key] = (label, [value for column in columns
                               for value in (mean(column), stdev(column))])
    return trials


def evaluate(trials):
    labels_by_person = defaultdict(set)
    groups, labels, inputs = [], [], []
    for (participant, _), (label, features) in trials.items():
        groups.append(participant)
        labels.append(label)
        inputs.append(features)
        labels_by_person[participant].add(label)
    if len(labels_by_person) < 3 or any(len(labels) != 2 for labels in labels_by_person.values()):
        raise ValueError("Need at least three participants, each with truth and lie trials.")
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))
    predictions = cross_val_predict(model, inputs, labels, groups=groups, cv=LeaveOneGroupOut())
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=["truth", "lie"]).ravel()
    print(f"Participants: {len(labels_by_person)} | Trials: {len(labels)}")
    print(f"Balanced accuracy: {balanced_accuracy_score(labels, predictions):.1%} (baseline: 50%)")
    print(f"Truth called lie: {fp}/{tn + fp} | Lie called truth: {fn}/{fn + tp}")
    labels, groups = np.asarray(labels), np.asarray(groups)
    for person in sorted(labels_by_person):
        mask = groups == person
        print(f"  {person}: {balanced_accuracy_score(labels[mask], predictions[mask]):.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", nargs="+")
    args = parser.parse_args()
    try:
        evaluate(load_trials(args.recordings))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")
