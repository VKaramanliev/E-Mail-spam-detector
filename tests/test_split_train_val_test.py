from __future__ import annotations

from spam_detector.train import split_train_val_test


def test_split_train_val_test_sizes_and_no_overlap():
    texts = [f"t{i}" for i in range(40)]
    labels = (["ham"] * 20) + (["spam"] * 20)

    split = split_train_val_test(texts, labels, random_state=42, test_size=0.15, val_size=0.15)

    s_train = set(split.x_train)
    s_val = set(split.x_val)
    s_test = set(split.x_test)

    assert s_train.isdisjoint(s_val)
    assert s_train.isdisjoint(s_test)
    assert s_val.isdisjoint(s_test)

    assert len(s_train) + len(s_val) + len(s_test) == len(texts)

    assert "ham" in split.y_train and "spam" in split.y_train
    assert "ham" in split.y_val and "spam" in split.y_val
    assert "ham" in split.y_test and "spam" in split.y_test