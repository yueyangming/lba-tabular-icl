# Dataset provenance

The study uses 14 existing public tabular datasets. No new dataset is redistributed by this repository. Data is downloaded through the OpenML API and cached by the user's local OpenML installation.

The canonical names, OpenML dataset IDs, and task types are recorded in [`configs/datasets.yaml`](../configs/datasets.yaml). Using numeric dataset IDs avoids ambiguity between datasets that share a name but have different OpenML versions.

## Split protocol

Dataset-specific source, citation, and license metadata are available on the following OpenML records. The repository does not assign its own license to these datasets or redistribute model weights.

| Dataset | OpenML record |
| --- | --- |
| California Housing | [43939](https://www.openml.org/d/43939) |
| Wine Quality | [287](https://www.openml.org/d/287) |
| Concrete Compressive Strength | [44959](https://www.openml.org/d/44959) |
| Energy Efficiency | [43918](https://www.openml.org/d/43918) |
| kin8nm | [189](https://www.openml.org/d/189) |
| Diabetes | [37](https://www.openml.org/d/37) |
| Adult | [1590](https://www.openml.org/d/1590) |
| German Credit | [31](https://www.openml.org/d/31) |
| Phoneme | [1489](https://www.openml.org/d/1489) |
| Vehicle | [54](https://www.openml.org/d/54) |
| Segment | [36](https://www.openml.org/d/36) |
| Page Blocks | [30](https://www.openml.org/d/30) |
| Satimage | [182](https://www.openml.org/d/182) |
| Letter | [6](https://www.openml.org/d/6) |

The standard split protocol is:

- Training: 60%
- Validation: 20%
- Test: 20%
- Default random seed: 42
- Classification splits are stratified.
- Returned data frames and targets are re-indexed after splitting.

The implementation is in `rail/data/openml_loader.py` and `rail/data/split.py`.

Users should review each OpenML dataset page for its original source, citation, license, and usage restrictions. Derived row indices and predictions in the released artifacts refer only to these public datasets; raw feature tables are not included.
