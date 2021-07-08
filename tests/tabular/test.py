from potok.core import DataDict, Pipeline
from potok.tabular import TabularData, Folder, Validation, LightGBM, SyntheticData, TransformY
from typing import List, Iterator, Tuple


def generate_regression_data() -> Tuple[DataDict, DataDict]:
    gene = SyntheticData()
    train = gene.create_train()
    test = gene.create_test()
    data = DataDict(train=train, test=test)
    return data.X, data.Y


def test_lightgbm_regression() -> DataDict:
    x, y = generate_regression_data()
    x = DataDict(data_1=x)
    y = DataDict(data_1=y)
    transform = TransformY(transform='square', target='Target')
    folder = Folder(n_folds=3, seed=2424)
    validation = Validation(folder)
    algo = LightGBM(target='Target', features=['X'])
    model = Pipeline(validation, algo, shapes=[1,  3])
    prediction = model.fit_predict(x, y)
    return prediction


if __name__ == "__main__":
    pred = test_lightgbm_regression()
