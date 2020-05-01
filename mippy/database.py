from typing import List, Mapping, Set, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from addict import Dict
from sqlalchemy import or_, Table, select, create_engine, MetaData

__all__ = ["DataBase"]

root = Path(__file__).parent.parent


class DataBase(object):
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        self.db_metadata = MetaData(self.engine)
        self.data_table = self.create_table("DATA")
        self.metadata_table = self.create_table("METADATA")

    def __repr__(self) -> str:
        name = type(self).__name__
        return "{name}()".format(name=name)

    def create_table(self, table_name: str) -> Table:
        return Table(table_name, self.db_metadata, autoload=True)

    def read_data(self, parameters: Dict) -> pd.DataFrame:
        columns: List[str] = sum((cols for cols in parameters.columns.values()), [])

        data = self.select_columns_from_data(
            columns=columns, datasets=parameters.datasets, filter_=parameters.filter
        )
        return data

    def select_columns_from_data(
        self, columns: List[str], datasets: List[str], filter_: Optional[Mapping]
    ) -> pd.DataFrame:
        dataset_clause = or_(*[self.data_table.c.dataset == ds for ds in datasets])
        sel_stmt = select([self.data_table.c[col] for col in columns]).where(
            dataset_clause
        )
        if filter_:
            filter_clause = self.build_filter_clause(filter_)
            sel_stmt = sel_stmt.where(filter_clause)
        data = pd.read_sql(sel_stmt, self.engine)
        data.replace("", np.nan, inplace=True)  # fixme remove
        data = data.dropna()
        return data

    def build_filter_clause(self, filter_: Mapping):
        values = [v for v in filter_.values()][0]
        key = [k for k in filter_.keys()][0]
        return or_(*[self.data_table.c[key] == val for val in values])

    def get_datasets(self) -> Set[str]:
        stmt = select([self.data_table.c.dataset])
        res = self.engine.execute(stmt)
        datasets = set()
        for row in res:
            datasets.add(str(row[0]))
        return datasets
