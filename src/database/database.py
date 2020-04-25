import re
import numpy as np
import pandas as pd
from sqlalchemy import between, not_, and_, or_, Table, select, create_engine, MetaData


class DataBase(object):
    def __init__(self, db_path):
        self.db_path = db_path
        self.engine = create_engine("sqlite:///{}".format(self.db_path), echo=False)
        self.db_metadata = MetaData(self.engine)
        self.data_table = self.create_table("DATA")
        self.metadata_table = self.create_table("METADATA")

    def __repr__(self):
        name = type(self).__name__
        return "{name}()".format(name=name)

    def create_table(self, table_name):
        return Table(table_name, self.db_metadata, autoload=True)

    def read_data_from_db(self, parameters):
        columns = sum((col.names for col in parameters.columns.values()), [])

        data = self.select_columns_from_data(
            columns=columns, dataset=parameters.dataset, filter=parameters.filter
        )
        return data

    def select_columns_from_data(self, columns, dataset, filter):
        dataset_clause = or_(*[self.data_table.c.dataset == ds for ds in dataset])
        sel_stmt = select([self.data_table.c[col] for col in columns]).where(
            dataset_clause
        )
        if filter:
            filter_clause = self.build_filter_clause(filter)
            sel_stmt = sel_stmt.where(filter_clause)
        data = pd.read_sql(sel_stmt, self.engine)
        data.replace("", np.nan, inplace=True)  # fixme remove
        data = data.dropna()
        return data

    def build_filter_clause(self, filter):
        values = [v for v in filter.values()][0]
        key = [k for k in filter.keys()][0]
        return or_(*[self.data_table.c[key] == val for val in values])

    def select_md_from_metadata(self, var_names, args):
        code_name = args.metadata_code_column
        label_name = args.metadata_label_column
        iscat_name = args.metadata_isCategorical_column
        enums_name = args.metadata_enumerations_column
        min_name = args.metadata_minValue_column
        max_name = args.metadata_maxValue_column
        label = dict()
        is_categorical = dict()
        enumerations = dict()
        minmax = dict()
        sel_stmt = select(
            [
                self.metadata_table.c[code_name],
                self.metadata_table.c[label_name],
                self.metadata_table.c[iscat_name],
                self.metadata_table.c[enums_name],
                self.metadata_table.c[min_name],
                self.metadata_table.c[max_name],
            ]
        ).where(or_(*[self.metadata_table.c[code_name] == vn for vn in var_names]))
        result = self.engine.execute(sel_stmt)
        for vn, la, ic, en, mi, ma in result:
            label[vn] = la
            is_categorical[vn] = ic
            if en:
                enumerations[vn] = re.split(r"\s*,\s*", en)
            minmax[vn] = (mi, ma)
        return label, is_categorical, enumerations, minmax
