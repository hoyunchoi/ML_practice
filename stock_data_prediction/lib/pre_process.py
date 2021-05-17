import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import utils

"""
    class pre_process

    (public) Members
    stocks : Name of stocks
    df : pandas dataframe of stock data
    col_name_list : column name list of self.df

    (public) Methods
    ---------------------------------------------------------------------
    check_missing_value(verbose) : Check the missing value(NaN) of raw stock data
    Args:
        verbose: If True, print number of Nan and location
    Return:
        Number of NaN
    ---------------------------------------------------------------------
    fill_missing_value() : Fill the missing value(NaN) by the average of non-NaN value at the same date
    Args:
        None
    Return:
        None
    ---------------------------------------------------------------------
    drop_column(col_list) : Drop the column at col_list of self.df
    Args:
        col_list: list of column name to drop
    Return:
        None
    ---------------------------------------------------------------------
    plot(self, col_list=None, save_path=None) : Plot data of self.df
    Args:
        col_list: list of column name to plot. If None, plot every columns available
        save_path : If given, save the plot to the save_path
    Return:
        None
    ---------------------------------------------------------------------
    show_corr(self, save_path=None) : Plot correlation between columns
    Args:
        save_path : If given, save the plot to the save_path
    Return:
        None
    ---------------------------------------------------------------------
    show_auto_corr(self, col_list=None, save_path=None) : Plot auto correlation of each columns
    Args:
        col_list : List of column name to get auto correlation. If None, plot every columns available
        save_path : If given, save the plot to the save_path
    Return:
        None
    ---------------------------------------------------------------------
    save(self, file_name=None, fmt='parquet', comp='snappy', verbose=True) : Save processed data at "preprocessed_data"
    Args:
        file_name : Name of file to be stored. If not given, use default name. EX) IBM.parquet.snappy
        fmt : Format of saved file
        comp : Compression method
        verbose : If true, print the success message
    Return:
        None
    ---------------------------------------------------------------------
    print, head, dtype : Pass each function to self.df
"""


class pre_process():
    def __init__(self, stocks, precision='32'):
        self.stocks = stocks
        self.precision = precision
        self.np_float_dtype = getattr(np, 'float' + precision)
        self.np_int_dtype = getattr(np, 'int' + precision)

        #* Read csv and store into self.df
        file_name = os.path.join("data", stocks + "_2006-01-01_to_2018-01-01.csv")
        self.df = pd.read_csv(file_name,
                              index_col='Date',
                              parse_dates=['Date'],
                              dtype={"Open": self.np_float_dtype,
                                     "High": self.np_float_dtype,
                                     "Low": self.np_float_dtype,
                                     "Close": self.np_float_dtype,
                                     "Volume": self.np_int_dtype})
        self.col_name_list = list(self.df.columns)

    def check_missing_value(self, verbose=True):
        self.missing_loc, self.missing_num = {}, 0
        for col in self.col_name_list:
            try:
                idx_list = np.where(np.isnan(self.df[col]))[0]
                for idx in idx_list:
                    self.missing_num += 1
                    if idx in self.missing_loc:
                        self.missing_loc[idx].append(col)
                    else:
                        self.missing_loc[idx] = [col]
            except TypeError:  # * For column 'Name'
                continue

        if verbose:
            print("Total number of missing data in {}:".format(self.stocks), self.missing_num)
            if self.missing_num:
                for idx, col_name in self.missing_loc.items():
                    print("{}:".format(self.df.index[idx].date()), ', '.join(col_name))
        return self.missing_num

    def fill_missing_value(self):
        for idx in self.missing_loc:
            average = np.nanmean(self.df.iloc[idx].values)
            self.df.iloc[idx] = self.df.iloc[idx].fillna(average)

    def drop_column(self, col_list):
        self.col_name_list = [col for col in self.col_name_list if col not in col_list]
        self.df = self.df.loc[:, self.col_name_list]

    def plot(self, col_list=None, save_path=None):
        if not col_list:
            col_list = self.col_name_list

        num_cols = len(col_list)
        fig = plt.figure(figsize=(12, 3 * num_cols))

        for i in range(num_cols):
            ax = plt.subplot(num_cols, 1, i + 1)
            self.df[col_list[i]].plot(ax=ax)
            ax.set_title(col_list[i], fontsize=20)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path)
        else:
            fig.show()

    def show_corr(self, save_path=None):
        fig, ax = plt.subplots(figsize=(5, 5))
        corr = self.df[self.col_name_list].corr()
        mask = np.identity(len(corr))
        sns.heatmap(corr, ax=ax, annot=True, mask=mask, cmap='coolwarm', vmin=-1, vmax=1)
        if save_path:
            fig.savefig(save_path)
        else:
            fig.show()

    def show_auto_corr(self, col_list=None, save_path=None):
        self.auto_corr = {}
        if not col_list:
            col_list = self.col_name_list

        num_cols = len(col_list)
        lag_range = np.arange(1, 101, 1)
        fig = plt.figure(figsize=(12, 3 * num_cols))

        for i, col in enumerate(col_list):
            ax = plt.subplot(num_cols, 1, i + 1)
            auto_corr = []
            for j, lag in enumerate(lag_range):
                auto_corr.append(self.df[col].autocorr(lag=lag))
            self.auto_corr[col] = auto_corr

            ax.plot(lag_range, auto_corr)
            ax.set_title(col_list[i], fontsize=20)
            ax.set_xlim(lag_range[0], lag_range[-1])
        fig.supxlabel("lag", fontsize=20)
        fig.supylabel("auto_corr", fontsize=20)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path)
        else:
            fig.show()

    def save(self, file_name=None, fmt='parquet', comp='snappy', verbose=True):
        dir_path = os.path.join("data", "pre_" + self.precision)
        if file_name:
            file_path = os.path.join(dir_path, file_name)
        else:
            file_path = os.path.join(dir_path, utils.pre_processed_name(self.stocks, fmt, comp))
        if fmt == 'parquet':
            self.df.to_parquet(file_path, compression=comp)
        else:
            self.df.to_csv(file_path + ".csv")

        if verbose:
            print("Saved data:", file_path)

    def print(self):
        print(self.df)

    def head(self):
        return self.df.head()

    def dtypes(self):
        return self.df.dtypes
