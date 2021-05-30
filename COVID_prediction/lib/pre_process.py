import os
import numpy as np
from numpy.lib.function_base import extract
import pandas as pd
import matplotlib.pyplot as plt


def save(df: pd.core.frame.DataFrame, save_path: str):
    df.to_parquet(save_path, compression='snappy')

class pre_process():
    def __init__(self):
        #* Load every data
        self.kor_data = self._load_data('kor_daily')
        self.kor_inspection_data = self._load_data('kor_inspection')
        self.kor_OxCGRT = self._load_data('kor_OxCGRT')

        #* Check last date of each data
        self.last_date = self.kor_data.index[-1].date()
        assert self.last_date == self.kor_inspection_data.index[-1].date(), "Check last date"

        #* Get effective number (without inspection number) and reorder
        self.kor_data = self.kor_data.sub(self.kor_inspection_data, axis='columns', fill_value=0)

        #* Append OxCGRT index
        self.kor_data = self.kor_data.join(self.kor_OxCGRT)
        self.kor_data = self.kor_data.fillna(method='ffill', axis=0)

        #* Get daily data
        self.kor_daily_data = self._get_daily_data()

        #* Genrate new column: positive_ratio
        self.kor_daily_data['positive_ratio'] = self.kor_daily_data['confirmed'] / self.kor_daily_data['tested']
        self.kor_daily_data['positive_ratio'] = self.kor_daily_data['positive_ratio'].fillna(0)

    def _load_data(self, type):
        return pd.read_parquet(path=os.path.join('data', type + '.parquet.snappy'))

    def _get_daily_data(self):
        cumulative_column = ['confirmed', 'death', 'released', 'negative', 'tested']
        kor_daily_data = self.kor_data.copy()
        kor_daily_data[cumulative_column] = self.kor_data[cumulative_column].diff().fillna(0)
        return kor_daily_data

    def _check_missing_value(self, verbose=True):
        missing_loc, missing_num = {}, 0
        for col in self.kor_daily_data.columns:
            try:
                idx_list = np.where(np.isnan(self.kor_daily_data[col]))[0]
                for idx in idx_list:
                    missing_num += 1
                    if idx in missing_loc:
                        missing_loc[idx].append(col)
                    else:
                        missing_loc[idx] = [col]
            except TypeError:  # * For column 'Name'
                continue

        if verbose:
            print("Total number of missing data: {}".format(missing_num))
            if missing_num:
                for idx, col_name in missing_loc.items():
                    print("{}:".format(self.kor_daily_data.index[idx].date()), ', '.join(col_name))
        return missing_num

    def shift(self, col_name, period, fill_na=0, in_place=False):
        if in_place:
            self.kor_daily_data[col_name] = self.kor_daily_data[col_name].shift(period)
        else:
            temp = np.array(self.kor_daily_data[col_name].shift(period))
            temp = np.nan_to_num(temp, copy=False, nan=fill_na)
            return temp

    def get_correlation_shift(self, col_name1, col_name2, method='pearson'):
        shift_range = np.arange(-50, 50, 1)
        correlation_list = []
        for period in shift_range:
            corr = self.kor_daily_data[col_name1].corr(self.kor_daily_data[col_name2].shift(period), method=method)
            correlation_list.append(corr)
        return shift_range, np.array(correlation_list)

    def save(self):
        if self._check_missing_value():
            print("Not Saved!")
            return
        self.kor_daily_data.to_parquet(path=os.path.join('data', str(self.last_date) + '.parquet.snappy'))

    def plot(self, col_list: list):
        num_row = int(np.ceil(len(col_list) / 2))
        fig = plt.figure(figsize=(20, 10 * num_row))
        date = self.kor_data.index

        for i, col in enumerate(col_list):
            ax = plt.subplot(num_row, 2, i + 1)
            ax1 = ax.twinx()
            try:
                ax.plot(date, self.kor_data[col], 'b-', linewidth=2, label="cumulative " + col)
                ax.set_xlim(date[0], date[-1])
                ax.set_ylim(bottom=0)
                ax.set_xlabel("Date", fontsize=25)
                ax.tick_params(axis='both', labelsize=20, direction='in')
            except KeyError:
                ax1.set_ylim(top=0.2)

            ax1.plot(date, self.kor_daily_data[col], 'r-', linewidth=2, label="daily " + col)
            ax1.set_xlim(date[0], date[-1])
            ax1.set_ylim(bottom=0)
            ax1.set_xlabel("Date", fontsize=25)
            ax1.tick_params(axis='both', labelsize=20, direction='in')

            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax1.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc='upper left', frameon=False, fontsize=25)

        fig.supxlabel("Date", fontsize=25)
        fig.supylabel("Num", fontsize=25)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.show()



if __name__ == "__main__":
    url = 'https://raw.githubusercontent.com/jooeungen/coronaboard_kr/master/kr_daily.csv'
    kor_daily = pd.read_csv(url, index_col='date', parse_dates=['date'])
    kor_daily['critical'] = kor_daily['critical'].fillna(0).astype(dtype=np.int64)
    kor_daily = kor_daily.loc[:, ['confirmed', 'death', 'released', 'negative', 'tested', 'critical']]
    save(kor_daily, save_path=os.path.join('../data', 'kor_daily.parquet.snappy'))
    print("Saved kor_daily.parquet.snappy")

    url = 'https://raw.githubusercontent.com/jooeungen/coronaboard_kr/master/kr_regional_daily.csv'
    kor_inspection = pd.read_csv(url, index_col='date', parse_dates=['date'])
    kor_inspection = kor_inspection.loc[kor_inspection['region'] == '검역']
    kor_inspection.drop(labels='region', axis='columns', inplace=True)
    save(kor_inspection, save_path=os.path.join('../data', 'kor_inspection.parquet.snappy'))
    print("Saved kor_inspection.parquet.snappy")

    url = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest_combined.csv'
    kor_OxCGRT = pd.read_csv(url, index_col='Date', parse_dates=['Date'])
    kor_OxCGRT = kor_OxCGRT.loc[kor_OxCGRT['CountryCode'] == 'KOR'].drop(labels=['CountryName', 'CountryCode', 'RegionName', 'RegionCode', 'Jurisdiction'], axis='columns')
    kor_OxCGRT = kor_OxCGRT.loc[:, ['C1_combined_numeric',
                                    'C2_combined_numeric',
                                    'C3_combined_numeric',
                                    'C4_combined_numeric',
                                    'C6_combined_numeric',
                                    'C7_combined_numeric',
                                    'C8_combined_numeric',
                                    'H1_combined_numeric',
                                    'H2_combined_numeric',
                                    'H3_combined_numeric',
                                    'H6_combined_numeric',
                                    'StringencyIndex',
                                    'ContainmentHealthIndex']]
    kor_OxCGRT.columns = ['C1', 'C2', 'C3', 'C4', 'C6', 'C7', 'C8', 'H1', 'H2', 'H3', 'H6', 'Stringency', 'Containment']
    save(kor_OxCGRT, save_path=os.path.join('../data', 'kor_OxCGRT.parquet.snappy'))
    print("Saved kor_OxCGRT.parquet.snappy")
