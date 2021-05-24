import os
import numpy as np
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
        self.kor_data[['stringency_index']] = self.kor_data[['stringency_index']].fillna(method='ffill', axis=0)

        #* Get daily data
        self.kor_daily_data = self._get_daily_data()

    def _load_data(self, type):
        return pd.read_parquet(path=os.path.join('data', type + '.parquet.snappy'))

    def _get_daily_data(self):
        kor_daily_data = pd.DataFrame()
        for column in ['confirmed', 'death', 'released', 'negative', 'tested']:
            kor_daily_data[column] = self.kor_data[column].diff().fillna(0)
        kor_daily_data[['critical', 'stringency_index']] = self.kor_data[['critical', 'stringency_index']]
        return kor_daily_data

    def save(self):
        self.kor_daily_data.to_parquet(path=os.path.join('data', str(self.last_date) + '.parquet.snappy'))

    def plot(self, col_list: list):
        num_row = int(np.ceil(len(col_list) / 2))
        fig = plt.figure(figsize=(20, 10 * num_row))
        date = self.kor_data.index

        for i, col in enumerate(col_list):
            ax = plt.subplot(num_row, 2, i + 1)
            ax.plot(date, self.kor_data[col], 'b-', linewidth=2, label="cumulative " + col)
            ax.set_xlim(date[0], date[-1])
            ax.set_ylim(bottom=0)
            ax.set_xlabel("Date", fontsize=25)
            ax.tick_params(axis='both', labelsize=20, direction='in')

            ax1 = ax.twinx()
            ax1.plot(date, self.kor_daily_data[col], 'r-', linewidth=2, label="daily" + col)
            ax1.set_xlim(date[0], date[-1])
            ax1.set_ylim(bottom=0)
            ax1.set_xlabel("Date", fontsize=25)
            ax1.tick_params(axis='both', labelsize=20, direction='in')

            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax1.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc='upper left', frameon=False, fontsize=25)

        fig.supxlabel("Date", fontsize=25)
        fig.supylabel("Num")
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

    url = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/timeseries/stringency_index.csv'
    kor_OxCGRT = pd.read_csv(url)
    kor_OxCGRT = kor_OxCGRT.loc[kor_OxCGRT['country_code'] == 'KOR'].T.iloc[3:].astype(dtype=np.float64)
    kor_OxCGRT.index, kor_OxCGRT.columns = pd.to_datetime(kor_OxCGRT.index), ['stringency_index']
    save(kor_OxCGRT, save_path=os.path.join('../data', 'kor_OxCGRT.parquet.snappy'))
    print("Saved kor_OxCGRT.parquet.snappy")
