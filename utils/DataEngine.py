import pandas as pd
import os
import re
import logging

logging.basicConfig(level=logging.INFO)


class DataEngine:
    def __init__(self, base_path):
        self.data = {}
        i=0
        for folder in sorted(os.listdir(base_path)):
            logging.debug(f'Searching {folder}')
            if folder == '.DS_Store':
                continue
            i+=1
            folder_path = os.path.join(base_path, folder)
            price_df = pd.DataFrame()
            trade_df = pd.DataFrame()
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                logging.debug(f'\tLoading {file_path}')
                df = pd.read_csv(file_path, delimiter=';')
                if re.search('price', file):
                    price_df = pd.concat([price_df, df])
                elif re.search('trade', file):
                    day = int(re.search(r'day_(-?[0-3])', file).group(1))
                    df['day'] = day
                    trade_df = pd.concat([trade_df, df])
                else:
                    raise ValueError('File not a price or trade df.')
            
            price_df = None if price_df.empty else self._clean_df(price_df)
            trade_df = None if trade_df.empty else self._clean_df(trade_df)

            self.data[f'round_{i}'] = {
                'price_df': price_df, 
                'trade_df': trade_df
                }
    
    def _clean_df(self, df):
        df.sort_values(by=['day', 'timestamp'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['timestamp'] = df['timestamp'] + 1_000_000 * (df['day'] + 2)

        return df

if __name__ == '__main__':
    base_path = 'data2023'
    de = DataEngine(base_path)
    print(de.data['round_3']['price_df'].head())

