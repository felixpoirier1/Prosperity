import pandas as pd
import os
import re
from typing import Tuple, Dict
import json

class DataEngine:
    def __init__(self, base_path):
        self.data = {}
        i=0
        for folder in sorted(os.listdir(base_path)):
            if folder == '.DS_Store':
                continue
            i+=1
            folder_path = os.path.join(base_path, folder)
            price_df = pd.DataFrame()
            trade_df = pd.DataFrame()
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
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
    
class LogInterpreter:
    _LANDMARKS = {
        "sandbox": ("Sandbox logs:", "Activities log:"),
        "activities": ("Activities log:", "Trade History:"),
        "trades": ("Trade History:", "")
    }
    def __init__(self, log_path):
        self.log_path = log_path

    def __instantiate_file(self):
        self.file = open(self.log_path, "r")
    
    def getActivities(self, as_df=False):
        """
        returns the activities table, which contains the following columns:
        - day
        - timestamp
        - product
        - bid_price_1
        - bid_volume_1
        - bid_price_2
        - bid_volume_2
        - bid_price_3
        - bid_volume_3
        - ask_price_1
        - ask_volume_1
        - ask_price_2
        - ask_volume_2
        - ask_price_3
        - ask_volume_3
        - mid_price
        - profit_and_loss
        """
        self.__instantiate_file()
        for line in self.file:
            if line.strip() == self._LANDMARKS["activities"][0]:
                break

        columns = self.file.readline().strip().split(";")
        activities: Dict[Tuple[int,int], Dict]= {}
        for line in self.file:
            if line.strip() == self._LANDMARKS["activities"][1]:
                break
            if not line.strip():
                continue
            day, timestamp, product, *prices = line.strip().split(";")
            activities[(int(day), int(timestamp))] = dict(zip(columns, [int(day), int(timestamp), product, *map(lambda x: None if x == "" else float(x), prices)]))
        if as_df:
            activities = pd.DataFrame.from_dict(activities, orient='index', columns=columns)
        return activities
        
    def getTrades(self, as_df=False):
        """
        returns the trades table, which contains the following columns:
        - timestamp
        - buyer
        - seller
        - symbol
        - currency
        - price
        - quantity
        """
        self.__instantiate_file()
        for line in self.file:
            if line.strip() == self._LANDMARKS["trades"][0]:
                break
        
        trades: Dict[Tuple[int,int], Dict]= {}
        buff = ""

        for line in self.file:
            if not line.strip():
                continue
            if line.strip() == self._LANDMARKS["trades"][1]:
                break
            buff += line
        trades = json.loads(buff)
        if as_df:
            return pd.DataFrame(trades)
        return trades

    def getSandboxLogs(self, as_df=False):
        """
        returns the sandbox logs table, which contains the following columns:
        - timestamp
        - message
        """
        self.__instantiate_file()
        for line in self.file:
            if line.strip() == self._LANDMARKS["sandbox"][0]:
                break
        
        sandbox_logs: Dict[Tuple[int,int], Dict]= {}
        buff = "["

        for line in self.file:
            if not line.strip():
                continue
            if line.strip() == self._LANDMARKS["sandbox"][1]:
                break
            buff += line
            if line.strip() == "}":
                buff = buff.rstrip("\n") + ",\n"
        buff = buff [:-2]+ "\n]"
        print(buff)
        sandbox_logs = json.loads(buff)
        if as_df:
            return pd.DataFrame(sandbox_logs)
        return sandbox_logs

if __name__ == "__main__":
    log_path = '../logs/sf_rework__8_4__17_41.log'
    log_interpreter = LogInterpreter(log_path)
    print(log_interpreter.getTrades(True))
    print(log_interpreter.getActivities(True))
    print(log_interpreter.getSandboxLogs(True))