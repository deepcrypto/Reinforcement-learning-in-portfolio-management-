# -*- coding: utf-8 -*-
"""
Notes: it's for downloading data
"""

class DataDownloader:
    def __init__(self,config):
        start_date = config["data"]["start_date"]
        end_date = config["data"]["end_date"]
        market_types = config["data"]["mark_types"]
        ktype = config["data"]["ktype"]
        for market in market_types:
            if market=='stock':
                stock_list=ts.get_stock_basics().index
                self.stock_data=[]
                for stock in stock_list:
                    print("---Downloading:",stock)
                    self.stock_data.extend(ts.get_k_data(stock,start=start_date,end=end_date,ktype=ktype).values)
    def save_data(self):
        pd.DataFrame(self.stock_data, columns=['time', 'open', 'close', 'high', 'low', 'volume', 'code']).to_csv(
            'stock_data.csv', index=False)