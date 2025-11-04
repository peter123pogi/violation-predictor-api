import pandas as pd

class DF:
    def __init__(self, path):
        self.path = "./dataset/{}.csv".format(path)
        pd.set_option('display.max_rows', None)
        
    def set_df_path(self, s):
        self.path = "./dataset/{}".format(s)
        
    def get_path(self):
        return self.path
    
    def append_df(self, data):
        new_data = pd.DataFrame(data)
        df = self.get_df()
        df = pd.concat([df, new_data], ignore_index=True)
        df.to_csv(self.get_path(), index=False)
    
    def get_df(self):
        df = pd.read_csv(self.get_path())
        return df
    
    