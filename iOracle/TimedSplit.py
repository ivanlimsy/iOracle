from datetime import datetime as dt

class train_val_split:
    
    def __init__(self, df, duration=30, window=1, prediction_period=5, start = '2018-01-01', end = '2020-03-31'):
        self.duration = duration # training period, for dates
        self.window = window # rolling window freq, for dates
        self.prediction_period = prediction_period # prediciton horizon, for dates
        self.start = start
        self.end = end
        self.df = df.loc[(df.index >= start) & (df.index <= end)].reset_index()
        self.start_ind = self.df[self.df.Date>=self.start].index.min()
        
    @staticmethod
    def _strfdate(date):
        return dt.strftime(date, '%Y-%m-%d')
           
    
    def split_by_date(self):
        self.end_ind = self.start_ind + self.duration
        
        dates = []
        
        while self.end_ind <=  len(self.df) - self.prediction_period:
            date_start = self._strfdate(self.df.Date[self.start_ind])
            date_end = self._strfdate(self.df.Date[self.end_ind])
            
            dates.append([date_start, date_end])
            
            self.start_ind += self.window
            self.end_ind = self.start_ind + self.duration
            
        return dates
    
    def _chk_split(self, date_split):
        for n in range(len(date_split)-1):
            if date_split[n] >= date_split[n+1]:
                return False
        if date_split[0] < self.start:
            return False
        if date_split[-1] > self.end:
            return False
        return True
            
    
    def split_by_index(self, date_split = ['2018-09-30','2019-06-30','2020-03-31']):
        
        if not self._chk_split(date_split):
            return "Check date split again"
        
        ind_out = []            
        for n, date in enumerate(date_split):
            self.train_end_ind = self.df[self.df.Date<=date].index.max()
            
            if n < len(date_split)-1:
                val_end = self.df[self.df.Date<=date_split[n+1]].index.max()
            else:
                val_end = self.df.index.max()
            
            #check boundary_dates
            print(self.df.Date[self.train_end_ind], self.df.Date[self.train_end_ind+1], self.df.Date[val_end])
                
            ind_out.append((list(range(self.start_ind, self.train_end_ind+1)), list(range(self.train_end_ind+1, val_end+1))))
            
        return ind_out
    
    def get_val_map(self, start='2020-06-01', end='2020-12-31'):
        start_ind = self.df[self.df.Date>=start].index.min()
        end_ind = self.df[self.df.Date<=end].index.max()
        
        return {self._strfdate(self.df.Date[n-5]):self._strfdate(self.df.Date[n]) for n in range(start_ind, end_ind+1)}
        