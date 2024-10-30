import pandas as pd
import numpy as np

class RigidityProcessor():
    def __init__(self, filepath:str):
        self.filepath = filepath

    def read_file(self):
        data = pd.read_csv(self.filepath, skiprows=2, sep='\t')
        return data
    
    def processing_file(self):
        data = self.read_file()
        chanb_index = data.loc[data['Time'] == 'Channel B'].index[0]
        chana_df=data.iloc[1:chanb_index]
        chanb_df=data.iloc[chanb_index+3:len(data)]
        all_df = chana_df.copy()
        all_df.columns = ['Time', 'Channel A']
        all_df['Channel B'] = chanb_df[' Value'].values
        all_df = all_df[['Time', 'Channel B', 'Channel A']].reset_index(drop=True)

        convert_df = all_df[['Time', 'Channel B']]
        convert_df.reset_index(inplace=True, drop=True)
        convert_df['Channel B'] = pd.to_numeric(convert_df['Channel B'])
        convert_df['Torque (mNm)'] = all_df['Channel A'].apply(lambda x: (4256.6 * float(x) - 211.64)/1000)
        convert_df['Deviation (Degrees)'] = all_df['Channel B'].apply(lambda x: float(x)*(40/1400)-(40/14))

        convert_df['Cond1'] = False
        convert_df['Cond2'] = False

        # Identify blocks which satisfy the condition specified above

        for i in [x+1 for x in range(len(convert_df)-3)]:
            vals = convert_df.loc[i-1:i+2,'Channel B']
            if sum(vals.diff().dropna().values<-5) == 3:
                convert_df.at[i, 'Cond1'] = True
            if sum(vals.diff().dropna().values>5) == 3:
                convert_df.at[i, 'Cond2'] = True

        # Collect these blocks together in a nested list

        in_block = False
        block1_dct = {}
        for i in range(len(convert_df)):
            val = convert_df.at[i, 'Cond1']
            if not in_block:
                if val:
                    in_block = True
                    index = i
            if in_block:
                if not val:
                    block1_dct[index] = i
                    in_block = False

        in_block = False
        block2_dct = {}
        for i in range(len(convert_df)):
            val = convert_df.at[i, 'Cond2']
            if not in_block:
                if val:
                    in_block = True
                    index = i
            if in_block:
                if not val:
                    block2_dct[index] = i
                    in_block = False

        cond1_blocks = [list(convert_df['Torque (mNm)'][k:v]) for k,v in block1_dct.items()]
        cond2_blocks = [list(convert_df['Torque (mNm)'][k:v]) for k,v in block2_dct.items()]

        b1_counts = [len(x) for x in cond1_blocks]
        b1_avgs = [np.average(x) for x in cond1_blocks]
        b1_fin_vals = [x[len(x)-1] for x in cond1_blocks]
        b1_to_maintain = list(abs(b1_counts-np.average(b1_counts))<6)

        filt_b1_avgs = [x for x, flag in zip(b1_avgs, b1_to_maintain) if flag]
        filt_b1_fin_vals = [x for x, flag in zip(b1_fin_vals, b1_to_maintain) if flag]

        b2_counts = [len(x) for x in cond2_blocks]
        b2_avgs = [np.average(x) for x in cond2_blocks]
        b2_fin_vals = [x[len(x)-1] for x in cond2_blocks]
        b2_to_maintain = list(abs(b2_counts-np.average(b2_counts))<6)

        filt_b2_avgs = [x for x, flag in zip(b2_avgs, b2_to_maintain) if flag]
        filt_b2_fin_vals = [x for x, flag in zip(b2_fin_vals, b2_to_maintain) if flag]

        green_res = {
            'G Mean': np.average(filt_b1_avgs),
            'SD Gmean': np.std(filt_b1_avgs, ddof=1),
            'End Mean': np.average(filt_b1_fin_vals),
            'SD Emean': np.std(filt_b1_fin_vals, ddof=1)
        }

        red_res = {
            'G Mean': np.average(filt_b2_avgs),
            'SD Gmean': np.std(filt_b2_avgs, ddof=1),
            'End Mean': np.average(filt_b2_fin_vals),
            'SD Emean': np.std(filt_b2_fin_vals, ddof=1)
        }

        other_outputs = [
            np.max(convert_df.loc[convert_df['Cond1']]['Torque (mNm)']),
            np.average(convert_df.loc[convert_df['Cond1']]['Torque (mNm)']),
            np.max(convert_df.loc[convert_df['Cond2']]['Torque (mNm)']),
            np.average(convert_df.loc[convert_df['Cond2']]['Torque (mNm)'])
        ]

        return green_res, red_res, convert_df
    

    def get_rigidity_results(self):
        green_res, red_res, underlying_data = self.processing_file()
        self.green_results = green_res
        self.red_results = red_res
        self.underlying_data = underlying_data
