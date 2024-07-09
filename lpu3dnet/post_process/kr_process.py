import numpy as np
import pandas as pd
from scipy.optimize import curve_fit



class Corey_fit:
    def __init__(self,kr_df):
        # input is kr_df
        self.kr_df = kr_df
        self.sw = self.kr_df['sw'].values
        self.krw = self.kr_df['kr_water'].values
        self.krnw = self.kr_df['kr_air'].values

        self.par_krw = None
        self.cov_krw = None

        self.par_krnw = None
        self.cov_krnw = None

        # irreducible water saturation
        self.swirr = None
        # residual gas saturation
        self.sor = None


    def krw_model(self,sw, krw0, swirr, sor, nw):
        denominator = 1 - sor - swirr
        if denominator == 0:
            return np.zeros_like(sw)
        return krw0 * np.maximum(0, np.minimum(1, (sw - swirr) / denominator)) ** nw

    def kro_model(self,sw, kro0, sor, swirr, no):
        
        so = 1 - sw
        denominator = 1 - sor - swirr
        if denominator == 0:
            return np.zeros_like(so)
        return kro0 * np.maximum(0, np.minimum(1, (so - sor) / denominator)) ** no
    
    def curve_fit(self):

        init_guess_krw = [1, 0.1, 0.2, 2]  # [krw0, swirr, sor, nw]
        init_guess_kro = [1, 0.2, 0.1, 2]  # [kro0, sor, swirr, no]

        bounds_krw = ([0, 0, 0, 1], [1, 1, 1, 6])
        bounds_kro = ([0, 0, 0, 1], [1, 1, 1, 6])

        self.par_krw, self.cov_krw = curve_fit(
            self.krw_model,
            self.sw,
            self.krw,
            p0=init_guess_krw,
            bounds=bounds_krw
            )
        
        self.par_krnw, self.cov_krnw = curve_fit(
            self.kro_model,
            self.sw,
            self.krnw,
            p0=init_guess_kro,
            bounds=bounds_kro
            )

        self.swirr = self.par_krw[1]
        self.sor = self.par_krnw[1]

    def generate_kr_data(self):
        # curve fitting
        self.curve_fit()
        
        Sw_fit = np.linspace(0, 1, 100)
        krw_fit = self.krw_model(Sw_fit, *self.par_krw)
        krnw_fit = self.kro_model(Sw_fit, *self.par_krnw)
        # construct the dataframe
        krw_df = pd.DataFrame({
            'sw':Sw_fit,
            'krw':krw_fit,
            'krnw':krnw_fit
            }
            )
        
        # make sure all krw or krnw are between 0 and 1
        krw_df.loc[krw_df['krw'] > 1, 'krw'] = 1
        krw_df.loc[krw_df['krw'] < 0, 'krw'] = 0
        krw_df.loc[krw_df['krnw'] > 1, 'krnw'] = 1
        krw_df.loc[krw_df['krnw'] < 0, 'krnw'] = 0

        if self.swirr + self.sor >= 0.8:
            # if the sum of swirr and sor is greater than 0.9, then the kr data is not good and too extreme
            return krw_df, [self.par_krw,self.par_krnw],False
        else:
            return krw_df,[self.par_krw,self.par_krnw],True



class Exponential_fit:
    def __init__(self,kr_df):
        # input is kr_df
        self.kr_df = kr_df
        self.sw = self.kr_df['sw'].values
        self.krw = self.kr_df['kr_water'].values
        self.krnw = self.kr_df['kr_air'].values

        self.par_krw = None
        self.cov_krw = None

        self.par_krnw = None
        self.cov_krnw = None

    def krw_water(self,Sw, a, b, Swi):
        return a * np.exp(b * (Sw - Swi))


    def krnw_model(self,Sw, krnw0, c):
        # Ensure krnw goes to 0 as Sw approaches 1
        return krnw0 * np.exp(-c * Sw / (1 - Sw))
    
    def curve_fit(self):
        self.par_krw, self.cov_krw = curve_fit(
            self.krw_water, self.sw, self.krw, p0=[1, 1, 0.2]
            ) 
        self.par_krnw, self.cov_krnw = curve_fit(
            self.krnw_model,
            self.sw,
            self.krnw, p0=[1, 1]
            )

    def generate_kr_data(self):
        # curve fitting
        self.curve_fit()
        
        Sw_fit = np.linspace(0, 1, 100)
        krw_fit = self.krw_water(Sw_fit, *self.par_krw)
        krnw_fit = self.krnw_model(Sw_fit, *self.par_krnw)
        # construct the dataframe
        krw_df = pd.DataFrame({
            'sw':Sw_fit,
            'krw':krw_fit,
            'krnw':krnw_fit
            }
            )
        
        # make sure all krw or krnw are between 0 and 1
        krw_df.loc[krw_df['krw'] > 1, 'krw'] = 1
        krw_df.loc[krw_df['krw'] < 0, 'krw'] = 0
        krw_df.loc[krw_df['krnw'] > 1, 'krnw'] = 1
        krw_df.loc[krw_df['krnw'] < 0, 'krnw'] = 0

        return krw_df


def convert_dict_to_pd(dict_kr):
    df_kr = pd.DataFrame(
        {
        'snw':dict_kr['snw'],
        'sw': dict_kr['sw'],
        'kr_air':dict_kr['kr_air'],
        'kr_water':dict_kr['kr_water']
                            }
                            )
    return df_kr


# some utility function
def aggregate_kr(kr_dict_list):
    df_list = []
    for kr_dict in kr_dict_list:

        df_kr = convert_dict_to_pd(kr_dict)
        df_list.append(df_kr)

    df_all = pd.concat(df_list, ignore_index=True)
    df_all = df_all.dropna()

    return df_all
