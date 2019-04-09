#!/usr/local/bin/python3.7

import sys
import getopt
import re
import os

import pandas as pd
import numpy as np
import scipy
from scipy import stats


def main(argv):
    """
    Fonction qui calcule la prime à payer dans le cadre d'une assurance parametrique
    :return:
    """
    # default values
    station = 'courchevel'
    mounth = '02'
    valeur = 10000
    taux = 0.005
    n_jour = [10, 15, 20, 25, 28]
    try:
        opts, args = getopt.getopt(argv, "hs:m:v:t:n:", ["station=", "month=", "valeur=", "taux=" "n_jour="])
    except getopt.GetoptError:
        print('main_script.py -s <station> -o <output_file>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Compute the insurance premium for a function \n'
                  + 'options : \n\n'
                  + '\t -s, --station\t station for which we compute the premium\n'
                  + '\t -m, --month\t month for which we compute the premium\n'
                  + '\t -v, --valeur\t payout\n'
                  + '\t -t, --taux\t risk free rate\n'
                  + '\t -n, --n_jour\t threshold that trigger the payout\n\n'
                  + 'example : \n\n'
                  + '\t./main_script.py -s tignes -m 01\n\n')
            sys.exit()
        elif opt in ("-s", "--station"):
            station = arg
        elif opt in ("-m", "--mounth"):
            mounth = arg
        elif opt in ("-v", "--valeur"):
            valeur = float(arg)
        elif opt in ("-t", "--taux"):
            taux = float(arg)
        elif opt in ("-n", "--n_jour"):
            n_jour = list(float(arg))

    directory = 'data_snow_fr_2010-2019/'
    # ID station
    id_station = pd.read_csv(directory + 'postesNivo.csv')  # DF containing id, station, altitude etc.
    id_station['Nom'] = id_station['Nom'].apply(lambda s: s.lower())
    ID = id_station[id_station['Nom'] == station].ID.values.astype(int)[0]

    regexp = re.compile(r'^nivo\.201.' + mounth + '\.csv$')
    files = [f for f in os.listdir(directory) if regexp.match(f)]

    df = load_station_month(ID, files)
    df.columns = ['id', 'date', 'altitude', 'ht_neige']
    df['station'] = [station for i in df.id]
    df["ht_neige"] = pd.to_numeric(df["ht_neige"])

    # Grouper les mesures par jour (moyenne ht neige mesuree par jour)
    df['date'] = df.date.dt.to_period('D')
    df = df.groupby(['date'], as_index=False).agg({'id': 'first', 'station': 'first', 'altitude': 'first',
                                                   'ht_neige': 'mean'
                                                   })

    epaisseur = 0.3
    df['is_neige'] = df['ht_neige'].apply(lambda x: 1 if x > epaisseur else 0)

    # nbr de jours (is neige) pour le mois choisi par annee
    df['date'] = [i[0:7] for i in df['date'].astype(str)]
    df = df.groupby(['date'], as_index=False).agg({'id': 'first', 'station': 'first', 'altitude': 'first',
                                                   'ht_neige': 'mean', 'is_neige': 'sum'
                                                   })
    best_window, tmp = find_best_window(df)
    df['rolling_mean'] = df['is_neige'].rolling(best_window).mean()
    df = df.dropna().reset_index(drop=True)
    print(df)

    # Model Actuariel
    df_premium: pd.DataFrame = compute_premium(df,
                                               rate=taux,
                                               payout=valeur,
                                               threshold_vector=n_jour,
                                               best_window=best_window,
                                               )

    print('Pour assurer la station ' + station + ' au mois de ' + mounth + ': \n')
    print(df_premium)
    return True


def load_station_month_1year(ID, file):
    """
    function that load one file for one station, one month, one year
    :param ID: ID of the station of interest
    :param file: name of the csv file to read
    :return: df containing the monthly data for the station of interest, replaicing  mq by np.NaN, by 0 for'ht_neige'
    """
    df = pd.read_csv(file, sep=';', parse_dates=['date'])
    df = df[df.numer_sta == ID]
    df = df[['numer_sta', 'date', 'haut_sta', 'ht_neige']]

    df = df.replace('mq', np.NAN)
    df['ht_neige'] = df['ht_neige'].fillna(0.0)
    return df


def load_station_month(ID, files):
    """
    function that loads all the files per year for one month and one station
    :param ID: ID of the station of interest
    :param files: list of files of the month of interest from 2010 to 2019
    :return: pandas DataFrame containing the data for 1 station
    """
    directory = 'data_snow_fr_2010-2019/'

    df = load_station_month_1year(ID, directory + files[0])
    for f in files[1:]:
        df_tmp = load_station_month_1year(ID, directory + f)
        df = pd.concat([df, df_tmp])
    return df


def find_best_window(df, window_range=[1, 2, 3]):
    """
    function that optimize the window size for the rolling mean. It selects the window that maximize the shapiro test.
    :param df: pandas DataFrame that contains the data on which the different
    :param window_range: range of windows to explore
    :return: the value of the window for the rolling mean
    """
    test_results = []
    for i in window_range:
        # for-loop that compute the shapiro test for the different values of the window of teh rolling mean
        test = df.copy(deep=True)
        test['rolling_mean'] = test['is_neige'].rolling(i).mean()
        test.dropna(inplace=True)
        test = test.reset_index(drop=True)
        test_results.append(stats.shapiro(np.log(test['rolling_mean']))[1])

    window = window_range[test_results.index(max(test_results))]
    return window, max(test_results)


def compute_premium(df: pd.DataFrame, payout: float = 10000, rate: float = 0.005,
                    threshold_vector: list = [10, 15, 20, 25, 28], best_window: int = 2,
                    epsilon: float = 1e-4) -> pd.DataFrame:
    """
    function that compute the insurance premium.
    :param df: pandas DataFrame containing the meteo data
    :param payout: expected payout
    :param rate: risk-free rate
    :param threshold_vector: threshold on which the insurance is paid
    :param best_window: best window for the rolling average
    :param epsilon: small float to avoid zeros in the log
    :return: pandas DataFrame containing the premium, the probability of the event
    """
    r_t = [round((e + sum(df['is_neige'].values[-best_window + 1:])) / best_window, 2) for e in threshold_vector]
    r0 = df['rolling_mean'].values[-1]
    # calculation of the volatility
    data = [np.log(i + epsilon) for i in df['rolling_mean']]
    sigma = np.std(data)
    mu = np.mean(data)

    # calculation of the risk neutral probability
    d2 = [(np.log(r0 / r_t[i]) + (rate - sigma ** 2 / 2)) / sigma for i in range(len(r_t))]
    n_d2 = [round(scipy.stats.norm(0, 1).cdf(-e), 2) for e in d2]
    premium = [round(payout * np.exp(-rate) * e, 2) for e in n_d2]

    # calculation of the natural probability
    natural_proba = [round(scipy.stats.norm(mu, sigma).cdf(np.log(e)), 2) for e in r_t]

    # intialise data of lists.
    d = {'Payout': [payout] * 5,
         'Seuil': threshold_vector,
         # 'RT': r_t,
         'Proba': natural_proba,
         'Prime': premium,
         }
    # Create DataFrame
    return pd.DataFrame(data=d)

if __name__ == "__main__":
    main(sys.argv[1:])
