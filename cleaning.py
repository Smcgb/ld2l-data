# used for cleaning a specfic season of the ld2l league pulled from the ld2lprototype.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import squarify

#global variables

# medals in order of rank

medals_order = ['H1', 'H2', 'H3', 'H4', 'H5',
                'G1', 'G2', 'G3', 'G4', 'G5',
                'C1', 'C2', 'C3', 'C4', 'C5',
                'A1', 'A2', 'A3', 'A4', 'A5',
                'L1', 'L2', 'L3', 'L4', 'L5',
                'N1', 'N2', 'N3', 'N4']

medals = {
    11: 'H1',12: 'H2',13: 'H3',14: 'H4',15: 'H5',
    21: 'G1',22: 'G2',23: 'G3',24: 'G4',25: 'G5',
    31: 'C1',32: 'C2',33: 'C3',34: 'C4',35: 'C5',
    41: 'A1',42: 'A2',43: 'A3',44: 'A4',45: 'A5',
    51: 'L1',52: 'L2',53: 'L3',54: 'L4',55: 'L5',
    61: 'N1',62: 'N2',63: 'N3',64: 'N4'
}

#this dict needs to be manually updated between seasons
#TODO Automate dict creation
teamID_dict = {
    8975614: 'Los Muertas',
    8975076: 'Sand Gangers',
    8979763: 'Outlaw Destroyers',
    8976019: 'Princess Gang',
    8975581: "Riki Blinders",
    8328421: "Arkosh Gaming",
    8975282: "MonkeyKings of NY",
    8980656: "TRASH BROS",
    8980666: "delightful dolphin (King Pirate)",
    8975838: "Alch Capone",
    8980472: "MS-322",
    8975289: "Leftside Lunchboxes"
}

def clean_date(df):
    """changes a single incorrect date to the correct date
    Modifies the dataframe in place and returns nothing"""

    df['date'] = df['date'].replace('2023-01-23', '2023-01-22')

def clean_duration(df):
    """re-engineer match duration by dividing gpm by 60 for the
    first entry in the match and applying that time to every entry
    with the same match_id"""

    df['duration'] = df['duration'].replace(0, np.nan)
    df['duration'] = df['duration'].fillna(df.groupby('match_id')['gold_per_min'].transform('first') / 60)


def clean_teams(df):
    """Maps the teamID to the team name
    Modifies the dataframe in place and returns nothing"""

    df['teamID'] = df['teamID'].map(teamID_dict)

def clean_medals(df):
    """Maps the rank_tier to the medal and renumbers the rank tier to linear order instead of the weird order it is in
    Modifies the dataframe in place and returns nothing"""

    # create a new column that maps the rank_tier to the medal
    df['medal'] = df['rank_tier'].map(medals)

    # change rank tier to be the order its key appears in the medals dict + 1
    df['rank_tier'] = df['rank_tier'].map(lambda x: list(medals.keys()).index(x) + 1)


def ranks(df):
    """Creates a dataframe that contains the rank tier and medal for each player
    Plots the data and returns nothing"""


    df_ranks = df.groupby('account_id')[['rank_tier', 'medal']].first().reset_index()

    plt.figure(figsize=(20,10))
    sns.countplot(x='medal', data=df_ranks, order=medals_order)
    plt.title('Medals of Players Distribution')
    plt.axvline(x=np.median(df_ranks['rank_tier']), color='pink', linestyle='solid', label='median')
    plt.axvline(x=round(np.mean(df_ranks['rank_tier'])), color='skyblue', linestyle='solid', label='mean')
    #standard deviation
    plt.axvline(x=round(np.mean(df_ranks['rank_tier']) + np.std(df_ranks['rank_tier'])), color='black', linestyle='dashed', label='std')
    plt.axvline(x=round(np.mean(df_ranks['rank_tier']) - np.std(df_ranks['rank_tier'])), color='black', linestyle='dashed')
    plt.legend()
    plt.show()

def three_plus(df):
    """Creates a filtered dataframe that only includes players who have played in 3 or more games
    Plots the data and returns nothing"""
    df_ranks = df.groupby('account_id')[['rank_tier', 'medal']].first().reset_index()
    df_3plus = df[df['account_id'].isin(df['account_id'].value_counts()[df['account_id'].value_counts() >= 3].index)]
    df_3plus_winrate = df_3plus.groupby('account_id')[['win']].mean().reset_index()
    df_3plus_winrate['win'] = df_3plus_winrate['win'].map(lambda x: round(x, 2) * 100)
    df_3plus_winrate = df_3plus_winrate.merge(df_ranks, on='account_id', how='left')

    plt.figure(figsize=(20,10))
    sns.barplot(x='medal', y='win', data=df_3plus_winrate, order=medals_order)
    plt.title('Winrate of Players with 3 or More Matches')
    plt.show()

def medals_winrate(df):
    """ This function lazily repeats the same code as the three_plus function and prints winrates for each medal
    returns nothing"""
    df_ranks = df.groupby('account_id')[['rank_tier', 'medal']].first().reset_index()
    df_3plus = df[df['account_id'].isin(df['account_id'].value_counts()[df['account_id'].value_counts() >= 3].index)]
    df_3plus_winrate = df_3plus.groupby('account_id')[['win']].mean().reset_index()
    df_3plus_winrate['win'] = df_3plus_winrate['win'].map(lambda x: round(x, 2) * 100)
    df_3plus_winrate = df_3plus_winrate.merge(df_ranks, on='account_id', how='left')
    low_rank = df_3plus_winrate[df_3plus_winrate['rank_tier'] <= 10]
    middle_rank = df_3plus_winrate[(df_3plus_winrate['rank_tier'] > 10) & (df_3plus_winrate['rank_tier'] <= 23)]
    high_rank = df_3plus_winrate[df_3plus_winrate['rank_tier'] > 23]

    print('Low Rank Winrate: ', round(np.mean(low_rank['win']), 2), ' +/- ', round(np.std(low_rank['win']), 2))
    print('Middle Rank Winrate: ', round(np.mean(middle_rank['win']), 2) , ' +/- ', round(np.std(middle_rank['win']), 2))
    print('High Rank Winrate: ', round(np.mean(high_rank['win']), 2) , ' +/- ', round(np.std(high_rank['win']), 2))
#%%

def xpmgpmkda(df):
    """plots xpm, gpm, and kda for each medal"""

    plt.figure(figsize=(20,10))
    sns.barplot(x='medal', y='gold_per_min', data=df, order=medals_order)
    plt.axhline(y=np.mean(df['gold_per_min']), color='gray', linestyle='solid', label='mean')
    plt.axhline(y=np.median(df['gold_per_min']), color='orange', linestyle='solid', label='median')
    plt.title('Average Gold Per Minute by Medal')
    plt.legend()
    plt.show()

    plt.figure(figsize=(20,10))
    sns.barplot(x='medal', y='xp_per_min', data=df, order=medals_order)
    plt.title('Average Experience Per Minute by Medal')
    plt.axhline(y=np.mean(df['xp_per_min']), color='gray', linestyle='solid', label='mean')
    plt.axhline(y=np.median(df['xp_per_min']), color='orange', linestyle='solid', label='median')
    plt.legend()
    plt.show()

    plt.figure(figsize=(20,10))
    sns.barplot(x='medal', y='kills', data=df, order=medals_order)
    plt.title('Average Kills by Medal')
    plt.axhline(y=np.mean(df['kills']), color='gray', linestyle='solid', label='mean')
    plt.axhline(y=np.median(df['kills']), color='orange', linestyle='solid', label='median')
    plt.show()

def die_pingers(df):
    """plots the total pings of the top 10 pingers.
    If your name appears here, be ashamed of yourself"""

    # # convert duration from seconds to minutes

    df['duration'] = df['duration']/60

    # create new feature, pings_per_minute

    df['pings_per_minute'] = df['pings']/df['duration']

    # find top 10 pingers with pings_per_minute

    df_pings = df.groupby('personaname')[['pings_per_minute']].mean().reset_index()

    plt.figure(figsize=(20,10))
    sns.barplot(x='personaname', y='pings_per_minute', data=df_pings.sort_values('pings_per_minute', ascending=False).head(10))
    plt.show()

    df_pings = df.groupby('personaname')[['pings']].sum().reset_index()

    #visualize top 10 most toxic players

    plt.figure(figsize=(20,10))
    sns.barplot(x='personaname', y='pings', data=df_pings.sort_values('pings', ascending=False).head(10))

    plt.title('Top 10 Most Toxic Players')
    plt.show()

    plt.figure(figsize=(20,10))
    sns.barplot(x='personaname', y='pings', data=df_pings.sort_values('pings', ascending=False).tail(10))

    plt.title('Top 10 Least Toxic Players (by pings)')
    plt.show()

    df_team_pings = df.groupby('teamID')['pings'].sum().reset_index()

    plt.figure(figsize=(20,10))
    sns.barplot(x='teamID', y='pings', data=df_team_pings.sort_values('pings', ascending=False))
    plt.show()

def time_series_plotting(df):
    """Plots variable time series for the regular season"""

    teams = df['teamID'].unique()

    df_kda = df.groupby('date')[['kda']].mean().reset_index()

    # turn kda into a cumulative average
    df_kda['kda'] = df_kda['kda'].cumsum() / (df_kda.index + 1)


    plt.figure(figsize=(20,10))

    plt.plot(df_kda['date'], df_kda['kda'], label='LD2L', linewidth=4, color='black')
    #ensure axis goes from 0 to 5
    plt.ylim(0, 7)
    plt.title('Average KDA Over Time')

    # create a cumulative average for each team

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf', '#aec7e8', '#ffbb78']

    for team in teams:
        df_team = df[df['teamID'] == team]
        df_team_kda = df_team.groupby('date')[['kda']].mean().reset_index()
        df_team_kda['kda'] = df_team_kda['kda'].cumsum() / (df_team_kda.index + 1)

        plt.plot(df_team_kda['date'], df_team_kda['kda'], label=team, linewidth=2, linestyle='dashed', color=colors[0])
        colors.pop(0)

    # plot teams cumulative average kda over time
    plt.legend()
    plt.show()

    # do the same for gpm

    df_gpm = df.groupby('date')[['gold_per_min']].mean().reset_index()

    # turn gpm into a cumulative average
    df_gpm['gold_per_min'] = df_gpm['gold_per_min'].cumsum() / (df_gpm.index + 1)

    plt.figure(figsize=(20,10))

    plt.plot(df_gpm['date'], df_gpm['gold_per_min'], label='LD2L', linewidth=4, color='black')
    # ensure axis goes from 0 to an appropriate maximum value
    plt.title('Average GPM Over Time')

    # create a cumulative average for each team

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf', '#aec7e8', '#ffbb78']

    for team in teams:
        df_team = df[df['teamID'] == team]
        df_team_gpm = df_team.groupby('date')[['gold_per_min']].mean().reset_index()
        df_team_gpm['gold_per_min'] = df_team_gpm['gold_per_min'].cumsum() / (df_team_gpm.index + 1)

        plt.plot(df_team_gpm['date'], df_team_gpm['gold_per_min'], label=team, linewidth=2, linestyle='dashed', color=colors[0])
        colors.pop(0)

    # plot teams cumulative average gpm over time
    plt.legend()
    plt.show()

    # do the same for xpm

    df_xpm = df.groupby('date')[['xp_per_min']].mean().reset_index()

    # turn xp_per_minute into a cumulative average
    df_xpm['xp_per_min'] = df_xpm['xp_per_min'].cumsum() / (df_xpm.index + 1)

    plt.figure(figsize=(20,10))

    plt.plot(df_xpm['date'], df_xpm['xp_per_min'], label='LD2L', linewidth=4, color='black')
    plt.title('Average XP Per Minute Over Time')

    # create a cumulative average for each team

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf', '#aec7e8', '#ffbb78']

    for team in teams:
        df_team = df[df['teamID'] == team]
        df_team_xpm = df_team.groupby('date')[['xp_per_min']].mean().reset_index()
        df_team_xpm['xp_per_min'] = df_team_xpm['xp_per_min'].cumsum() / (df_team_xpm.index + 1)

        plt.plot(df_team_xpm['date'], df_team_xpm['xp_per_min'], label=team, linewidth=2, linestyle='dashed', color=colors[0])
        colors.pop(0)

    # plot teams cumulative average xp_per_minute over time
    plt.legend()
    plt.show()

def roshan_haters(df):
    """Plots the number of roshans killed by each player"""

    df_roshans = df.groupby('personaname')['roshans_killed'].sum().reset_index()

    plt.figure(figsize=(20,10))
    sns.barplot(x='personaname', y='roshans_killed', data=df_roshans.sort_values('roshans_killed', ascending=False).head(10))

    plt.title('Top 10 Roshan Killers')
    plt.show()

def most_common_heroes(df):
    """Plots the most common heroes played as a tree map"""

    df_heroes = df.groupby('hero_id')['hero_id'].count().reset_index(name='count').sort_values(['count'], ascending=False)

    #filter to top 10 heroes
    df_heroes = df_heroes.head(10)

    sns.set_style('whitegrid')
    plt.figure(figsize=(20,10))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # plot the tree map with counts and hero ids
    labels = []
    for hero in df_heroes['hero_id']:
        labels.append(f"{hero} {df_heroes[df_heroes['hero_id'] == hero]['count'].values[0]}")
    squarify.plot(sizes=df_heroes['count'], label=labels, alpha=.8, color=colors)
    plt.axis('off')
    plt.show()


def hero_puddle(df):
    """shows which players have the least hero diversity with a minimum of 3 games played"""

    # get players with at least 3 games played

    df_players = df.groupby('personaname')['personaname'].count().reset_index(name='count').sort_values(['count'], ascending=False)
    df_players = df_players[df_players['count'] >= 3]
    # get the number of unique heroes played by each player

    df_heroes = df.groupby('personaname')['hero_id'].nunique().reset_index(name='count').sort_values(['count'], ascending=False)
    df_heroes = df_heroes[df_heroes['personaname'].isin(df_players['personaname'])]

    plt.figure(figsize=(20,10))
    sns.barplot(x='personaname', y='count', data=df_heroes.sort_values('count', ascending=False).head(10))
    plt.title('Top 10 Players with the Least Hero Diversity')
    plt.show()

    # show most diverse players
    plt.figure(figsize=(20,10))
    sns.barplot(x='personaname', y='count', data=df_heroes.sort_values('count', ascending=True).head(10))
    plt.title('Top 10 Players with the Most Hero Diversity')
    plt.show()




def numbah_one_stunna(df):
    """plots the top 5 player with the highest sum of stun durations
    and separately plots the top 5 heroes with the highest stun durations"""

    df_player_stuns = df.groupby('personaname')['stuns'].sum().reset_index().sort_values('stuns', ascending=False).head(5)

    plt.figure(figsize=(20,10))
    sns.barplot(x='personaname', y='stuns', data=df_player_stuns)
    plt.title('Top 5 Players with the Highest Sum of Stun Durations')
    plt.show()

    df_hero_stuns = df.groupby('hero_id')['stuns'].sum().reset_index().sort_values('stuns', ascending=False).head(5)
    plt.figure(figsize=(20,10))
    sns.barplot(x='hero_id', y='stuns', data=df_hero_stuns)
    plt.title('Top 5 Heroes with the Highest Sum of Stun Durations')
    plt.show()

    highest_stun_game = df[df['stuns'] == df['stuns'].max()]
    print(highest_stun_game[['personaname', 'hero_id', 'stuns', 'match_id']])

def decision_tree_ranks(df):

    """Creates a decision tree to rank teams based on their performance"""

    # create a dataframe with only the columns we want to use

    df_ranks = df[['teamID', 'kda', 'gold_per_min', 'xp_per_min',
                   'last_hits', 'denies', 'roshans_killed', 'tower_damage',
                   'hero_damage', 'hero_healing', 'kda', 'win']]

    # win is the target variable

    X = df_ranks.drop('win', axis=1)

    y = df_ranks['win']

    # split the data into training and testing data

