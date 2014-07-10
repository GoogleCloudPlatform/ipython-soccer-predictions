"""
    Ranks soccer teams by computing a power index based
    on game outcomes.
"""

import numpy as np
from numpy.linalg import LinAlgError
import pandas as pd

import world_cup

def _build_team_matrix(data, target_col):
    """ Given a dataframe of games, builds a sparse power matrix.
        We expect the input data to have two back to back rows for
        each game. The first row will have information about the home
        team, the second row will have information about the away team.
        The matrix we compute will have columns representing teams and
        rows representing games. For each game, the home team will have
        a positive value that team's column. The away team will have a
        negative value in that column. Since home advantage is so
        important in soccer, we discount the home team by a certain
        margin. Note that we also have to be somewhat careful here,
        because for world cup data, we use values of is_home that are
        not binary (that is, they range between 0,0.0 and 1.0.
        The final column in the power matrix is a points value,
        computed as the difference between the target column for the
        home team and the target column for the away team.
    """
    teams = {}
    nrows = len(data) / 2
    for teamid in data['teamid']:
        teams[str(teamid)] = pd.Series(np.zeros(nrows))

    result = pd.Series(np.empty(nrows))
    teams[target_col] = result

    current_season = None
    current_discount = 1.0

    for game in xrange(nrows):
        home = data.iloc[game * 2]
        away = data.iloc[game * 2 + 1]
        if home['seasonid'] != current_season:
            # Discount older seasons.
            current_season = home['seasonid']
            current_discount *= 0.9
            print "New season %s" % (current_season,)

        home_id = str(home['teamid'])
        away_id = str(away['teamid'])
        points = home[target_col] - away[target_col]

        # Discount home team's performance.
        teams[home_id][game] = (1.0 + home['is_home'] * .25) / current_discount
        teams[away_id][game] = (-1.0 - away['is_home'] * .25) / current_discount
        result[game] = points

    return pd.DataFrame(teams)


def _build_power(games, outcomes, coerce_fn, acc=0.0001, alpha=1.0, snap=True):
    """ Builds power model over a set of related games (they 
        should all be from the same competition, for example).
        Given a series of games and their outcome, builds a logistic
        regression model that computes a relative ranking for the teams.
        Returns a dict of team id to power ranking between 0 and 1.
        If snap is set, the rankings are bucketed into quartiles. This
        is useful bcause we may only have rough estimates of power
        rating and we don't want to get a false specificity.
    """
    outcomes = pd.Series([coerce_fn(val) for val in outcomes])
    model = world_cup.build_model_logistic(outcomes, games, 
        acc=acc, alpha=alpha)

    # print model.summary()
    params = np.exp(model.params)
    del params['intercept']
    params = params[params != 1.0]
    max_param = params.max()
    min_param = params.min()
    param_range = max_param - min_param
    if len(params) == 0 or param_range < 0.0001:
        return None
    
    params = params.sub(min_param)
    params = params.div(param_range)
    qqs = np.percentile(params, [20, 40, 60, 80])
    def _snap(val): 
        """ Snaps a value to a quartile. """
        for idx in xrange(len(qqs)):
            if (qqs[idx] > val):
                return idx * 0.25
        return 1.0
      
    if snap:
        # Snap power data to rough quartiles.
        return params.apply(_snap).to_dict()
    else:
        return params.to_dict()


def _get_power_map(competition, competition_data, col, coerce_fn):
    """ Given the games in a competition and the target column
        describing the result, compute a power ranking of the teams.
        Since the 'fit' is likely to be fairly loose, we may
        have to try several times with different regularization and
        alpha parameters before we get it to converge.
        Returns a map of team id to power ranking.
    """
    acc = 0.000001
    alpha = 0.5
    while True:
        if alpha < 0.1:
            print "Skipping power ranking for competition %s column %s" % (
                competition, col)
            return {}
        try:
            games = _build_team_matrix(competition_data, col)
            outcomes = games[col]
            del games[col]
            competition_power = _build_power(games, outcomes, coerce_fn, acc,
                                             alpha)
            if not competition_power:
                alpha /= 2
                print 'Reducing alpha for %s to %f due lack of range' % (
                    competition, alpha)
            else:
                return competition_power
        except LinAlgError, err:
            alpha /= 2  
            print 'Reducing alpha for %s to %f due to error %s' % (
                competition, alpha, err)


def add_power(data, power_train_data, cols):
    """ Adds a number of power columns to a data frame.
        Splits the power_train_data into competitions (since those will
        have disjoint power statistics; for example, EPL teams don't play
        MLS teams (in regular games), so trying to figure out which team is
        stronger based on wins and losses isn't going to be useful.

        Each entry in cols should be a column name that will be used to 
        predict, a function that wil evaluate the difference in that
        column between the two teams that played a game, and a final
        name that will be used to name the resulting power column. 
        
        Returns a data frame that is equivalent to 'data' ammended with
        the power statistics for the primary team in the row.
    """
   
    data = data.copy()
    competitions = data['competitionid'].unique()
    for (col, coerce_fn, final_name) in cols:
        power = {}
        for competition in competitions:
            competition_data = power_train_data[
                power_train_data['competitionid'] == competition]
            power.update(
                _get_power_map(competition, competition_data, col, coerce_fn))

        names = {}
        power_col = pd.Series(np.zeros(len(data)), data.index)
        for index in xrange(len(data)):
            teamid = str(data.iloc[index]['teamid'])
            names[data.iloc[index]['team_name']] = power.get(teamid, 0.5)
            power_col.iloc[index] = power.get(teamid, 0.5)
        print ['%s: %0.03f' % (x[0], x[1])
               for x in sorted(names.items(), key=(lambda x: x[1]))]
        data['power_%s' % (final_name)] = power_col
    return data
