#!/usr/bin/python2.7
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
    Predicts soccer outcomes using logistic regression.
"""

import random
import math

import numpy as np
import pandas as pd
import pylab as pl
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import statsmodels.api as sm


def _drop_unbalanced_matches(data):
    """  Because we don't have data on both teams during a match, we
         want to drop any match we don't have info about both teams. 
         This can happen if we have fewer than 10 previous games from
         a particular team.
    """
    keep = []
    index = 0
    data = data.dropna()
    while index < len(data) - 1:
        skipped = False
        for col in data:
            if isinstance(col, float) and math.isnan(col):
                keep.append(False)
                index += 1
                skipped = True
             
        if skipped:
            pass
        elif data.iloc[index]['matchid'] == data.iloc[index+1]['matchid']:
            keep.append(True)
            keep.append(True)
            index += 2
        else:
            keep.append(False)
            index += 1
    while len(keep) < len(data):
        keep.append(False)
    results = data[keep]
    if len(results) % 2 != 0:
        raise Exception('Unexpected results')
    return results


def _swap_pairwise(col):
    """ Swap rows pairwise; i.e. swap row 0 and 1, 2 and 3, etc.  """
    col = pd.np.array(col)
    for index in xrange(0, len(col), 2):
        val = col[index]
        col[index] = col[index + 1]
        col[index+1] = val
    return col


def _splice(data):
    """ Splice both rows representing a game into a single one. """
    data = data.copy()
    opp = data.copy()
    opp_cols = ['opp_%s' % (col,) for col in opp.columns]
    opp.columns = opp_cols
    opp = opp.apply(_swap_pairwise)
    del opp['opp_is_home']
    
    return data.join(opp)


def split(data, test_proportion=0.4):
    """ Splits a dataframe into a training set and a test set.
        Must be careful because back-to-back rows are expeted to
        represent the same game, so they both must go in the 
        test set or both in the training set.
    """
    
    train_vec = []
    if len(data) % 2 != 0:
        raise Exception('Unexpected data length')
    while len(train_vec) < len(data):
        rnd = random.random()
        train_vec.append(rnd > test_proportion) 
        train_vec.append(rnd > test_proportion)
            
    test_vec = [not val for val in train_vec]
    train = data[train_vec]
    test = data[test_vec]
    if len(train) % 2 != 0:
        raise Exception('Unexpected train length')
    if len(test) % 2 != 0:
        raise Exception('Unexpected test length')
    return (train, test)


def _extract_target(data, target_col):
    """ Removes the target column from a data frame, returns the target
        col and a new data frame minus the target. """
    target = data[target_col]
    train_df = data.copy()
    del train_df[target_col]
    return target, train_df


def _check_eq(value): 
    """ Returns a function that checks whether the value equals a
        particular integer.
    """
    return lambda (x): int(x) == int(value)


L1_ALPHA = 16.0
def build_model_logistic(target, data, acc=0.00000001, alpha=L1_ALPHA):
    """ Trains a logistic regresion model. target is the target.
        data is a dataframe of samples for training. The length of 
        target must match the number of rows in data.
    """ 
    data = data.copy()
    data['intercept'] = 1.0
    logit = sm.Logit(target, data, disp=False)
    return logit.fit_regularized(maxiter=1024, alpha=alpha, acc=acc, disp=False)


def validate(label, target, predictions, baseline=0.5, compute_auc=False,
             quiet=True):
    """ Validates binary predictions, computes confusion matrix and AUC.

      Given a vector of predictions and actual values, scores how well we
      did on a prediction. 

      Args:
        label: label of what we're validating
        target: vector of actual results
        predictions: predicted results. May be a probability vector,
          in which case we'll sort it and take the most confident values
          where baseline is the proportion that we want to take as True
          predictions. If a prediction is 1.0 or 0.0, however, we'll take
          it to be a true or false prediction, respectively.
        compute_auc: If true, will compute the AUC for the predictions. 
          If this is true, predictions must be a probability vector.
    """

    if len(target) != len(predictions):
        raise Exception('Length mismatch %d vs %d' % (len(target), 
                                                      len(predictions)))
    if baseline > 1.0:
        # Baseline number is expected count, not proportion. Get the proportion.
        baseline = baseline * 1.0 / len(target)

    zipped = sorted(zip(target, predictions), key=lambda tup: -tup[1])
    expect = len(target) * baseline
    
    (true_pos, true_neg, false_pos, false_neg) = (0, 0, 0, 0)
    for index in xrange(len(target)):
        (yval, prob) = zipped[index]
        if float(prob) == 0.0:
            predicted = False
        elif float(prob) == 1.0:
            predicted = True
        else:
            predicted = index < expect
        if predicted:
            if yval:
                true_pos += 1
            else:
                false_pos += 1 
        else:
            if yval:
                false_neg += 1
            else:
                true_neg += 1
    pos = true_pos + false_neg
    neg = true_neg + false_pos
    # P(1 | predicted(1)) and P(0 | predicted(f))
    pred_t = true_pos + false_pos
    pred_f = true_neg + false_neg
    prob1_t = true_pos * 1.0 / pred_t if pred_t > 0.0 else -1.0
    prob0_f = true_neg * 1.0 / pred_f if pred_f > 0.0 else -1.0
              
    # Lift = P(1 | t) / P(1)
    prob_1 = pos * 1.0 / (pos + neg)
    lift = prob1_t / prob_1 if prob_1 > 0 else 0.0
              
    accuracy = (true_pos + true_neg) * 1.0 / len(target)
              
    if compute_auc:
        y_bool =  [True if yval else False for (yval, _) in zipped]
        x_vec = [xval for (_, xval) in zipped]
        auc_value = roc_auc_score(y_bool, x_vec)
        fpr, tpr, _ = roc_curve(y_bool, x_vec)
        pl.plot(fpr, tpr, lw=1.5,
            label='ROC %s (area = %0.2f)' % (label, auc_value))
        pl.xlabel('False Positive Rate', fontsize=18)
        pl.ylabel('True Positive Rate', fontsize=18)
        pl.title('ROC curve', fontsize=18)
        auc_value = '%0.03g' % auc_value
    else:
        auc_value = 'NA'

    print '(%s) Lift: %0.03g Auc: %s' % (label, lift, auc_value)
    if not quiet:
        print '    Base: %0.03g Acc: %0.03g P(1|t): %0.03g P(0|f): %0.03g' % (
            baseline, accuracy, prob1_t, prob0_f)
        print '    Fp/Fn/Tp/Tn p/n/c: %d/%d/%d/%d %d/%d/%d' % (
            false_pos, false_neg, true_pos, true_neg, pos, neg, len(target))
    

def _coerce_types(vals):
    """ Makes sure all of the values in a list are floats. """
    return [1.0 * val for val in vals]


def _coerce(data): 
    """ Coerces a dataframe to all floats, and standardizes the values. """
    return _standardize(data.apply(_coerce_types))


def _standardize_col(col):
    """ Standardizes a single column (subtracts mean and divides by std
        dev). 
    """
    std = np.std(col)
    mean = np.mean(col)
    if abs(std) > 0.001:
        return col.apply(lambda val: (val - mean)/std)
    else:
        return col


def _standardize(data):
    """ Standardizes a dataframe. All fields must be numeric. """
    return data.apply(_standardize_col)


def _clone_and_drop(data, drop_cols):
    """ Returns a copy of a dataframe that doesn't have certain columns. """
    clone = data.copy()
    for col in drop_cols:
        if col in clone.columns:
            del clone[col]
    return clone


def _normalize(vec):
    """ Normalizes a list so that the total sum is 1. """
    total = float(sum(vec))
    return [val / total for val in vec]


def _games(data):
    """ Drops odd numbered rows in a column. This is used when we
        have two rows representing a game, and we only need 1. """
    return data[[idx % 2 == 0 for idx in xrange(len(data))]] 
  

def _team_test_prob(target):
    """ We predict both team A beating team B and team B beating 
        team A. Use predictions in both directions to come up with
        an overall probability.
    """
    results = []
    for idx in range(len(target)/2):
        game0 = float(target.iloc[idx*2])
        game1 = float(target.iloc[idx*2+1])
        results.append(game0/(game0+game1))
    return results


def extract_predictions(data, predictions):
    """' Joins a dataframe containing match data with one
         containing predictions, returning a dataframe with
         team names, predicted values, and if available, the
         actual outcome (in points).
    """
    probs = _team_test_prob(predictions)
    teams0 = []
    teams1 = []
    points = []
    for game in xrange(len(data)/2):
        if data['matchid'].iloc[game*2] != data['matchid'].iloc[game*2+1]:
            raise Exception('Unexpeted match id %d vs %d', (
                               data['matchid'].iloc[game * 2],
                               data['matchid'].iloc[game * 2 + 1]))
        team0 = data['team_name'].iloc[game * 2]
        team1 = data['op_team_name'].iloc[game * 2]
        if 'points' in data.columns: 
            points.append(data['points'].iloc[game * 2])
        teams0.append(team0)
        teams1.append(team1)
    results =  pd.DataFrame(
        {'team_name': pd.Series(teams0), 
         'op_team_name': pd.Series(teams1),
         'predicted': pd.Series(probs).mul(100)},
         columns = ['team_name', 'op_team_name', 'predicted'])

    expected_winner = []
    for game in xrange(len(results)):
        row = results.iloc[game]
        col = 'team_name' if row['predicted'] >= 50 else 'op_team_name' 
        expected_winner.append(row[col])

    results['expected'] = pd.Series(expected_winner)

    if len(points) > 0:
        winners = []
        for game in xrange(len(results)):
            row = results.iloc[game]
            point = points[game]
            if point > 1.1:
                winners.append(row['team_name'])
            elif point < 0.9:
                winners.append(row['op_team_name'])
            elif point > -0.1:
                winners.append('draw')
            else:
                winners.append('NA')
        results['winner'] = pd.Series(winners)
        results['points'] = pd.Series(points)
    return results


def _check_data(data):
    """ Walks a dataframe and make sure that all is well. """ 
    i = 0
    if len(data) % 2 != 0:
        raise Exception('Unexpeted length')
    matches = data['matchid']
    teams = data['teamid']
    op_teams = data['op_teamid']
    while i < len(data) - 1:
        if matches.iloc[i] != matches.iloc[i + 1]:
            raise Exception('Match mismatch: %s vs %s ' % (
                            matches.iloc[i], matches.iloc[i + 1]))
        if teams.iloc[i] != op_teams.iloc[i + 1]:
            raise Exception('Team mismatch: match %s team %s vs %s' % (
                            matches.iloc[i], teams.iloc[i], 
                            op_teams.iloc[i + 1]))
        if teams.iloc[i + 1] != op_teams.iloc[i]:
            raise Exception('Team mismatch: match %s team %s vs %s' % (
                            matches.iloc[i], teams.iloc[i + 1], 
                            op_teams.iloc[i]))
        i += 2


def prepare_data(data):
    """ Drops all matches where we don't have data for both teams. """
    data = data.copy()
    data = _drop_unbalanced_matches(data)
    _check_data(data)
    return data


def train_model(data, ignore_cols):
    """ Trains a logistic regression model over the data. Columns that 
        are passed in ignore_cols are considered metadata and not used
        in the model building.
    """
    # Validate the data
    data = prepare_data(data)
    target_col = 'points'
    (train, test) = split(data)
    (y_train, x_train) = _extract_target(train, target_col)
    x_train2 = _splice(_coerce(_clone_and_drop(x_train, ignore_cols)))

    y_train2 = [int(yval) == 3 for yval in y_train]
    model = build_model_logistic(y_train2, x_train2, alpha=8.0)
    return (model, test)


def predict_model(model, test, ignore_cols):
    """ Runs a simple predictor that will predict if we expect a team to 
        win. 
    """
      
    x_test = _splice(_coerce(_clone_and_drop(test, ignore_cols)))
    x_test['intercept'] = 1.0
    predicted =  model.predict(x_test)
    result = test.copy()
    result['predicted'] = predicted
    return result

