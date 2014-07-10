"""
    Turns raw statistics about soccer matches into features we use
    for prediction. Combines a number of games of history to compute
    aggregates that can be used to predict the next game.
"""

from pandas.io import gbq

import match_stats

# Games that have stats available. Not all games in the match_games table
# will have stats (e.g. they might be in the future).
MATCH_GAME_WITH_STATS = """
    SELECT * FROM (%(match_games)s)
    WHERE matchid in (
        SELECT matchid FROM (%(stats_table)s) GROUP BY matchid)
    """ % {'match_games': match_stats.match_games_table(),
           'stats_table': match_stats.team_game_summary_query()}        

# Combines statistics from both teams in a match.
# For each two records matching the pattern (m, t1, <stats1>) and
# (m, t2, <stats2>) where m is the match id, t1 and t2 are the two teams,
# stats1 and stats2 are the statistics for those two teams, combines them
# into a single row (m, t1, t2, <stats1>, <stats2>) where all of the
# t2 field names are decorated with the op_ prefix. For example, teamid becomes
# op_teamid, and pass_70 becomes op_pass_70.
_TEAM_GAME_OP_SUMMARY =  """
    SELECT cur.matchid as matchid,
      cur.teamid as teamid,
      cur.passes as passes,
      cur.bad_passes as bad_passes,
      cur.pass_ratio as pass_ratio,
      cur.corners as corners,
      cur.fouls as fouls,
      cur.cards as cards,
      cur.goals as goals,
      cur.shots as shots,
      cur.is_home as is_home,
      cur.team_name as team_name,
      cur.pass_80 as pass_80,
      cur.pass_70 as pass_70,
      cur.expected_goals as expected_goals,
      cur.on_target as on_target,
      cur.length as length,

      opp.teamid as op_teamid,
      opp.passes as op_passes,
      opp.bad_passes as op_bad_passes,
      opp.pass_ratio as op_pass_ratio,
      opp.corners as op_corners,
      opp.fouls as op_fouls,
      opp.cards as op_cards,
      opp.goals as op_goals,
      opp.shots as op_shots,
      opp.team_name as op_team_name,
      opp.pass_80 as op_pass_80,
      opp.pass_70 as op_pass_70,
      opp.expected_goals as op_expected_goals,
      opp.on_target as op_on_target,

      cur.competitionid as competitionid,
      cur.seasonid as seasonid,

      if (opp.shots > 0, cur.shots / opp.shots, cur.shots * 1.0)
          as shots_op_ratio,
      if (opp.goals > 0, cur.goals / opp.goals, cur.goals * 1.0)
          as goals_op_ratio,
      if (opp.pass_ratio > 0, cur.pass_ratio / opp.pass_ratio, 1.0)
          as pass_op_ratio,

      if (cur.goals > opp.goals, 3,
        if (cur.goals == opp.goals, 1, 0)) as points,
      cur.timestamp as timestamp,

    FROM (%(team_game_summary)s) cur
    JOIN (%(team_game_summary)s) opp
    ON cur.matchid = opp.matchid
    WHERE cur.teamid != opp.teamid
    ORDER BY cur.matchid, cur.teamid
      """ % {'team_game_summary': match_stats.team_game_summary_query()}


def get_match_history(history_size): 
    """ For each team t in each game g, computes the N previous game 
        ids where team t played, where N is the history_size (number
        of games of history we use for prediction). The statistics of
        the N previous games will be used to predict the outcome of 
        game g.
    """
    return """
        SELECT h.teamid as teamid, h.matchid as matchid,
        h.timestamp as timestamp, 
        m1.timestamp as previous_timestamp, 
        m1.matchid as previous_match
        FROM (
            SELECT teamid, matchid, timestamp, 
            LEAD(matchid, 1) OVER (
                PARTITION BY teamid ORDER BY timestamp DESC)
                as last_matchid,
            LEAD(timestamp, 1) OVER (
                PARTITION BY teamid ORDER BY timestamp DESC)
                as last_match_timestamp,
            LEAD(timestamp, %(history_size)d) OVER (
                PARTITION BY teamid ORDER BY timestamp DESC)
                as nth_last_matchid,
            LEAD(timestamp, %(history_size)d) OVER (
                PARTITION BY teamid ORDER BY timestamp DESC)
                as nth_last_match_timestamp,
            FROM (%(match_games)s) 
        ) h
        JOIN (%(match_games_with_stats)s) m1
        ON h.teamid = m1.teamid
        WHERE
        h.nth_last_match_timestamp is not NULL AND
        h.last_match_timestamp IS NOT NULL AND
        m1.timestamp >= h.nth_last_match_timestamp AND 
        m1.timestamp <= h.last_match_timestamp 

        """ % {'history_size': history_size, 
               'match_games_with_stats': MATCH_GAME_WITH_STATS,
               'match_games': match_stats.match_games_table()}

def get_history_query(history_size): 
    """ Computes summary statistics for the N preceeding matches. """
    return """
        SELECT  
            summary.matchid as matchid,
            pts.teamid as teamid,
            pts.op_teamid as op_teamid,
            pts.competitionid as competitionid,
            pts.seasonid as seasonid,
            pts.is_home as is_home,
            pts.team_name as team_name,
            pts.op_team_name as op_team_name,
            pts.timestamp as timestamp,

            summary.avg_points as avg_points,
            summary.avg_goals as avg_goals,
            summary.op_avg_goals as op_avg_goals,

            summary.pass_70 as pass_70,
            summary.pass_80 as pass_80,
            summary.op_pass_70 as op_pass_70,
            summary.op_pass_80 as op_pass_80,
            summary.expected_goals as expected_goals,
            summary.op_expected_goals as op_expected_goals,
            summary.passes as passes,
            summary.bad_passes as bad_passes,
            summary.pass_ratio as pass_ratio,
            summary.corners as corners,
            summary.fouls as fouls,
            summary.cards as cards,
            summary.shots as shots,

            summary.op_passes as op_passes,
            summary.op_bad_passes as op_bad_passes,
            summary.op_corners as op_corners,
            summary.op_fouls as op_fouls,
            summary.op_cards as op_cards,
            summary.op_shots as op_shots,

            summary.goals_op_ratio as goals_op_ratio,
            summary.shots_op_ratio as shots_op_ratio,
            summary.pass_op_ratio as pass_op_ratio,

        FROM (
            SELECT hist.matchid as matchid,
                hist.teamid as teamid,
                AVG(games.pass_70) as pass_70, 
                AVG(games.pass_80) as pass_80, 
                AVG(games.op_pass_70) as op_pass_70, 
                AVG(games.op_pass_80) as op_pass_80, 
                AVG(games.expected_goals) as expected_goals, 
                AVG(games.op_expected_goals) as op_expected_goals, 
                AVG(games.passes) as passes, 
                AVG(games.bad_passes) as bad_passes, 
                AVG(games.pass_ratio) as pass_ratio,
                AVG(games.corners) as corners, 
                AVG(games.fouls) as fouls,
                AVG(games.cards) as cards, 
                AVG(games.goals) as avg_goals, 
                AVG(games.points) as avg_points, 
                AVG(games.shots) as shots,
                AVG(games.op_passes) as op_passes, 
                AVG(games.op_bad_passes) as op_bad_passes, 
                AVG(games.op_corners) as op_corners,
                AVG(games.op_fouls) as op_fouls,
                AVG(games.op_cards) as op_cards,   
                AVG(games.op_shots) as op_shots, 
                AVG(games.op_goals) as op_avg_goals, 
                AVG(games.goals_op_ratio) as goals_op_ratio,
                AVG(games.shots_op_ratio) as shots_op_ratio,
                AVG(games.pass_op_ratio) as pass_op_ratio,
            FROM (%(match_history)s)  hist
            JOIN (%(team_game_op_summary)s) games
            ON hist.previous_match = games.matchid and
                hist.teamid = games.teamid
            GROUP BY matchid, teamid
        ) as summary
        JOIN (%(match_games)s) pts on summary.matchid = pts.matchid
            and summary.teamid = pts.teamid
        WHERE summary.matchid <> '442291'
        ORDER BY matchid, is_home DESC
        """ % {'team_game_op_summary': _TEAM_GAME_OP_SUMMARY,
               'match_games': match_stats.match_games_table(),
               'match_history': get_match_history(history_size)}

def get_history_query_with_goals(history_size):
    """ Expands the history_query, which summarizes statistics from past games
        with the result of who won the current game. This information will not
        be availble for future games that we want to predict, but it will be
        available for past games. We can then use this information to train our
        models.
    """
    return """
        SELECT   
            h.matchid as matchid,
            h.teamid as teamid,
            h.op_teamid as op_teamid,
            h.competitionid as competitionid,
            h.seasonid as seasonid,
            h.is_home as is_home,
            h.team_name as team_name,
            h.op_team_name as op_team_name,
            h.timestamp as timestamp,

            g.goals as goals,
            op.goals as op_goals,
            if (g.goals > op.goals, 3,
              if (g.goals == op.goals, 1, 0)) as points,

            h.avg_points as avg_points,
            h.avg_goals as avg_goals,
            h.op_avg_goals as op_avg_goals,

            h.pass_70 as pass_70,
            h.pass_80 as pass_80,
            h.op_pass_70 as op_pass_70,
            h.op_pass_80 as op_pass_80,
            h.expected_goals as expected_goals,
            h.op_expected_goals as op_expected_goals,
            h.passes as passes,
            h.bad_passes as bad_passes,
            h.pass_ratio as pass_ratio,
            h.corners as corners,
            h.fouls as fouls,
            h.cards as cards,
            h.shots as shots,

            h.op_passes as op_passes,
            h.op_bad_passes as op_bad_passes,
            h.op_corners as op_corners,
            h.op_fouls as op_fouls,
            h.op_cards as op_cards,
            h.op_shots as op_shots,

            h.goals_op_ratio as goals_op_ratio,
            h.shots_op_ratio as shots_op_ratio,
            h.pass_op_ratio as pass_op_ratio,

        FROM (%(history_query)s) h
        JOIN (%(match_goals)s) g
        ON h.matchid = g.matchid and h.teamid = g.teamid
        JOIN (%(match_goals)s) op
        ON h.matchid = op.matchid and h.op_teamid = op.teamid
        ORDER BY timestamp DESC, matchid, is_home 
        """ % {'history_query': get_history_query(history_size),
               'match_goals': match_stats.match_goals_table()}

def get_wc_history_query(history_size): 
    """ Identical to the history_query (which, remember, does not have
        outcomes) but gets history for world-cup games.
    """
    return """
        SELECT * FROM (%(history_query)s) WHERE competitionid = 4
        ORDER BY timestamp DESC, matchid, is_home 
        """ % {'history_query': get_history_query(history_size)}

def get_wc_features(history_size):
    """ Runs a bigquery query that gets the features that can be used
        to predict the world cup.
    """
    return gbq.read_gbq(get_wc_history_query(history_size))

def get_features(history_size):
    """ Runs a BigQuery query to get features that can be used to train
         a machine learning model.
    """
    return gbq.read_gbq(get_history_query_with_goals(history_size))

def get_game_summaries():
    """ Runs a BigQuery Query that gets game summaries. """
    return gbq.read_gbq("""
        SELECT * FROM (%(team_game_op_summary)s) 
        ORDER BY timestamp DESC, matchid, is_home 
        """ % {'team_game_op_summary': _TEAM_GAME_OP_SUMMARY})
