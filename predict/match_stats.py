"""
    Builds statistics about soccer games from touch by touch data.

    For the public version, the raw tables aren't available, but
    we're including the queries here in order to show what kind
    of data is available and what kind of statistics are possible.
"""

# Raw touch-by-touch BigQuery table. This table contains data licensed from Opta
# and so cannot be shared widely.
# _TOUCH_TABLE = 'cloude-sandbox:toque.touches'
# Set the touch table to None to use the summary table.
_TOUCH_TABLE = None

# Table containing games that were played. Allows us to map team ids to team
# names and figure out which team was home and which team was away.
_MATCH_GAMES_TABLE = """
    SELECT * FROM [cloude-sandbox:public.match_games_table]
"""

# Computing the score of the game is actually fairly tricky to do from the
# touches table, so we use this view to keep track of game score.
_MATCH_GOALS_TABLE = """
    SELECT * FROM [cloude-sandbox:public.match_goals_table_20140708]
"""

# View that computes the statistics (features) about the games.
GAME_SUMMARY = """
    SELECT * FROM [cloude-sandbox:public.team_game_summary_20140708]
"""

# Event type ids from the touches table.
_PASS_ID = 1
_FOUL_ID = 4
_CORNER_ID = 6
_SHOT_IDS = [13, 14, 15, 16]
_SHOT_ID_STRINGS = ','.join([str(sid) for sid in _SHOT_IDS])
_GOAL_ID = 16
_CARD_ID = 17
_HALF_ID = 32
_GAME_ID = 34

# Qualifiers
_OWN_GOAL_QUALIFIER_ID = 28

# Computes the expected goal statistic, based on shot location.
_EXPECTED_GOALS_MATCH = """
SELECT matchid, teamid,
  COUNT(eventid) as sot,
  SUM(typeid = 16) as goals,
  AVG(dist) as avgdist,
  sum(pG) as xG
FROM (SELECT *, 
  IF(dist = 1, 1, 1.132464 - 0.303866* LN(dist)) as pG
FROM (SELECT matchid, teamid, playerid, eventid, typeid, x, y,
      ROUND(SQRT(POW((100-x),2) + POW((50-y),2))) as dist,
      IF(typeid = 16,1,0) as goal
FROM (SELECT matchid, teamid, playerid, eventid, typeid, x, y,
    SUM(qualifiers.type = 82) WITHIN RECORD AS blck,
    SUM(qualifiers.type IN (26,9)) WITHIN RECORD AS penfk,
    SUM(qualifiers.type = 28) WITHIN RECORD AS og,
   FROM [%(touch_table)s] 
   WHERE typeid IN (14, 15,16))
WHERE blck = 0 AND og = 0 AND penfk = 0)
WHERE dist < 40)
GROUP BY matchid, teamid
ORDER BY matchid, teamid
   """ % {'touch_table': _TOUCH_TABLE}

# Subquery to compute raw number of goals scored. Does not take
# into account own-goals (i.e. if a player scores an own-goal against
# his own team, it will count as a goal for that team.
# Computes passing statistics. 
# pass_80: Number of passes completed in the attacking fifth of the field.
# pass_70: NUmber of passes completed in the attacking third of the field.
_PASS_STATS = """
SELECT matchid, teamid, SUM(pass_80) as pass_80, SUM(pass_70) as pass_70
FROM (
   SELECT matchid, teamid, outcomeid, 
   if (x > 80 and outcomeid = 1, 1, 0) as pass_80,
   if (x > 70 and outcomeid = 1, 1, 0) as pass_70 
  FROM [%(touch_table)s] WHERE typeid = %(pass)s)
GROUP BY matchid, teamid
""" % {'touch_table' : _TOUCH_TABLE,
       'pass': _PASS_ID}

# Subquery that tracks own goals so we can later attribute them to
# the other team.
_OWN_GOALS_BY_TEAM_SUBQUERY = """
SELECT matchid, teamid, count(*) as own_goals
FROM [%(touch_table)s] 
WHERE typeid = %(goal)d AND qualifiers.type = %(own_goal)d 
GROUP BY matchid, teamid
""" % {'touch_table': _TOUCH_TABLE,
       'goal': _GOAL_ID, 
       'own_goal': _OWN_GOAL_QUALIFIER_ID}

# Subquery that credits an own goal to the opposite team.
_OWN_GOAL_CREDIT_SUBQUERY = """
SELECT games.matchid as matchid, 
  games.teamid as credit_team,
  og.teamid as deduct_team,
  og.own_goals as cnt
FROM 
(%(own_goals)s) og
JOIN (
  SELECT matchid, teamid, 
  FROM [%(touch_table)s]
  GROUP BY matchid, teamid) games
ON og.matchid = games.matchid 
WHERE games.teamid <> og.teamid
""" % {'touch_table': _TOUCH_TABLE,
       'own_goals': _OWN_GOALS_BY_TEAM_SUBQUERY}

# Simple query that computes the number of goals in a game
# (not counting penalty kicks that occur after a draw).
# This is not sufficient to determine the score, since the
# data will attribute own goals to the wrong team.
_RAW_GOAL_AND_GAME_SUBQUERY = """
SELECT  matchid, teamid, goal, game, timestamp,
FROM (
  SELECT matchid, teamid, 
    if (typeid == %(goal)d and periodid != 5, 1, 0) as goal,      
    if (typeid == %(game)d, 1, 0) as game,
    eventid,
    timestamp,
  FROM [%(touch_table)s]
  WHERE typeid in (%(goal)d, %(game)d))
""" % {'goal': _GOAL_ID, 
       'game': _GAME_ID,
       'touch_table': _TOUCH_TABLE}

# Score by game and team, not adjusted for own goals.
_RAW_GOAL_BY_GAME_AND_TEAM_SUBQUERY = """
SELECT matchid, teamid, SUM(goal) as goals, 
    MAX(TIMESTAMP_TO_USEC(timestamp)) as timestamp,
FROM (%s)
GROUP BY matchid, teamid
""" % (_RAW_GOAL_AND_GAME_SUBQUERY)

# Compute the number of goals in the game. To do this, we want to subtract off
# any own goals a team scored against themselves, and add the own goals that a
# team's opponent scored.
MATCH_GOALS_QUERY = """
SELECT matchid, teamid , goals + delta as goals, timestamp as timestamp 
FROM (
    SELECT goals.matchid as matchid , goals.teamid as teamid,
        goals.goals as goals,
        goals.timestamp as timestamp,
        if (cr.cnt is not NULL, INTEGER(cr.cnt), INTEGER(0)) 
            - if (de.cnt is not NULL, INTEGER(de.cnt), INTEGER(0)) as delta
    FROM (%s) goals
    LEFT OUTER JOIN (%s) cr
    ON goals.matchid = cr.matchid and goals.teamid = cr.credit_team
    LEFT OUTER JOIN (%s) de
    ON goals.matchid = de.matchid and goals.teamid = de.deduct_team
)
""" % (_RAW_GOAL_BY_GAME_AND_TEAM_SUBQUERY,
       _OWN_GOAL_CREDIT_SUBQUERY,
       _OWN_GOAL_CREDIT_SUBQUERY)

# Query that summarizes statistics by team and by game.
# Statistics computed are:
# passes: number of passes per minute completed in the game.
# bad_passes: number of passes per minute that were not completed.
# pass_ratio: proportion of passes that were completed.
# corners: number of corner kicks awarded per minute.
# shots: number of shots taken per minute.
# fouls: number of fouls committed per minute.
# cards: number of cards (yellow or red) against members of the team.
# pass_{70,80}: number of completed passes per minute in the attacking {70,80%}
#     zone.
# is_home: whether this game was played at home.
# expected_goals: number of goals expected given the number and location 
#     of shots on goal.
# on_target: number of shots on target per minute
_TEAM_GAME_SUMMARY = """
SELECT 
t.matchid as matchid,
t.teamid as teamid,
t.passes / t.length as passes,
t.bad_passes / t.length as bad_passes,
t.passes / (t.passes + t.bad_passes + 1) as pass_ratio,
t.corners / t.length as corners,
t.fouls / t.length  as fouls,
t.shots / t.length  as shots,
t.cards as cards,
p.pass_80 / t.length as pass_80,
p.pass_70 / t.length as pass_70,
TIMESTAMP_TO_MSEC(t.timestamp) as timestamp,
g.goals as goals,
h.is_home as is_home,
h.team_name as team_name,
h.competitionid as competitionid,
h.seasonid as seasonid,
if (x.xG is not null, x.xG, 0.0) as expected_goals,
if (x.sot is not null, INTEGER(x.sot), INTEGER(0)) / t.length as on_target,
t.length as length
FROM (
 SELECT matchid, teamid,
      sum(pass) as passes,
      sum(bad_pass) as bad_passes,
      sum (corner) as corners,
      sum (foul) as fouls,      
      sum(shots) as shots,
      sum(cards) as cards,
      max(timestamp) as timestamp,
      max([min]) as length,
      1  as games,     
  FROM (
    SELECT matchid, teamid,       
      timestamp, [min],
      if (typeid == %(pass_id)d and outcomeid = 1, 1, 0) as pass,
      if (typeid == %(pass_id)d and outcomeid = 0, 1, 0) as bad_pass,
      if (typeid == %(foul)d and outcomeid = 1, 1, 0) as foul,
      if (typeid == %(corner)d and outcomeid = 1, 1, 0) as corner,
      if (typeid == %(half)d, 1, 0) as halves,
      if (typeid in (%(shots)s), 1, 0) as shots,
      if (typeid == %(card)d, 1, 0) as cards,                             
    FROM [%(touch_table)s]  as t    
    WHERE teamid != 0)    
 GROUP BY matchid, teamid 
) t
LEFT OUTER JOIN (%(match_goals)s) as g
ON t.matchid = g.matchid and t.teamid = g.teamid
JOIN
(%(pass_stats)s) p
ON
t.matchid = p.matchid and t.teamid = p.teamid
JOIN  (%(match_games)s) h
ON t.matchid = h.matchid AND t.teamid = h.teamid
LEFT OUTER JOIN (%(expected_goals)s) x
ON t.matchid = x.matchid AND t.teamid = x.teamid
""" % {'pass_id': _PASS_ID,
       'foul': _FOUL_ID,
       'corner': _CORNER_ID,
       'half': _HALF_ID,
       'shots':  _SHOT_ID_STRINGS,
       'card': _CARD_ID,
       'pass_stats': _PASS_STATS,
       'expected_goals': _EXPECTED_GOALS_MATCH,
       'touch_table': _TOUCH_TABLE,
       'match_games': _MATCH_GAMES_TABLE,
       'match_goals': _MATCH_GOALS_TABLE}


# Some of the games in the touches table have been ingested twice. If that
# is the case, scale the game statistics.
_TEAM_GAME_SUMMARY_CORRECTED = """
SELECT 
t.matchid as matchid,
t.teamid as teamid,
t.passes / s.event_count as passes,
t.bad_passes / s.event_count as bad_passes,
t.pass_ratio as pass_ratio,
t.corners / s.event_count as corners,
t.fouls / s.event_count as fouls,
t.shots / s.event_count as shots,
t.cards / s.event_count as cards,
t.pass_80 / s.event_count as pass_80,
t.pass_70 / s.event_count as pass_70,
t.timestamp as timestamp,
t.goals / s.event_count as goals,
t.is_home as is_home,
t.team_name as team_name,
t.competitionid as competitionid,
t.seasonid as seasonid,
t.expected_goals / s.event_count as expected_goals,
t.on_target / s.event_count as on_target,
t.length as length,

FROM  (%(team_game_summary)s) t
JOIN (
SELECT matchid, MAX(event_count) as event_count
FROM (
    SELECT matchid, COUNT(eventid) as event_count  
    FROM [%(touches_table)s]
    GROUP EACH BY matchid, eventid
) GROUP BY matchid
) s ON t.matchid = s.matchid
""" % {
       'team_game_summary': _TEAM_GAME_SUMMARY,
       'touches_table': _TOUCH_TABLE}

###
### Public queries / methods
###

def team_game_summary_query(): 
    """ Query that returns query statistics for both teams in a game. """
    if _TOUCH_TABLE:
        return _TEAM_GAME_SUMMARY_CORRECTED
    else:
        return GAME_SUMMARY
  
def match_goals_table():
    """ Returns the name of a table with goals scored per match. """
    return _MATCH_GOALS_TABLE

def match_games_table():
    """ Returns the name of a table containing basic data about matches. """
    return _MATCH_GAMES_TABLE

def get_non_feature_columns():
    """ Returns a list of the columns that are in our features dataframe that
        should not be used in prediction. These are essentially either metadata
        columns (team name, for example), or potential target variables that
        include the outcome. We want to make sure not to use the latter, since
        we don't want to use information about the current game to predict that
        same game.
    """
    return ['teamid', 'op_teamid', 'matchid', 'competitionid', 'seasonid',
            'goals', 'op_goals', 'points', 'timestamp', 'team_name', 
            'op_team_name']

def get_feature_columns(all_cols):
    """ Returns a list of all columns that should be used in prediction
        (i.e. all features that are in the dataframe but are not in the 
        features.get_non_feature_column() list).
    """
    return [col for col in all_cols if col not in get_non_feature_columns()]

