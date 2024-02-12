import duckdb
import numpy as np
import pandas as pd
from tqdm import tqdm

from relbench.data import Database, RelBenchLinkTask, RelBenchNodeTask, Table
from relbench.data.task_base import TaskType
from relbench.metrics import accuracy, average_precision, f1, mae, rmse, roc_auc
from relbench.utils import get_df_in_window

######## node prediction tasks ########


class EngageTask(RelBenchNodeTask):
    r"""Predict if a user will make any votes/posts/comments in the next 2 years."""

    name = "rel-stackex-engage"
    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "OwnerUserId"
    entity_table = "users"
    time_col = "timestamp"
    target_col = "contribution"
    timedelta = pd.Timedelta(days=365 * 2)
    metrics = [average_precision, accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        comments = db.table_dict["comments"].df
        votes = db.table_dict["votes"].df
        posts = db.table_dict["posts"].df
        users = db.table_dict["users"].df

        df = duckdb.sql(
            f"""
            WITH
            ALL_ENGAGEMENT AS (
                SELECT
                    p.id,
                    p.owneruserid as userid,
                    p.creationdate
                FROM
                    posts p
                UNION
                SELECT
                    v.id,
                    v.userid,
                    v.creationdate
                FROM
                    votes v
                UNION
                SELECT
                    c.id,
                    c.userid,
                    c.creationdate
                FROM
                    comments c
            ),

            ACTIVE_USERS AS (
                 SELECT
                    t.timestamp,
                    u.id,
                    count(distinct a.id) as n_engagement
                FROM timestamp_df t
                CROSS JOIN users u
                LEFT JOIN all_engagement a
                ON u.id = a.UserId
                    and a.CreationDate <= t.timestamp
                WHERE u.id != -1
                GROUP BY t.timestamp, u.id
             )
                SELECT
                    u.timestamp,
                    u.id as OwnerUserId,
                    IF(count(distinct a.id) >= 1, 1, 0) as contribution
                FROM
                    active_users u
                LEFT JOIN
                    all_engagement a
                ON
                    u.id = a.UserId AND
                    a.CreationDate > u.timestamp AND
                    a.CreationDate <= u.timestamp + INTERVAL '{self.timedelta}'
                where
                    u.n_engagement >= 1
                GROUP BY
                    u.timestamp, u.id
            ;

            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class VotesTask(RelBenchNodeTask):
    r"""Predict the number of upvotes that an existing question will receive in
    the next 2 years."""
    name = "rel-stackex-votes"
    task_type = TaskType.REGRESSION
    entity_col = "PostId"
    entity_table = "posts"
    time_col = "timestamp"
    target_col = "popularity"
    timedelta = pd.Timedelta(days=365 * 2)
    metrics = [mae, rmse]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        votes = db.table_dict["votes"].df
        posts = db.table_dict["posts"].df

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                p.id AS PostId,
                COUNT(distinct v.id) AS popularity
            FROM
                timestamp_df t
            LEFT JOIN
                posts p
            ON
                p.CreationDate <= t.timestamp AND
                p.owneruserid != -1 AND
                p.owneruserid is not null AND
                p.PostTypeId = 1
            LEFT JOIN
                votes v
            ON
                p.id = v.PostId AND
                v.CreationDate > t.timestamp AND
                v.CreationDate <= t.timestamp + INTERVAL '{self.timedelta}' AND
                v.votetypeid = 2
            GROUP BY
                t.timestamp,
                p.id
            ;

            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class BadgesTask(RelBenchNodeTask):
    r"""Predict if each user will receive in a new badge the next 2 years."""
    name = "rel-stackex-badges"
    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "UserId"
    entity_table = "users"
    time_col = "timestamp"
    target_col = "WillGetBadge"
    timedelta = pd.Timedelta(days=365 * 2)
    metrics = [average_precision, accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        users = db.table_dict["users"].df
        badges = db.table_dict["badges"].df

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                u.Id as UserId,
            CASE WHEN
                COUNT(b.Id) >= 1 THEN 1 ELSE 0 END AS WillGetBadge
            FROM
                timestamp_df t
            LEFT JOIN
                users u
            ON
                u.CreationDate <= t.timestamp
            LEFT JOIN
                badges b
            ON
                u.Id = b.UserID
                AND b.Date > t.timestamp
                AND b.Date <= t.timestamp + INTERVAL '{self.timedelta}'
            GROUP BY
                t.timestamp,
                u.Id
            """
        ).df()

        # remove any IderId rows that are NaN
        df = df.dropna(subset=["UserId"])
        df[self.entity_col] = df[self.entity_col].astype(
            int
        )  # for some reason duckdb returns float64 keys

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


######## link prediction tasks ########


class UserCommentOnPostTask(RelBenchLinkTask):
    r"""Predict if a user will comment on a specific post within 24hrs of the post being made."""

    name = "rel-stackex-comment-on-post"
    task_type = TaskType.LINK_PREDICTION
    source_entity_col = "UserId"
    source_entity_table = "users"
    destination_entity_col = "PostId"
    destination_entity_table = "posts"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=365 * 2)
    metrics = None  # TODO: add metrics

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        r"""Create Task object for UserCommentOnPostTask."""
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        users = db.table_dict["users"].df
        posts = db.table_dict["posts"].df
        comments = db.table_dict["comments"].df

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                c.UserId as UserId,
                LIST(DISTINCT p.id) AS PostId
            FROM
                timestamp_df t
            LEFT JOIN
                posts p
            ON
                p.CreationDate <= t.timestamp
            LEFT JOIN
                comments c
            ON
                p.id = c.PostId AND
                c.CreationDate > t.timestamp AND
                c.CreationDate <= t.timestamp + INTERVAL '{self.timedelta} days'
            WHERE
                c.UserId is not null AND
                p.owneruserid != -1 AND
                p.owneruserid is not null
            ;
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={
                self.source_entity_col: self.source_entity_table,
                self.destination_entity_col: self.destination_entity_table,
            },
            pkey_col=None,
            time_col=self.time_col,
        )
