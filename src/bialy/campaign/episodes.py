"""Bind whole-episode judgments to captured stores and registered sessions."""
import json

from .grading import episode
from .journal import CampaignFault, canonical, digest


class Episodes:
    def __init__(self, journal, lineage):
        self.db, self.lineage = journal.db, lineage
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS episodes(epoch TEXT NOT NULL, root_session TEXT NOT NULL,
          task_id TEXT NOT NULL, store_sha256 TEXT NOT NULL, evidence TEXT NOT NULL,
          evidence_sha256 TEXT NOT NULL, PRIMARY KEY(epoch, root_session));
        CREATE TABLE IF NOT EXISTS episode_sessions(epoch TEXT NOT NULL, session_id TEXT NOT NULL,
          root_session TEXT NOT NULL, PRIMARY KEY(epoch,session_id));
        ''')

    def record(self, epoch, root_session, task_id, sessions, specification, receipt, reviews, grader):
        task = self.lineage.task(task_id)
        if digest(specification) != task['spec_hash']:
            raise CampaignFault('episode_task_specification_changed')
        if (receipt.get('root_session') != root_session or receipt.get('epoch') != epoch
                or not receipt.get('store_sha256') or not sessions or root_session not in sessions
                or len(sessions) != len(set(sessions))):
            raise CampaignFault('episode_session_binding_invalid')
        verdict = episode(specification, receipt, reviews, grader)
        evidence = {'specification': specification, 'receipt': receipt, 'reviews': reviews,
                    'grader': grader, 'verdict': verdict, 'sessions': sorted(sessions)}
        with self.db:
            prior = self.db.execute('SELECT evidence_sha256 FROM episodes WHERE epoch=? AND root_session=?',
                                    (epoch, root_session)).fetchone()
            if prior and prior[0] != digest(evidence):
                raise CampaignFault('episode_evidence_changed')
            # Validate every binding before publishing the episode.
            for session in sessions:
                bound = self.lineage.session_task(epoch, session)
                if bound['root'] != task['root'] or bound['partition'] != task['partition']:
                    raise CampaignFault('episode_worker_lineage_mismatch')
                previous = self.db.execute('SELECT root_session FROM episode_sessions WHERE epoch=? AND session_id=?',
                                           (epoch, session)).fetchone()
                if previous and previous[0] != root_session:
                    raise CampaignFault('session_episode_changed')
            self.db.execute('INSERT OR IGNORE INTO episodes VALUES(?,?,?,?,?,?)',
                            (epoch, root_session, task_id, receipt['store_sha256'], canonical(evidence), digest(evidence)))
            self.db.executemany('INSERT OR IGNORE INTO episode_sessions VALUES(?,?,?)',
                                [(epoch, session, root_session) for session in sessions])
        return verdict

    def admit(self, epoch, session, store_sha256):
        row = self.db.execute('''SELECT episodes.* FROM episode_sessions JOIN episodes
            ON episodes.epoch=episode_sessions.epoch AND episodes.root_session=episode_sessions.root_session
            WHERE episode_sessions.epoch=? AND episode_sessions.session_id=?''', (epoch, session)).fetchone()
        if row is None or row['store_sha256'] != store_sha256:
            raise CampaignFault('episode_missing_or_store_changed')
        evidence = json.loads(row['evidence'])
        if digest(evidence) != row['evidence_sha256']:
            raise CampaignFault('episode_evidence_changed')
        verdict = episode(evidence['specification'], evidence['receipt'], evidence['reviews'], evidence['grader'])
        if verdict != evidence['verdict'] or verdict['verdict'] != 'pass':
            raise CampaignFault('episode_not_qualified')
        return {'episode_sha256': verdict['episode_sha256'], 'evidence_sha256': row['evidence_sha256'],
                'root_session': row['root_session']}
