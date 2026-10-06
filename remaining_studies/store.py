from generative_assembly.storage import Store, read


class SuiteStore(Store):
    """Include frozen teacher ancestry in student checkpoint job identities."""
    def jobs(self, stage=None, split=None):
        rows=super().jobs(stage,split)
        if stage=='E6_TRAIN':
            latest={}
            for row in rows:
                if row['arm'] not in latest or row['started']>latest[row['arm']]['started']:
                    latest[row['arm']]=row
            return list(latest.values())
        return rows
    def run(self, stage, case, arm, fn, **kwargs):
        if stage == 'E6_TRAIN':
            path = self.root/'pseudo_labels'/'train.json'
            labels = read(path)['labels'] if path.exists() else []
            lookup = {r['job_id']: r for r in self.jobs()}
            teacher_ids = sorted({label['teacher_job'] for label in labels})
            kwargs['parents'] = list(kwargs.get('parents', [])) + [lookup[jid] for jid in teacher_ids]
        return super().run(stage, case, arm, fn, **kwargs)
