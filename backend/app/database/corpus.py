"""Read immutable document snapshots belonging to one completed analysis."""
import psycopg
from psycopg.rows import dict_row

from app.config import database_url


def load_corpus(company: str, analysis_id: str | None = None):
    with psycopg.connect(database_url(), row_factory=dict_row, connect_timeout=10) as connection:
        query = '''SELECT a.id, c.name FROM analyses a JOIN companies c ON c.id=a.company_id
                   WHERE a.status='completed' AND lower(c.name)=lower(%s)'''
        parameters = [company]
        if analysis_id:
            query += ' AND a.id=%s'
            parameters.append(analysis_id)
        run = connection.execute(query + ' ORDER BY a.completed_at DESC LIMIT 1', parameters).fetchone()
        if not run:
            raise LookupError('No completed analysis for this company')
        rows = connection.execute('''SELECT ad.snapshot, ad.role,
            COALESCE(jsonb_agg(jsonb_build_object('aspect',p.aspect,'sentiment',p.sentiment,
                'pain_points',p.pain_points,'evidence',p.evidence) ORDER BY p.aspect) FILTER (WHERE p.id IS NOT NULL), '[]') AS aspects
            FROM analysis_documents ad LEFT JOIN aspects p
              ON p.analysis_id=ad.analysis_id AND p.document_id=ad.document_id AND p.company_id=ad.company_id
            WHERE ad.analysis_id=%s GROUP BY ad.document_id,ad.company_id,ad.snapshot,ad.role
            ORDER BY ad.document_id''', (run['id'],)).fetchall()
        if not rows:
            raise LookupError('Analysis has no document snapshots; rerun Checkpoint 4 storage after applying schema')
    documents = [{**row['snapshot'], 'role': row['role'], 'aspects': row['aspects']} for row in rows]
    return str(run['id']), documents
