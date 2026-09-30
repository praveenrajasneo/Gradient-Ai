"""Print real negative-aspect document counts for one completed analysis."""

import argparse
import json

import psycopg
from psycopg.rows import dict_row

from app.config import database_url


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-id')
    args = parser.parse_args()
    with psycopg.connect(database_url(), row_factory=dict_row) as connection:
        if args.analysis_id:
            run = connection.execute("SELECT id FROM analyses WHERE id=%s AND status='completed'",
                                     (args.analysis_id,)).fetchone()
        else:
            run = connection.execute("SELECT id FROM analyses WHERE status='completed' ORDER BY completed_at DESC LIMIT 1").fetchone()
        if not run:
            raise RuntimeError('No completed analysis found')
        counts = connection.execute("""SELECT aspect, COUNT(*) AS negative_documents FROM aspects
            WHERE sentiment='negative' AND analysis_id=%s GROUP BY aspect
            ORDER BY negative_documents DESC, aspect""", (run['id'],)).fetchall()
        print(json.dumps({'analysis_id': str(run['id']), 'negative_aspects': counts}, indent=2))


if __name__ == '__main__':
    main()
