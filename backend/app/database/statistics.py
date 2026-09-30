"""Deterministic statistics. No model imports or LLM calls."""
import argparse
import json
from collections import Counter
from datetime import datetime

from app.classifiers.workplace_aspects import WORKPLACE_ASPECTS
from app.database.corpus import load_corpus


def calculate(documents: list[dict]) -> dict:
    sources, dates = Counter(), Counter()
    sentiments = {key: set() for key in ('positive', 'negative', 'neutral', 'mixed')}
    aspect_documents = {key: {} for key in WORKPLACE_ASPECTS}
    for doc in documents:
        sources[doc['source']] += 1
        published = doc.get('published_at')
        dates[datetime.fromisoformat(published.replace('Z', '+00:00')).strftime('%Y-%m') if published else 'unknown'] += 1
        for finding in doc['aspects']:
            aspect_documents[finding['aspect']][doc['document_id']] = finding['sentiment']
            sentiments[finding['sentiment']].add(doc['document_id'])
    aspects = []
    for aspect, matches in aspect_documents.items():
        counts = Counter(matches.values())
        aspects.append({'aspect': aspect, 'total_documents': len(matches),
                        **{key: counts[key] for key in sentiments},
                        'negative_ratio_percent': round(100 * counts['negative'] / len(matches), 1) if matches else None})
    return {'document_count': len(documents),
            'documents_by_sentiment': {key: len(ids) for key, ids in sentiments.items()},
            'aspect_frequency': aspects, 'source_frequency': dict(sources),
            'date_distribution': dict(sorted(dates.items())),
            'definitions': 'Sentiment document counts may overlap across aspects. Mixed is separate. '
                           'Negative ratio = negative document/aspect rows divided by all rows for that aspect. '
                           'Dates use publication month; unknown dates remain unknown.'}


def statistics(company, analysis_id=None):
    run, documents = load_corpus(company, analysis_id)
    return {'analysis_id': run, 'company': company, **calculate(documents)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--company', default='Microsoft')
    parser.add_argument('--analysis-id')
    args = parser.parse_args()
    print(json.dumps(statistics(args.company, args.analysis_id), indent=2))
