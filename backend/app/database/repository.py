import json
from uuid import NAMESPACE_URL, uuid5

import psycopg
from psycopg.types.json import Jsonb

from app.config import database_url
from app.models.document import Document
from app.models.intelligence import Aspect, PainPoint, Sentiment
from pydantic import TypeAdapter


def store_report(report: dict) -> tuple[str, list[dict]]:
    if report.get("status") != "extracted":
        raise ValueError("Only fully extracted reports can be stored")
    identity = report["input_hash"] + json.dumps(report["config"], sort_keys=True)
    analysis_id = uuid5(NAMESPACE_URL, identity)
    stored_config = {**report['config'], 'extraction_rejections': [
        {'document_id': item['document']['document_id'], **rejection}
        for item in report['documents'] for rejection in item.get('extraction_rejections', [])
    ]}
    documents = [Document.model_validate(item["document"]) for item in report["documents"]]
    if not documents or len({doc.company for doc in documents}) != 1:
        raise ValueError("A report must contain documents for exactly one company")
    with psycopg.connect(database_url(), connect_timeout=10) as connection:
        company_id = connection.execute(
            "INSERT INTO companies(name) VALUES (%s) ON CONFLICT(name) DO UPDATE SET name=excluded.name RETURNING id",
            (documents[0].company,)).fetchone()[0]
        connection.execute(
            """INSERT INTO analyses(id,company_id,input_hash,config,status) VALUES (%s,%s,%s,%s,'running')
               ON CONFLICT(id) DO UPDATE SET status='running', completed_at=NULL, error=NULL, config=excluded.config""",
            (analysis_id, company_id, report["input_hash"], Jsonb(stored_config)))
        # Repeating exactly the same run replaces its observations atomically.
        connection.execute("DELETE FROM aspects WHERE analysis_id=%s", (analysis_id,))
        connection.execute("DELETE FROM analysis_documents WHERE analysis_id=%s", (analysis_id,))
        for document, item in zip(documents, report["documents"]):
            connection.execute("INSERT INTO sources(name,source_type) VALUES (%s,%s) ON CONFLICT DO NOTHING",
                               (document.source, document.source_type))
            connection.execute(
                """INSERT INTO documents(id,company_id,source,source_type,title,text,url,author,published_at,collected_at)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT(id,company_id) DO UPDATE SET title=excluded.title,text=excluded.text,
                   url=excluded.url,author=excluded.author,published_at=excluded.published_at,
                   collected_at=excluded.collected_at""",
                (document.document_id, company_id, document.source, document.source_type,
                 document.title, document.text, str(document.url), document.author,
                 document.published_at, document.collected_at))
            connection.execute(
                'INSERT INTO analysis_documents(analysis_id,document_id,company_id,snapshot) VALUES (%s,%s,%s,%s)',
                (analysis_id, document.document_id, company_id, Jsonb(document.model_dump(mode='json'))))
            for aspect in item["aspects"]:
                TypeAdapter(Aspect).validate_python(aspect["aspect"])
                TypeAdapter(Sentiment).validate_python(aspect["sentiment"])
                for evidence in aspect["evidence"]:
                    if evidence["text"] not in document.text:
                        raise ValueError("Aspect evidence is not present in the document")
                for raw_point in aspect["pain_points"]:
                    point = PainPoint.model_validate(raw_point)
                    if point.aspect != aspect["aspect"] or not any(
                        e["sentiment"] == "negative" and point.evidence_quote in e["text"]
                        for e in aspect["evidence"]
                    ):
                        raise ValueError("Pain point lacks matching negative evidence")
                connection.execute(
                    """INSERT INTO aspects(analysis_id,document_id,company_id,aspect,sentiment,pain_point,
                       confidence,evidence,pain_points) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (analysis_id, document.document_id, company_id, aspect["aspect"], aspect["sentiment"],
                     aspect["pain_point"], aspect["confidence"], Jsonb(aspect["evidence"]), Jsonb(aspect["pain_points"])))
        connection.execute("UPDATE analyses SET status='completed',completed_at=now() WHERE id=%s", (analysis_id,))
        rows = connection.execute(
            "SELECT aspect, COUNT(*) FROM aspects WHERE sentiment='negative' AND analysis_id=%s GROUP BY aspect ORDER BY COUNT(*) DESC, aspect",
            (analysis_id,)).fetchall()
    return str(analysis_id), [{"aspect": aspect, "negative_documents": count} for aspect, count in rows]
