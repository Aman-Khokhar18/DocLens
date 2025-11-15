from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
import os
import json

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", None)  

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

def upsert_entities(tx, entities, source_file, section_path, doc_id):
    for ent in entities:
        extra_dict = ent.get("extra") or {}
        extra_json = json.dumps(extra_dict, ensure_ascii=False)

        tx.run(
            """
            MERGE (e:DocEntity {id: $id})
            SET e.name        = $name,
                e.type        = $type,
                e.description = $description,
                e.extra       = $extra_json,
                e.source_files  = coalesce(e.source_files, []) +
                                  CASE WHEN $source_file IS NULL THEN [] ELSE [$source_file] END,
                e.section_paths = coalesce(e.section_paths, []) +
                                  CASE WHEN $section_path IS NULL THEN [] ELSE [$section_path] END,
                e.doc_ids       = coalesce(e.doc_ids, []) +
                                  CASE WHEN $doc_id IS NULL THEN [] ELSE [$doc_id] END
            """,
            id=ent.get("id"),
            name=ent.get("name"),
            type=ent.get("type"),
            description=ent.get("description", ""),
            extra_json=extra_json,
            source_file=source_file,
            section_path=section_path,
            doc_id=doc_id,
        )


def upsert_relations(
    tx,
    relations: List[Dict[str, Any]],
    source_file: str,
):
    for rel in relations:
        tx.run(
            """
            MERGE (s:DocEntity {id: $subject})
            MERGE (o:DocEntity {id: $object})
            MERGE (s)-[r:DOC_REL {predicate: $predicate}]->(o)
            SET r.description  = $description,
                r.source_files = coalesce(r.source_files, []) +
                                 CASE WHEN $source_file IS NULL THEN [] ELSE [$source_file] END
            """,
            subject=rel.get("subject"),
            predicate=rel.get("predicate"),
            object=rel.get("object"),
            description=rel.get("description", ""),
            source_file=source_file,
        )
