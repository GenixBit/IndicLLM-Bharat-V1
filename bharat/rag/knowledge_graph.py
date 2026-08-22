"""Sovereign Knowledge Graph Engine for Multi-Hop Relational Reasoning in IndicLLM-Bharat.

Supports entity extraction, multi-hop relationship traversal (e.g. Company -> develops -> Product),
and contextual graph subgraph retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphEntity:
    entity_id: str
    name: str
    entity_type: str  # Person, Organization, Technology, Location, Event, Paper
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphRelationship:
    source_id: str
    relation_type: str  # develops, founded_by, uses, located_in, launched_by
    target_id: str
    weight: float = 1.0


class KnowledgeGraph:
    """In-memory sovereign knowledge graph with multi-hop neighbor search."""

    def __init__(self) -> None:
        self.entities: dict[str, GraphEntity] = {}
        self.name_to_id: dict[str, str] = {}
        self.adjacency: dict[str, list[GraphRelationship]] = {}
        self._populate_sovereign_entities()

    def add_entity(self, entity: GraphEntity) -> None:
        self.entities[entity.entity_id] = entity
        self.name_to_id[entity.name.lower()] = entity.entity_id
        if entity.entity_id not in self.adjacency:
            self.adjacency[entity.entity_id] = []

    def add_relationship(self, rel: GraphRelationship) -> None:
        if rel.source_id in self.entities and rel.target_id in self.entities:
            self.adjacency[rel.source_id].append(rel)

    def _populate_sovereign_entities(self) -> None:
        """Seed high-value Indian and global scientific knowledge entities."""
        e1 = GraphEntity(
            "isro", "ISRO", "Organization", {"founded": 1969, "headquarters": "Bengaluru"}
        )
        e2 = GraphEntity(
            "chandrayaan3",
            "Chandrayaan-3",
            "Technology",
            {"launch_year": 2023, "objective": "Lunar South Pole"},
        )
        e3 = GraphEntity(
            "vikram_sarabhai",
            "Dr. Vikram Sarabhai",
            "Person",
            {"role": "Father of Indian Space Program"},
        )
        e4 = GraphEntity(
            "aryabhata", "Aryabhata", "Person", {"era": "Classical Indian Astronomy & Mathematics"}
        )
        e5 = GraphEntity("nqm", "National Quantum Mission", "Project", {"target": "50-1000 qubits"})
        e6 = GraphEntity(
            "bharat_llm", "IndicLLM-Bharat", "Technology", {"languages": 22, "context": "32k"}
        )

        for e in [e1, e2, e3, e4, e5, e6]:
            self.add_entity(e)

        self.add_relationship(GraphRelationship("isro", "founded_by", "vikram_sarabhai"))
        self.add_relationship(GraphRelationship("isro", "launched", "chandrayaan3"))
        self.add_relationship(GraphRelationship("chandrayaan3", "lands_on", "lunar_south_pole"))
        self.add_relationship(
            GraphRelationship("bharat_llm", "developed_for", "22_scheduled_languages")
        )

    def search_subgraph(self, query: str, max_hops: int = 2) -> list[str]:
        """Find matching entities and perform multi-hop neighborhood extraction."""
        q_lower = query.lower()
        matched_entity_ids: list[str] = []

        for name, ent_id in self.name_to_id.items():
            if name in q_lower or any(word in name for word in q_lower.split()):
                matched_entity_ids.append(ent_id)

        facts: list[str] = []
        visited: set[str] = set()

        def dfs(curr_id: str, depth: int) -> None:
            if depth > max_hops or curr_id in visited:
                return
            visited.add(curr_id)
            curr_ent = self.entities.get(curr_id)
            if not curr_ent:
                return

            for rel in self.adjacency.get(curr_id, []):
                target_ent = self.entities.get(rel.target_id)
                target_name = target_ent.name if target_ent else rel.target_id
                facts.append(f"{curr_ent.name} --({rel.relation_type})--> {target_name}")
                dfs(rel.target_id, depth + 1)

        for m_id in matched_entity_ids:
            dfs(m_id, 1)

        return facts
