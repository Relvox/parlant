# Copyright 2024 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from contextlib import AsyncExitStack
import importlib
import json
import os
import shutil
from typing import cast, Any
import chromadb
from chromadb.api.types import IncludeEnum
from lagom import Container
from pymongo import AsyncMongoClient
from typing_extensions import NoReturn
from pathlib import Path
import sys
import rich
from rich.prompt import Confirm, Prompt

from parlant.adapters.db.mongo_db import MongoDocumentDatabase
from parlant.adapters.vector_db.chroma import ChromaDatabase
from parlant.core.common import generate_id, md5_checksum
from parlant.core.contextual_correlator import ContextualCorrelator
from parlant.core.loggers import LogLevel, StdoutLogger
from parlant.core.nlp.embedding import EmbedderFactory
from parlant.core.persistence.common import ObjectId
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentDatabase,
    identity_loader,
)
from parlant.core.persistence.document_database_helper import _MetadataDocument

DEFAULT_HOME_DIR = "runtime-data" if Path("runtime-data").exists() else "parlant-data"
PARLANT_HOME_DIR = Path(os.environ.get("PARLANT_HOME", DEFAULT_HOME_DIR))
PARLANT_HOME_DIR.mkdir(parents=True, exist_ok=True)

EXIT_STACK = AsyncExitStack()

sys.path.append(PARLANT_HOME_DIR.as_posix())
sys.path.append(".")

LOGGER = StdoutLogger(
    correlator=ContextualCorrelator(),
    log_level=LogLevel.INFO,
)


def backup_data() -> None:
    if Confirm.ask("Do you want to backup your data before migration?"):
        default_backup_dir = PARLANT_HOME_DIR.parent / "parlant-data.orig"
        try:
            backup_dir = Prompt.ask("Enter backup directory path", default=str(default_backup_dir))
            shutil.copytree(PARLANT_HOME_DIR, backup_dir, dirs_exist_ok=True)
            rich.print(f"[green]Data backed up to {backup_dir}")
        except Exception as e:
            rich.print(f"[red]Failed to backup data: {e}")
            die(f"Error backing up data: {e}")


async def migrate() -> None:
    await is_migration_required()

    rich.print("[green]Starting migration process...")

    backup_data()
    mongo_client = AsyncMongoClient[Any](
        "mongodb+srv://guest:guess@cluster0.6q2ty.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    )

    agents_db = await EXIT_STACK.enter_async_context(
        MongoDocumentDatabase(mongo_client, "agents", LOGGER)
    )
    await migrate_document_database(agents_db, "agents")

    context_variables_db = await EXIT_STACK.enter_async_context(
        MongoDocumentDatabase(mongo_client, "context_variables", LOGGER)
    )
    await migrate_document_database(context_variables_db, "variables")

    tags_db = await EXIT_STACK.enter_async_context(
        MongoDocumentDatabase(mongo_client, "tags", LOGGER)
    )
    await migrate_document_database(tags_db, "tags")

    customers_db = await EXIT_STACK.enter_async_context(
        MongoDocumentDatabase(mongo_client, "customers", LOGGER)
    )
    await migrate_document_database(customers_db, "customers")

    sessions_db = await EXIT_STACK.enter_async_context(
        MongoDocumentDatabase(mongo_client, "sessions", LOGGER)
    )
    await migrate_document_database(sessions_db, "sessions")

    guideline_tool_associations_db = await EXIT_STACK.enter_async_context(
        MongoDocumentDatabase(mongo_client, "guideline_tool_associations", LOGGER)
    )
    await migrate_document_database(guideline_tool_associations_db, "associations")

    guidelines_db = await EXIT_STACK.enter_async_context(
        MongoDocumentDatabase(mongo_client, "guidelines", LOGGER)
    )
    await migrate_document_database(guidelines_db, "guidelines")

    guideline_connections_db = await EXIT_STACK.enter_async_context(
        MongoDocumentDatabase(mongo_client, "guideline_connections", LOGGER)
    )
    await migrate_document_database(guideline_connections_db, "guideline_connections")

    evaluations_db = await EXIT_STACK.enter_async_context(
        MongoDocumentDatabase(mongo_client, "evaluations", LOGGER)
    )
    await migrate_document_database(evaluations_db, "evaluations")

    services_db = await EXIT_STACK.enter_async_context(
        MongoDocumentDatabase(mongo_client, "services", LOGGER)
    )
    await migrate_document_database(services_db, "tool_services")

    await migrate_glossary()

    await upgrade_agents()
    rich.print("[green]Migration completed successfully")


async def migrate_document_database(db: DocumentDatabase, collection_name: str) -> None:
    rich.print(f"[green]Migrating {collection_name} database...")
    try:
        collection = await db.get_collection(
            collection_name,
            BaseDocument,
            identity_loader,
        )

    except ValueError:
        rich.print(f"[yellow]Collection {collection_name} not found, skipping...")
        return

    try:
        metadata_collection = await db.get_collection(
            "metadata",
            BaseDocument,
            identity_loader,
        )
        await db.delete_collection("metadata")

    except ValueError:
        pass

    metadata_collection = await db.get_or_create_collection(
        "metadata",
        _MetadataDocument,
        identity_loader,
    )

    if document := await collection.find_one({}):
        await metadata_collection.insert_one(
            {
                "id": ObjectId(generate_id()),
                "version": document["version"],
            }
        )
        rich.print(f"[green]Successfully migrated {collection_name} database")
    else:
        rich.print(f"[yellow]No documents found in {collection_name} collection.")


async def migrate_glossary() -> None:
    rich.print("[green]Starting glossary migration...")
    try:
        embedder_factory = EmbedderFactory(Container())

        db = await EXIT_STACK.enter_async_context(
            ChromaDatabase(LOGGER, PARLANT_HOME_DIR, embedder_factory)
        )

        try:
            old_collection = db.chroma_client.get_collection("glossary")
        except chromadb.errors.InvalidCollectionException:
            rich.print("[yellow]Glossary collection not found, skipping...")
            return

        if docs := old_collection.peek(limit=1)["metadatas"]:
            document = docs[0]

            version = document["version"]

            embedder_module = importlib.import_module(
                f"{old_collection.metadata['embedder_module_path']}_service"
            )
            embedder_type = getattr(
                embedder_module,
                old_collection.metadata["embedder_type_path"],
            )

            all_items = old_collection.get(
                include=[IncludeEnum.documents, IncludeEnum.embeddings, IncludeEnum.metadatas]
            )
            rich.print(f"[green]Found {len(all_items['ids'])} items to migrate")

            chroma_unembedded_collection = next(
                (
                    collection
                    for collection in db.chroma_client.list_collections()
                    if collection.name == "glossary_unembedded"
                ),
                None,
            ) or db.chroma_client.create_collection(name="glossary_unembedded")

            chroma_new_collection = next(
                (
                    collection
                    for collection in db.chroma_client.list_collections()
                    if collection.name == db.format_collection_name("glossary", embedder_type)
                ),
                None,
            ) or db.chroma_client.create_collection(
                name=db.format_collection_name("glossary", embedder_type)
            )

            if all_items["metadatas"] is None:
                rich.print("[yellow]No metadatas found in glossary collection, skipping...")
                return

            for i in range(len(all_items["metadatas"])):
                assert all_items["documents"] is not None
                assert all_items["embeddings"] is not None

                new_doc = {
                    **all_items["metadatas"][i],
                    "checksum": md5_checksum(all_items["documents"][i]),
                }

                chroma_unembedded_collection.add(
                    ids=[all_items["ids"][i]],
                    documents=[str(new_doc["content"])],
                    metadatas=[cast(chromadb.types.Metadata, new_doc)],
                    embeddings=[0],
                )

                chroma_new_collection.add(
                    ids=[all_items["ids"][i]],
                    documents=[str(new_doc["content"])],
                    metadatas=[cast(chromadb.types.Metadata, new_doc)],
                    embeddings=all_items["embeddings"][i],
                )

            # Version starts at 1
            chroma_unembedded_collection.modify(
                metadata={"version": 1 + len(all_items["metadatas"])}
            )
            chroma_new_collection.modify(metadata={"version": 1 + len(all_items["metadatas"])})

            await db.upsert_metadata(
                "version",
                version,
            )
            rich.print("[green]Successfully migrated glossary data")

        db.chroma_client.delete_collection(old_collection.name)
        rich.print("[green]Cleaned up old glossary collection")

    except Exception as e:
        rich.print(f"[red]Failed to migrate glossary: {e}")
        die(f"Error migrating glossary: {e}")


async def is_migration_required() -> None:
    rich.print("[green]Checking if migration is required...")
    with open(PARLANT_HOME_DIR / "agents.json", "r") as f:
        raw_data = json.load(f)
        agents = raw_data.get("agents")

    if agents is None:
        rich.print("[yellow]No agents found, skipping...")
        return

    for agent in agents:
        if agent["version"] == "0.1.0":
            return

    die("Migration is not required")


async def upgrade_agents() -> None:
    rich.print("[green]Starting agents migration...")

    with open(PARLANT_HOME_DIR / "agents.json", "r") as f:
        raw_data = json.load(f)
        agents = raw_data.get("agents")

    if agents is None:
        rich.print("[yellow]No agents found, skipping...")
        return

    for agent in agents:
        agent["version"] = "0.2.0"

    raw_data["agents"] = agents
    raw_data["metadata"][0].update({"version": "0.2.0"})

    with open(PARLANT_HOME_DIR / "agents.json", "w") as f:
        json.dump(raw_data, f)


def die(message: str) -> NoReturn:
    rich.print(f"[red]{message}")
    print(message, file=sys.stderr)
    sys.exit(1)


def main() -> None:
    try:
        asyncio.run(migrate())
    except Exception as e:
        die(str(e))


if __name__ == "__main__":
    main()
