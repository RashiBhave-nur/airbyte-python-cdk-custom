#
# Copyright (c) 2023 Airbyte, Inc., all rights reserved.
#

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Tuple

import dpath
import numpy as np
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.utils import stringify_dict
from langchain_core.documents.base import Document
from sentence_transformers import SentenceTransformer

from airbyte_cdk.destinations.vector_db_based.config import (
    ProcessingConfigModel,
    SeparatorSplitterConfigModel,
    TextSplitterConfigModel,
)
from airbyte_cdk.destinations.vector_db_based.utils import create_stream_identifier
from airbyte_cdk.models import (
    AirbyteRecordMessage,
    ConfiguredAirbyteCatalog,
    ConfiguredAirbyteStream,
    DestinationSyncMode,
)
from airbyte_cdk.utils.traced_exception import AirbyteTracedException, FailureType

METADATA_STREAM_FIELD = "_ab_stream"
METADATA_RECORD_ID_FIELD = "_ab_record_id"

CDC_DELETED_FIELD = "_ab_cdc_deleted_at"


@dataclass
class Chunk:
    page_content: Optional[str]
    metadata: Dict[str, Any]
    record: AirbyteRecordMessage
    embedding: Optional[List[float]] = None


headers_to_split_on = [
    "(?:^|\n)# ",
    "(?:^|\n)## ",
    "(?:^|\n)### ",
    "(?:^|\n)#### ",
    "(?:^|\n)##### ",
    "(?:^|\n)###### ",
]


# Cache the sentence transformer model to avoid reloading
@lru_cache(maxsize=1)
def _get_sentence_transformer(model_name: str = "all-MiniLM-L6-v2"):
    """Cache and return sentence transformer model"""
    return SentenceTransformer(model_name)


class DocumentProcessor:
    """
    DocumentProcessor is a helper class that generates documents from Airbyte records.

    It is used to generate documents from records before writing them to the destination:
    * The text fields are extracted from the record and concatenated to a single string.
    * The metadata fields are extracted from the record and added to the document metadata.
    * The document is split into chunks using semantic similarity instead of fixed-size chunking.

    The Writer class uses the DocumentProcessor class to internally generate documents from records - in most cases you don't need to use it directly,
    except if you want to implement a custom writer.

    The config parameters specified by the ProcessingConfigModel has to be made part of the connector spec to allow the user to configure the document processor.
    Calling DocumentProcessor.check_config(config) will validate the config and return an error message if the config is invalid.
    """

    streams: Mapping[str, ConfiguredAirbyteStream]

    @staticmethod
    def check_config(config: ProcessingConfigModel) -> Optional[str]:
        if config.text_splitter is not None and config.text_splitter.mode == "separator":
            for s in config.text_splitter.separators:
                try:
                    separator = json.loads(s)
                    if not isinstance(separator, str):
                        return f"Invalid separator: {s}. Separator needs to be a valid JSON string using double quotes."
                except json.decoder.JSONDecodeError:
                    return f"Invalid separator: {s}. Separator needs to be a valid JSON string using double quotes."
        return None

    def _get_text_splitter(
        self,
        chunk_size: int,
        chunk_overlap: int,
        splitter_config: Optional[TextSplitterConfigModel],
    ) -> RecursiveCharacterTextSplitter:
        """Create text splitter - kept for fallback purposes"""
        if splitter_config is None:
            splitter_config = SeparatorSplitterConfigModel(mode="separator")
        if splitter_config.mode == "separator":
            return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=[json.loads(s) for s in splitter_config.separators],
                keep_separator=splitter_config.keep_separator,
                disallowed_special=(),
            )
        if splitter_config.mode == "markdown":
            return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=headers_to_split_on[: splitter_config.split_level],
                is_separator_regex=True,
                keep_separator=True,
                disallowed_special=(),
            )
        if splitter_config.mode == "code":
            return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=RecursiveCharacterTextSplitter.get_separators_for_language(
                    Language(splitter_config.language)
                ),
                disallowed_special=(),
            )

    def __init__(self, config: ProcessingConfigModel, catalog: ConfiguredAirbyteCatalog):
        self.streams = {
            create_stream_identifier(stream.stream): stream for stream in catalog.streams
        }

        self.splitter = self._get_text_splitter(
            config.chunk_size, config.chunk_overlap, config.text_splitter
        )
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self.text_fields = config.text_fields
        self.metadata_fields = config.metadata_fields
        self.field_name_mappings = config.field_name_mappings
        self.logger = logging.getLogger("airbyte.document_processor")

    def process(self, record: AirbyteRecordMessage) -> Tuple[List[Chunk], Optional[str]]:
        """
        Generate documents from records.
        :param records: List of AirbyteRecordMessages
        :return: Tuple of (List of document chunks, record id to delete if a stream is in dedup mode to avoid stale documents in the vector store)
        """
        if CDC_DELETED_FIELD in record.data and record.data[CDC_DELETED_FIELD]:
            return [], self._extract_primary_key(record)
        doc = self._generate_document(record)
        if doc is None:
            text_fields = ", ".join(self.text_fields) if self.text_fields else "all fields"
            raise AirbyteTracedException(
                internal_message="No text fields found in record",
                message=f"Record {str(record.data)[:250]}... does not contain any of the configured text fields: {text_fields}. Please check your processing configuration, there has to be at least one text field set in each record.",
                failure_type=FailureType.config_error,
            )
        chunks = [
            Chunk(
                page_content=chunk_document.page_content,
                metadata=chunk_document.metadata,
                record=record,
            )
            for chunk_document in self._split_document(doc)
        ]
        id_to_delete = (
            doc.metadata[METADATA_RECORD_ID_FIELD]
            if METADATA_RECORD_ID_FIELD in doc.metadata
            else None
        )
        return chunks, id_to_delete

    def _generate_document(self, record: AirbyteRecordMessage) -> Optional[Document]:
        relevant_fields = self._extract_relevant_fields(record, self.text_fields)
        if len(relevant_fields) == 0:
            return None
        text = stringify_dict(relevant_fields)
        metadata = self._extract_metadata(record)
        return Document(page_content=text, metadata=metadata)

    def _extract_relevant_fields(
        self, record: AirbyteRecordMessage, fields: Optional[List[str]]
    ) -> Dict[str, Any]:
        relevant_fields = {}
        if fields and len(fields) > 0:
            for field in fields:
                values = dpath.values(record.data, field, separator=".")
                if values and len(values) > 0:
                    relevant_fields[field] = values if len(values) > 1 else values[0]
        else:
            relevant_fields = record.data
        return self._remap_field_names(relevant_fields)

    def _extract_metadata(self, record: AirbyteRecordMessage) -> Dict[str, Any]:
        metadata = self._extract_relevant_fields(record, self.metadata_fields)
        metadata[METADATA_STREAM_FIELD] = create_stream_identifier(record)
        primary_key = self._extract_primary_key(record)
        if primary_key:
            metadata[METADATA_RECORD_ID_FIELD] = primary_key
        return metadata

    def _extract_primary_key(self, record: AirbyteRecordMessage) -> Optional[str]:
        stream_identifier = create_stream_identifier(record)
        current_stream: ConfiguredAirbyteStream = self.streams[stream_identifier]
        # if the sync mode is deduping, use the primary key to upsert existing records instead of appending new ones
        if (
            not current_stream.primary_key
            or current_stream.destination_sync_mode != DestinationSyncMode.append_dedup
        ):
            return None

        primary_key = []
        for key in current_stream.primary_key:
            try:
                primary_key.append(str(dpath.get(record.data, key)))
            except KeyError:
                primary_key.append("__not_found__")
        stringified_primary_key = "_".join(primary_key)
        return f"{stream_identifier}_{stringified_primary_key}"

    def _split_document(self, doc: Document) -> List[Document]:
        """Split document using semantic similarity instead of fixed-size chunking"""
        return self._semantic_split_document(doc)

    def _semantic_split_document(
        self, 
        doc: Document, 
        similarity_threshold: float = 0.7,
        model_name: str = "all-MiniLM-L6-v2"
    ) -> List[Document]:
        """
        Split document using semantic similarity instead of fixed-size chunks.
        
        Args:
            doc: Document to split
            similarity_threshold: Cosine similarity threshold for grouping sentences
            model_name: Sentence transformer model name
            
        Returns:
            List of document chunks based on semantic similarity
        """
        text = doc.page_content
        if not text or len(text.strip()) == 0:
            return [doc]
        
        try:
            # Split text into sentences
            sentences = self._split_into_sentences(text)
            
            if len(sentences) <= 1:
                return [doc]
            
            self.logger.debug(f"Splitting {len(sentences)} sentences with semantic chunking")
            
            # Get sentence embeddings
            model = _get_sentence_transformer(model_name)
            embeddings = model.encode(sentences)
            
            # Group sentences semantically
            chunks = []
            current_chunk_sentences = [sentences[0]]
            current_token_count = self._count_tokens(sentences[0])
            
            for i in range(1, len(sentences)):
                sentence = sentences[i]
                sentence_tokens = self._count_tokens(sentence)
                
                # Check if adding this sentence would exceed max chunk size
                if current_token_count + sentence_tokens > self.chunk_size:
                    # Finalize current chunk
                    chunk_text = ' '.join(current_chunk_sentences)
                    chunks.append(Document(page_content=chunk_text, metadata=doc.metadata.copy()))
                    
                    current_chunk_sentences = [sentence]
                    current_token_count = sentence_tokens
                    continue
                
                # Calculate semantic similarity with the last sentence in current chunk
                last_idx = len(current_chunk_sentences) - 1
                original_last_idx = sentences.index(current_chunk_sentences[last_idx])
                
                if original_last_idx < len(embeddings) and i < len(embeddings):
                    prev_embedding = embeddings[original_last_idx]
                    curr_embedding = embeddings[i]
                    
                    # Cosine similarity
                    similarity = np.dot(prev_embedding, curr_embedding) / (
                        np.linalg.norm(prev_embedding) * np.linalg.norm(curr_embedding)
                    )
                    
                    if similarity >= similarity_threshold:
                        # Add to current chunk (semantically similar)
                        current_chunk_sentences.append(sentence)
                        current_token_count += sentence_tokens
                    else:
                        # Start new chunk (semantically different)
                        chunk_text = ' '.join(current_chunk_sentences)
                        chunks.append(Document(page_content=chunk_text, metadata=doc.metadata.copy()))
                        
                        current_chunk_sentences = [sentence]
                        current_token_count = sentence_tokens
                else:
                    # Fallback: add to current chunk
                    current_chunk_sentences.append(sentence)
                    current_token_count += sentence_tokens
            
            # Add final chunk
            if current_chunk_sentences:
                chunk_text = ' '.join(current_chunk_sentences)
                chunks.append(Document(page_content=chunk_text, metadata=doc.metadata.copy()))
            
            # Apply chunk overlap if specified
            if self.chunk_overlap > 0 and len(chunks) > 1:
                chunks = self._apply_chunk_overlap(chunks)
            
            self.logger.debug(f"Created {len(chunks)} semantic chunks")
            return chunks if chunks else [doc]
            
        except Exception as e:
            # Fallback to original method on any error
            self.logger.warning(f"Semantic chunking failed, falling back to original method: {e}")
            return self.splitter.split_documents([doc])

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Split on sentence endings, but keep the punctuation
        # This regex looks for sentence endings followed by whitespace or end of string
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Clean up sentences and filter empty ones
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the same method as the original splitter.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        try:
            # Use the original splitter's token counter if available
            if hasattr(self.splitter, '_tokenizer'):
                return len(self.splitter._tokenizer.encode(text))
            else:
                # Alternative: use tiktoken directly
                import tiktoken
                encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
                return len(encoding.encode(text))
        except Exception:
            # Fallback: rough approximation (words * 1.3 ≈ tokens)
            return int(len(text.split()) * 1.3)

    def _apply_chunk_overlap(self, chunks: List[Document]) -> List[Document]:
        """
        Apply chunk overlap to semantic chunks.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            List of chunks with overlap applied
        """
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.page_content
            
            # Add overlap from previous chunk
            if i > 0 and self.chunk_overlap > 0:
                prev_chunk = chunks[i - 1]
                prev_words = prev_chunk.page_content.split()
                
                # Take last N tokens as overlap (approximate)
                overlap_words = int(self.chunk_overlap / 1.3)  # Convert tokens to approximate words
                if len(prev_words) > overlap_words:
                    overlap_text = ' '.join(prev_words[-overlap_words:])
                    chunk_text = overlap_text + ' ' + chunk_text
            
            overlapped_chunks.append(Document(
                page_content=chunk_text,
                metadata=chunk.metadata
            ))
        
        return overlapped_chunks

    def _remap_field_names(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        if not self.field_name_mappings:
            return fields

        new_fields = fields.copy()
        for mapping in self.field_name_mappings:
            if mapping.from_field in new_fields:
                new_fields[mapping.to_field] = new_fields.pop(mapping.from_field)

        return new_fields
