#
# Copyright (c) 2023 Airbyte, Inc., all rights reserved.
#

import itertools
import traceback
from typing import Any, Iterable, Iterator, Tuple, Union

from airbyte_cdk.models import AirbyteRecordMessage, AirbyteStream


def format_exception(exception: Exception) -> str:
    return (
        str(exception)
        + "\n"
        + "".join(traceback.TracebackException.from_exception(exception).format())
    )


def create_chunks(
    iterable: Iterable[Any], 
    batch_size: int, 
    similarity_threshold: float = 0.7,
    model_name: str = "all-MiniLM-L6-v2"
) -> Iterator[Tuple[Any, ...]]:
    """
    A helper function to break an iterable into chunks based on semantic similarity.
    
    Args:
        iterable: The iterable to chunk
        batch_size: The maximum size of each chunk
        similarity_threshold: Similarity threshold (cosine similarity) for merging items
        model_name: Name of the sentence-transformer model to use for embeddings
        
    Returns:
        Iterator of tuples containing the chunked items
    """
    # Convert iterable to list for processing
    items = list(iterable)
    
    # If items is empty or batch_size is 1, use simple chunking
    if not items or batch_size <= 1:
        it = iter(items)
        chunk = tuple(itertools.islice(it, batch_size))
        while chunk:
            yield chunk
            chunk = tuple(itertools.islice(it, batch_size))
        return
    
    try:
        # Initialize embedding model
        model = SentenceTransformer(model_name)
        
        # Create semantic chunks
        current_items = []
        current_chunk_size = 0
        
        # Process items one by one
        for i, item in enumerate(items):
            # Get string representation for embedding
            item_str = str(item)
            
            # If this is the first item or current chunk is empty
            if not current_items:
                current_items.append(item)
                current_chunk_size = 1
                continue
            
            # If we've reached batch size, yield current chunk and start a new one
            if current_chunk_size >= batch_size:
                yield tuple(current_items)
                current_items = [item]
                current_chunk_size = 1
                continue
            
            # Get embeddings for comparison
            last_item_str = str(current_items[-1])
            
            # Calculate embeddings and similarity
            embeddings = model.encode([item_str, last_item_str])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            # If items are semantically similar, add to current chunk
            # Otherwise start a new chunk
            if similarity >= similarity_threshold:
                current_items.append(item)
                current_chunk_size += 1
            else:
                yield tuple(current_items)
                current_items = [item]
                current_chunk_size = 1
        
        # Yield the last chunk if not empty
        if current_items:
            yield tuple(current_items)
            
    except Exception as e:
        # Fall back to simple chunking if semantic chunking fails
        print(f"Semantic chunking failed, falling back to simple chunking: {format_exception(e)}")
        it = iter(items)
        chunk = tuple(itertools.islice(it, batch_size))
        while chunk:
            yield chunk
            chunk = tuple(itertools.islice(it, batch_size))

def create_stream_identifier(stream: Union[AirbyteStream, AirbyteRecordMessage]) -> str:
    if isinstance(stream, AirbyteStream):
        return str(stream.name if stream.namespace is None else f"{stream.namespace}_{stream.name}")
    else:
        return str(
            stream.stream if stream.namespace is None else f"{stream.namespace}_{stream.stream}"
        )
