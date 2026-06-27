import os
import glob
import config
from copy import deepcopy
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from chunking.reference_section import is_reference_section
from chunking.overlap import apply_sequential_child_overlap
from chunking.semantic_breakpoints import SemanticBreakpointDetector
from chunking.sentence_units import parse_atomic_units, parse_reference_line_units
from chunking.size_enforcer import enforce_child_chunk_sizes, expand_oversized_units


class DocumentChuncker:
    def __init__(self):
        self.__parent_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=config.HEADERS_TO_SPLIT_ON,
            strip_headers=False,
        )
        self.__min_parent_size = config.MIN_PARENT_SIZE
        self.__max_parent_size = config.MAX_PARENT_SIZE
        self.__child_chunk_size = config.CHILD_CHUNK_SIZE
        self.__semantic = SemanticBreakpointDetector()

    @staticmethod
    def __merge_metadata(target, source, prepend=False):
        for key, value in source.items():
            if key not in target:
                target[key] = value
            elif prepend:
                target[key] = f"{value} -> {target[key]}"
            else:
                target[key] = f"{target[key]} -> {value}"

    def create_chunks(self, path_dir=config.MARKDOWN_DIR):
        all_parent_chunks, all_child_chunks = [], []

        for doc_path_str in sorted(glob.glob(os.path.join(path_dir, "*.md"))):
            doc_path = Path(doc_path_str)
            parent_chunks, child_chunks = self.create_chunks_single(doc_path)
            all_parent_chunks.extend(parent_chunks)
            all_child_chunks.extend(child_chunks)

        return all_parent_chunks, all_child_chunks

    def create_chunks_single(self, md_path):
        doc_path = Path(md_path)

        with open(doc_path, "r", encoding="utf-8") as f:
            header_chunks = self.__parent_splitter.split_text(f.read())

        merged_parents = self.__merge_header_chunks(header_chunks)
        final_parents = self.__finalize_parent_chunks(merged_parents)

        all_parent_chunks, all_child_chunks = [], []
        self.__create_child_chunks(all_parent_chunks, all_child_chunks, final_parents, doc_path)
        return all_parent_chunks, all_child_chunks

    def __merge_header_chunks(self, chunks):
        """Merge title-based blocks under 2000 chars without crossing the 4000 hard cap."""
        if not chunks:
            return []

        merged = []
        index = 0
        total = len(chunks)

        while index < total:
            current = chunks[index]
            current_size = len(current.page_content)

            if current_size >= self.__min_parent_size:
                merged.append(current)
                index += 1
                continue

            next_index = index + 1
            while next_index < total:
                candidate = chunks[next_index]
                combined_size = len(current.page_content) + 2 + len(candidate.page_content)
                if combined_size > self.__max_parent_size:
                    break

                current.page_content += "\n\n" + candidate.page_content
                self.__merge_metadata(current.metadata, candidate.metadata)
                next_index += 1

                if len(current.page_content) >= self.__min_parent_size:
                    break

            merged.append(current)
            index = next_index

        return merged

    def __finalize_parent_chunks(self, chunks):
        parents = []

        for chunk in chunks:
            content = chunk.page_content
            if len(content) <= self.__max_parent_size:
                parents.append(chunk)
                continue

            if is_reference_section(chunk.metadata):
                units = parse_reference_line_units(content)
                parent_texts = enforce_child_chunk_sizes(
                    units,
                    self.__max_parent_size,
                    line_based_only=True,
                )
                for parent_text in parent_texts:
                    parent_doc = Document(
                        page_content=parent_text,
                        metadata=deepcopy(chunk.metadata),
                    )
                    parents.append(parent_doc)
                continue

            units = parse_atomic_units(content)
            units = expand_oversized_units(units, self.__max_parent_size)
            breakpoints = self.__semantic.detect_breakpoints(
                units,
                buffer_size=config.PARENT_CONTEXT_BUFFER,
                percentile=config.PARENT_SEMANTIC_PERCENTILE,
            )
            parent_texts = self.__semantic.build_parent_texts(
                units,
                breakpoints,
                self.__max_parent_size,
            )

            for parent_text in parent_texts:
                parent_doc = Document(
                    page_content=parent_text,
                    metadata=deepcopy(chunk.metadata),
                )
                parents.append(parent_doc)

        return parents

    def __create_child_chunks(self, all_parent_pairs, all_child_chunks, parent_chunks, doc_path):
        for index, parent_chunk in enumerate(parent_chunks):
            parent_id = f"{doc_path.stem}_parent_{index}"
            parent_chunk.metadata.update(
                {"source": f"{doc_path.stem}.pdf", "parent_id": parent_id}
            )
            all_parent_pairs.append((parent_id, parent_chunk))

            if is_reference_section(parent_chunk.metadata):
                units = parse_reference_line_units(parent_chunk.page_content)
                child_texts = enforce_child_chunk_sizes(
                    units,
                    self.__child_chunk_size,
                    line_based_only=True,
                )
            else:
                units = parse_atomic_units(parent_chunk.page_content)
                breakpoints = self.__semantic.detect_breakpoints(
                    units,
                    buffer_size=config.CHILD_CONTEXT_BUFFER,
                    percentile=config.CHILD_SEMANTIC_PERCENTILE,
                )
                segments = self.__semantic.split_segments(units, breakpoints)

                raw_child_chunks: list[str] = []
                for segment in segments:
                    raw_child_chunks.extend(
                        enforce_child_chunk_sizes(segment, self.__child_chunk_size)
                    )

                child_texts = apply_sequential_child_overlap(
                    raw_child_chunks,
                    self.__child_chunk_size,
                )

            for chunk_text in child_texts:
                all_child_chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata=deepcopy(parent_chunk.metadata),
                    )
                )
