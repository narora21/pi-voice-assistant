import re


class ChunkBatcher:
    """Batches text chunks by sentences or word count with character limits."""
    
    # Sentence endings: . ! ? followed by space or end
    SENTENCE_END = re.compile(r'[.!?](?:\s|$)')
    # Natural pause points: commas, semicolons, colons, dashes
    PAUSE_POINTS = re.compile(r'[,;:\-—](?:\s|$)')
    
    def __init__(
        self,
        min_chars: int = 50,
        max_chars: int = 200,
        min_words: int = 5,
    ) -> None:
        self._min_chars = min_chars
        self._max_chars = max_chars
        self._min_words = min_words
        self._buffer = ""
    
    def add(self, text: str) -> list[str]:
        """Add text and return any complete batches."""
        self._buffer += text
        return self._flush_ready()
    
    def flush(self) -> str | None:
        """Flush remaining buffer."""
        if self._buffer.strip():
            result = self._buffer.strip()
            self._buffer = ""
            return result
        return None
    
    def _flush_ready(self) -> list[str]:
        """Extract ready batches from buffer."""
        batches: list[str] = []
        
        while True:
            batch = self._try_extract_batch()
            if batch is None:
                break
            batches.append(batch)
        
        return batches
    
    def _try_extract_batch(self) -> str | None:
        """Try to extract a single batch from buffer."""
        # Not enough content yet
        if len(self._buffer) < self._min_chars:
            return None
        
        # Force split if buffer exceeds max
        if len(self._buffer) >= self._max_chars:
            return self._split_at_best_point(self._max_chars)
        
        # Look for sentence boundary
        match = self.SENTENCE_END.search(self._buffer)
        if match and match.end() >= self._min_chars:
            return self._extract_at(match.end())
        
        # Look for pause point if buffer is getting long
        if len(self._buffer) >= self._min_chars * 1.5:
            match = self.PAUSE_POINTS.search(self._buffer, pos=self._min_chars)
            if match:
                return self._extract_at(match.end())
        
        return None
    
    def _split_at_best_point(self, max_pos: int) -> str:
        """Split at best point within max_pos characters."""
        search_region = self._buffer[:max_pos]
        
        # Prefer sentence end
        match = None
        for m in self.SENTENCE_END.finditer(search_region):
            if m.end() >= self._min_chars:
                match = m
        if match:
            return self._extract_at(match.end())
        
        # Then pause point
        for m in self.PAUSE_POINTS.finditer(search_region):
            if m.end() >= self._min_chars:
                match = m
        if match:
            return self._extract_at(match.end())
        
        # Then word boundary
        last_space = search_region.rfind(' ')
        if last_space > self._min_chars:
            return self._extract_at(last_space + 1)
        
        # Last resort: hard cut
        return self._extract_at(max_pos)
    
    def _extract_at(self, pos: int) -> str:
        """Extract text up to pos from buffer."""
        result = self._buffer[:pos].strip()
        self._buffer = self._buffer[pos:].lstrip()
        return result
    
    def _word_count(self, text: str) -> int:
        return len(text.split())
