from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, asdict
import mmap, os
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class CacheKey:
    layer:int; expert:int; kind:str

@dataclass
class CacheStats:
    hits:int=0; misses:int=0; evictions:int=0; mapped_bytes:int=0; peak_mapped_bytes:int=0
    def to_dict(self)->dict[str,Any]: return asdict(self)

@dataclass
class _Window:
    mapping:mmap.mmap; mapped_offset:int; mapped_length:int; logical_offset:int; logical_length:int
    def close(self):
        try:self.mapping.close()
        except BufferError:pass

class MmapExpertCache:
    def __init__(self,path:str|Path,budget_bytes:int):
        if budget_bytes<=0: raise ValueError('budget_bytes must be positive')
        self.path=str(Path(path).expanduser().resolve()); self.fd=os.open(self.path,os.O_RDONLY)
        self.budget_bytes=int(budget_bytes); self.page_size=mmap.PAGESIZE
        self.entries:OrderedDict[CacheKey,_Window]=OrderedDict(); self.stats=CacheStats()
    def close(self):
        for w in self.entries.values(): w.close()
        self.entries.clear(); os.close(self.fd)
    def __enter__(self): return self
    def __exit__(self,*_): self.close()
    def _map(self,offset,length,eager):
        page=self.page_size; mo=offset//page*page; delta=offset-mo; ml=delta+length
        mm=mmap.mmap(self.fd,ml,flags=mmap.MAP_PRIVATE,prot=mmap.PROT_READ,offset=mo)
        if eager and hasattr(mm,'madvise') and hasattr(mmap,'MADV_WILLNEED'):
            try:mm.madvise(mmap.MADV_WILLNEED)
            except OSError:pass
        return _Window(mm,mo,ml,offset,length)
    def _evict_until(self,needed):
        if needed>self.budget_bytes: raise ValueError('single expert window exceeds cache budget')
        while self.stats.mapped_bytes+needed>self.budget_bytes and self.entries:
            _,w=self.entries.popitem(last=False); self.stats.mapped_bytes-=w.mapped_length; self.stats.evictions+=1
            if hasattr(w.mapping,'madvise') and hasattr(mmap,'MADV_DONTNEED'):
                try:w.mapping.madvise(mmap.MADV_DONTNEED)
                except OSError:pass
            w.close()
    def acquire(self,key:CacheKey,offset:int,length:int,*,eager=True)->memoryview:
        w=self.entries.pop(key,None)
        if w is not None:
            self.entries[key]=w; self.stats.hits+=1; s=w.logical_offset-w.mapped_offset
            return memoryview(w.mapping)[s:s+w.logical_length]
        self.stats.misses+=1; self._evict_until(length+self.page_size); w=self._map(offset,length,eager); self._evict_until(w.mapped_length)
        self.entries[key]=w; self.stats.mapped_bytes+=w.mapped_length; self.stats.peak_mapped_bytes=max(self.stats.peak_mapped_bytes,self.stats.mapped_bytes)
        s=w.logical_offset-w.mapped_offset; return memoryview(w.mapping)[s:s+w.logical_length]
    def prefetch(self,key,offset,length):
        v=self.acquire(key,offset,length,eager=True); v.release()
