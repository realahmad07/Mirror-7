from typing import Any, Dict, List, Optional, Callable
import time

class WorkspaceItem:
    def __init__(self, item_id: str, item_type: str, content: Dict[str, Any], priority: float, confidence: float, source: str, created_at: int, expires_at: Optional[int], dependencies: List[str]):
        self.id = item_id
        self.item_type = item_type
        self.content = content
        self.priority = priority
        self.confidence = confidence
        self.source = source
        self.created_at = created_at
        self.expires_at = expires_at
        self.dependencies = dependencies
        self.status = 'active'

class WorkspaceProcessor:
    def __init__(self, name: str, handles: List[str], process_func: Callable):
        self.name = name
        self.handles = handles
        self.process_func = process_func
        
    def process(self, item: WorkspaceItem) -> List[WorkspaceItem]:
        return self.process_func(item)

class CognitiveWorkspace:
    def __init__(self, max_items: int = 100):
        self.items = {}
        self.processors = []
        self.max_items = max_items
        self.current_time = 0
        self.log = []
        
    def post(self, item: WorkspaceItem):
        if len(self.items) >= self.max_items:
            active_items = list(self.items.values())
            if active_items:
                lowest = min(active_items, key=lambda x: x.priority)
                del self.items[lowest.id]
                self.log.append(f"Evicted {lowest.id}")
        self.items[item.id] = item
        self.log.append(f"Posted {item.id}")
        
    def register_processor(self, processor: WorkspaceProcessor):
        self.processors.append(processor)
        
    def focus(self) -> Optional[WorkspaceItem]:
        active_items = [i for i in self.items.values() if i.status == 'active']
        if not active_items: return None
        return max(active_items, key=lambda x: x.priority)
        
    def tick(self):
        self.current_time += 1
        to_delete = [item.id for item in self.items.values() if item.expires_at and item.expires_at <= self.current_time]
        for d in to_delete:
            del self.items[d]
            self.log.append(f"Expired {d}")
            
        focus_item = self.focus()
        if not focus_item: return
        
        for p in self.processors:
            if focus_item.item_type in p.handles:
                try:
                    new_items = p.process(focus_item)
                    for ni in new_items:
                        self.post(ni)
                except Exception as e:
                    self.log.append(f"Error in processor {p.name}: {e}")
        
        focus_item.status = 'resolved'
