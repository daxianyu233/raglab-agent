from __future__ import annotations
from dataclasses import dataclass,field
from typing import Any,Protocol

@dataclass
class E2ETurnObservation:
    answer:str=""
    completed_normally:bool=False
    tool_calls:list[dict[str,Any]]=field(default_factory=list)
    capability_groups_used:list[str]=field(default_factory=list)
    total_latency_ms:float|None=None
    input_tokens:int|None=None
    output_tokens:int|None=None
    total_tokens:int|None=None
    pending_human_approval:bool=False
    write_side_effect_count:int=0
    state:dict[str,Any]=field(default_factory=dict)

class FullAgentE2EAdapter(Protocol):
    def reset_case(self,case_id:str)->None: ...
    def apply_setup(self,setup:list[dict[str,Any]])->None: ...
    def send(self,user_input:str)->E2ETurnObservation: ...
    def inspect_state(self)->dict[str,Any]: ...
