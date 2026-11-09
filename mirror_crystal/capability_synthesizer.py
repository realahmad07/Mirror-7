"""Bounded example-guided discovery of new UOL programs and capabilities."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import hashlib, itertools, json, math
from typing import Any, Iterable, Mapping

from .operational_language import (UOL, UOLError, Lit, Var, Call, If, ArrayExpr, RecordExpr, Field, Index, Let, MapExpr, FoldExpr, Expression, Program, OperationalRule, RuleStatus, VariableSpec, RuleCatalog, expr_dict, _infer, _OPS, _type_name)
from .blueprint_contracts import EvidenceLevel, VerificationEvidence
from .capability_model import CapabilitySpec, PortContract
from .universal_representation import RepresentationProvenance

class SynthesisError(ValueError): pass
class SynthesisStatus(str,Enum): SYNTHESIZED="synthesized"; NO_SOLUTION="no_solution"; BUDGET_EXHAUSTED="budget_exhausted"

@dataclass(frozen=True)
class Example:
    inputs:Mapping[str,Any]; expected:Any

@dataclass(frozen=True)
class SynthesisGoal:
    name:str; domain:str; inputs:tuple[VariableSpec,...]; output_type:str
    training:tuple[Example,...]; validation:tuple[Example,...]
    constants:tuple[Any,...]=(0,1,2,-1); max_depth:int=2; candidate_budget:int=20000
    include_assumed_rules:bool=True
    def __post_init__(self):
        object.__setattr__(self,"inputs",tuple(self.inputs));object.__setattr__(self,"training",tuple(self.training));object.__setattr__(self,"validation",tuple(self.validation));object.__setattr__(self,"constants",tuple(self.constants));object.__setattr__(self,"domain",self.domain.casefold());object.__setattr__(self,"output_type",_type_name(self.output_type))
        if len(self.training)<2 or not self.validation:raise SynthesisError("need at least two training and one held-out example")
        if self.max_depth<0 or self.max_depth>4 or not 1<=self.candidate_budget<=250000:raise SynthesisError("synthesis search limits out of range")
        keys=[json.dumps(dict(x.inputs),sort_keys=True) for x in self.training+self.validation]
        if len(keys)!=len(set(keys)):raise SynthesisError("training and validation inputs must be disjoint")
        required={x.name for x in self.inputs}
        for example in self.training+self.validation:
            if set(example.inputs)!=required:raise SynthesisError("example inputs must match synthesis schema")

@dataclass(frozen=True)
class SynthesisReport:
    status:SynthesisStatus; program:Program|None; tested:int; train_passed:int; validation_passed:int; rule_ids:tuple[str,...]=(); blockers:tuple[str,...]=()
    @property
    def succeeded(self):return self.status is SynthesisStatus.SYNTHESIZED and self.program is not None

class CapabilitySynthesizer:
    VERSION="capability-synthesizer-v1"
    def __init__(self,language:UOL|None=None):self.language=language or UOL()
    def _terminals(self,goal):
        out=[]
        for port in goal.inputs:out.append((Var(port.name),port.value_type))
        for value in goal.constants:
            kind="integer" if isinstance(value,int) and not isinstance(value,bool) else "number" if isinstance(value,float) else "boolean" if isinstance(value,bool) else "text" if isinstance(value,str) else "null"
            out.append((Lit(value),kind))
        return out
    def _eligible_rules(self,goal):
        return tuple(r for r in self.language.rules.values() if (goal.domain in r.domains or "general" in r.domains or "*" in r.domains) and r.status not in {RuleStatus.UNKNOWN,RuleStatus.DISPUTED} and (goal.include_assumed_rules or r.status is not RuleStatus.ASSUMED))
    def _generate(self,goal,budget):
        terminals=self._terminals(goal);seen={json.dumps(expr_dict(e),sort_keys=True): (e,t,0) for e,t in terminals};front=list(seen.values())
        def collected(additions):
            merged=dict(seen)
            for item in additions:
                merged.setdefault(json.dumps(expr_dict(item[0]),sort_keys=True),item)
            return list(merged.values())
        for depth in range(1,goal.max_depth+1):
            additions=[]
            # Unary primitives and available rules.
            for e,t,_ in front:
                for op in ("negate","absolute","not","length"):
                    if budget[0]>=goal.candidate_budget:return collected(additions),True
                    budget[0]+=1
                    try:out=_infer(Call(op,(e,)),{x.name:x.value_type for x in goal.inputs},self.language.rules,set())
                    except UOLError:continue
                    key=json.dumps(expr_dict(Call(op,(e,))),sort_keys=True)
                    if key not in seen:additions.append((Call(op,(e,)),out,depth))
            pool=list(seen.values())
            for left in pool:
                for right in pool:
                    if budget[0]>=goal.candidate_budget:return collected(additions),True
                    for op in ("add","subtract","multiply","divide","modulo","power","minimum","maximum","equal","less_than","less_equal","greater_than","greater_equal","and","or","concat"):
                        budget[0]+=1
                        expr=Call(op,(left[0],right[0]))
                        try:out=_infer(expr,{x.name:x.value_type for x in goal.inputs},self.language.rules,set())
                        except UOLError:continue
                        if 1+max(left[2],right[2])>depth:continue
                        key=json.dumps(expr_dict(expr),sort_keys=True)
                        if key not in seen:additions.append((expr,out,depth))
                    if budget[0]>=goal.candidate_budget:return collected(additions),True
            # Supplied structured rules participate exactly like typed functions.
            for rule in self._eligible_rules(goal):
                choices=[]
                for p in rule.inputs:
                    matches=[x for x in pool if p.value_type in {"any",x[1]} or (p.value_type=="number" and x[1]=="integer")]
                    if not matches:choices=[];break
                    choices.append(matches)
                for selection in itertools.product(*choices) if choices else [()]:
                    budget[0]+=1
                    if budget[0]>=goal.candidate_budget:return collected(additions),True
                    expr=Call(rule.rule_id,tuple(x[0] for x in selection))
                    try:out=_infer(expr,{x.name:x.value_type for x in goal.inputs},self.language.rules,set())
                    except UOLError:continue
                    if 1+max((x[2] for x in selection),default=0)>depth:continue
                    key=json.dumps(expr_dict(expr),sort_keys=True)
                    if key not in seen:additions.append((expr,out,depth))
            for item in additions:
                key=json.dumps(expr_dict(item[0]),sort_keys=True)
                if key not in seen:seen[key]=item
            front=additions
            if not front:break
        return list(seen.values()),False
    @staticmethod
    def _rule_refs(expr,known):
        refs=set()
        def visit(e):
            if isinstance(e,Call):
                if e.operator in known:refs.add(e.operator)
                for a in e.args:visit(a)
            elif isinstance(e,If):visit(e.condition);visit(e.yes);visit(e.no)
        visit(expr);return tuple(sorted(refs))
    def synthesize(self,goal:SynthesisGoal)->SynthesisReport:
        if not isinstance(goal,SynthesisGoal):raise TypeError("goal must be SynthesisGoal")
        budget=[0];candidates,exhausted=self._generate(goal,budget);tested=0
        for expr,typ,depth in sorted(candidates,key=lambda x:(x[2],len(json.dumps(expr_dict(x[0]))))):
            if typ not in {goal.output_type,"any"} and not(goal.output_type=="number" and typ=="integer"):continue
            refs=self._rule_refs(expr,{r.rule_id for r in self.language.rules.values()})
            program=Program("uol-candidate:"+hashlib.sha256(json.dumps(expr_dict(expr),sort_keys=True).encode()).hexdigest()[:20],goal.name,goal.inputs,expr,goal.domain,rule_refs=refs)
            try:self.language.validate(program)
            except UOLError:continue
            tested+=1;train=valid=0;ok=True
            for sample in goal.training:
                try:actual=self.language.execute(program,sample.inputs)
                except (UOLError,RuntimeError,TypeError,ValueError,OverflowError):ok=False;break
                if not self._equal(actual,sample.expected):ok=False;break
                train+=1
            if not ok or train!=len(goal.training):continue
            for sample in goal.validation:
                try:actual=self.language.execute(program,sample.inputs)
                except (UOLError,RuntimeError,TypeError,ValueError,OverflowError):ok=False;break
                if not self._equal(actual,sample.expected):ok=False;break
                valid+=1
            if ok and valid==len(goal.validation):return SynthesisReport(SynthesisStatus.SYNTHESIZED,program,tested,train,valid,refs)
        status=SynthesisStatus.BUDGET_EXHAUSTED if exhausted else SynthesisStatus.NO_SOLUTION
        return SynthesisReport(status,None,tested,0,0,blockers=("search budget exhausted" if exhausted else "no candidate passed all held-out examples",))
    @staticmethod
    def _equal(a,b):
        if isinstance(a,bool) or isinstance(b,bool):return type(a) is type(b) and a==b
        if isinstance(a,(int,float)) and isinstance(b,(int,float)):return math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9)
        return a==b
    def to_candidate_capability(self,report:SynthesisReport,*,intent="any",output_name="result"):
        if not report.succeeded:raise SynthesisError("only a synthesized report can become a capability candidate")
        p=report.program
        return CapabilitySpec(capability_id=p.program_id,name=p.name,description="Synthesized UOL candidate; requires independent verification",domains=(p.domain,),intents=(intent,),inputs=tuple(PortContract(x.name,x.value_type) for x in p.inputs),outputs=(PortContract(output_name,"any"),),preconditions=("UOL preconditions",) if p.preconditions else (),postconditions=("UOL postconditions",) if p.postconditions else (),provenance=RepresentationProvenance(source="mirror_uol_synthesizer",source_ref=p.fingerprint,parent_ids=report.rule_ids),verification=VerificationEvidence(level=EvidenceLevel.UNVERIFIED))
