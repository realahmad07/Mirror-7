"""Mirror Universal Operational Language: typed, serializable, bounded operations."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import hashlib, json, math, re
from typing import Any, Iterable, Mapping

class UOLError(ValueError): pass
class UOLExecutionError(RuntimeError): pass
class RuleStatus(str, Enum):
    UNKNOWN="unknown"; ASSUMED="assumed"; OBSERVED="observed"; DERIVED="derived"; VERIFIED="verified"; DISPUTED="disputed"

_TYPES={"any","null","boolean","integer","number","text","array","record"}
_ALIASES={"bool":"boolean","int":"integer","float":"number","numeric":"number","string":"text","object":"record","list":"array"}
_OPS={"add":2,"subtract":2,"multiply":2,"divide":2,"modulo":2,"power":2,"minimum":2,"maximum":2,"equal":2,"less_than":2,"less_equal":2,"greater_than":2,"greater_equal":2,"and":2,"or":2,"not":1,"negate":1,"absolute":1,"concat":2,"length":1,"sum":1,"product":1}
_ID=re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]{0,127}$")

def _type_name(value):
    name=_ALIASES.get(str(value).strip().lower(),str(value).strip().lower())
    if name not in _TYPES: raise UOLError(f"unsupported type: {value}")
    return name

def _freeze(value):
    if value is None or isinstance(value,(str,bool,int)): return value
    if isinstance(value,float):
        if not math.isfinite(value): raise UOLError("non-finite number")
        return value
    if isinstance(value,Mapping):
        if any(not isinstance(k,str) for k in value): raise UOLError("record keys must be text")
        return {k:_freeze(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return tuple(_freeze(v) for v in value)
    raise UOLError(f"unsupported value: {type(value).__name__}")

def _kind(value):
    if value is None:return "null"
    if isinstance(value,bool):return "boolean"
    if isinstance(value,int):return "integer"
    if isinstance(value,float):return "number"
    if isinstance(value,str):return "text"
    if isinstance(value,(list,tuple)):return "array"
    if isinstance(value,Mapping):return "record"
    return "any"

def _matches(value,expected):
    expected=_type_name(expected); actual=_kind(value)
    return expected=="any" or actual==expected or (expected=="number" and actual=="integer")

@dataclass(frozen=True)
class VariableSpec:
    name:str; value_type:str="any"
    def __post_init__(self):
        if not _ID.fullmatch(self.name): raise UOLError("invalid variable name")
        object.__setattr__(self,"value_type",_type_name(self.value_type))

@dataclass(frozen=True)
class Lit: value:Any
@dataclass(frozen=True)
class Var: name:str
@dataclass(frozen=True)
class Call:
    operator:str; args:tuple[Any,...]
    def __post_init__(self):
        if not _ID.fullmatch(self.operator): raise UOLError("invalid operator id")
        object.__setattr__(self,"args",tuple(self.args))
@dataclass(frozen=True)
class If:
    condition:Any; yes:Any; no:Any
@dataclass(frozen=True)
class ArrayExpr: items:tuple[Any,...]
@dataclass(frozen=True)
class RecordExpr: fields:Mapping[str,Any]
@dataclass(frozen=True)
class Field:
    target:Any; name:str
@dataclass(frozen=True)
class Index:
    target:Any; index:Any
@dataclass(frozen=True)
class Let:
    name:str; value:Any; body:Any
@dataclass(frozen=True)
class MapExpr:
    items:Any; variable:str; body:Any
@dataclass(frozen=True)
class FoldExpr:
    items:Any; initial:Any; accumulator:str; variable:str; body:Any

Expression=(Lit|Var|Call|If|ArrayExpr|RecordExpr|Field|Index|Let|MapExpr|FoldExpr)

def expr_dict(e):
    if isinstance(e,Lit):return {"op":"lit","value":_freeze(e.value)}
    if isinstance(e,Var):return {"op":"var","name":e.name}
    if isinstance(e,Call):return {"op":"call","name":e.operator,"args":[expr_dict(x) for x in e.args]}
    if isinstance(e,If):return {"op":"if","condition":expr_dict(e.condition),"yes":expr_dict(e.yes),"no":expr_dict(e.no)}
    if isinstance(e,ArrayExpr):return {"op":"array","items":[expr_dict(x) for x in e.items]}
    if isinstance(e,RecordExpr):return {"op":"record","fields":{k:expr_dict(v) for k,v in e.fields.items()}}
    if isinstance(e,Field):return {"op":"field","target":expr_dict(e.target),"name":e.name}
    if isinstance(e,Index):return {"op":"index","target":expr_dict(e.target),"index":expr_dict(e.index)}
    if isinstance(e,Let):return {"op":"let","name":e.name,"value":expr_dict(e.value),"body":expr_dict(e.body)}
    if isinstance(e,MapExpr):return {"op":"map","items":expr_dict(e.items),"variable":e.variable,"body":expr_dict(e.body)}
    if isinstance(e,FoldExpr):return {"op":"fold","items":expr_dict(e.items),"initial":expr_dict(e.initial),"accumulator":e.accumulator,"variable":e.variable,"body":expr_dict(e.body)}
    raise UOLError("unsupported UOL expression")

def expr_from_dict(d):
    if not isinstance(d,Mapping):raise UOLError("expression must be an object")
    op=d.get("op")
    if op=="lit":return Lit(d.get("value"))
    if op=="var":return Var(str(d["name"]))
    if op=="call":return Call(str(d["name"]),tuple(expr_from_dict(x) for x in d.get("args",())))
    if op=="if":return If(expr_from_dict(d["condition"]),expr_from_dict(d["yes"]),expr_from_dict(d["no"]))
    if op=="array":return ArrayExpr(tuple(expr_from_dict(x) for x in d.get("items",())))
    if op=="record":return RecordExpr({k:expr_from_dict(v) for k,v in d["fields"].items()})
    if op=="field":return Field(expr_from_dict(d["target"]),str(d["name"]))
    if op=="index":return Index(expr_from_dict(d["target"]),expr_from_dict(d["index"]))
    if op=="let":return Let(str(d["name"]),expr_from_dict(d["value"]),expr_from_dict(d["body"]))
    if op=="map":return MapExpr(expr_from_dict(d["items"]),str(d["variable"]),expr_from_dict(d["body"]))
    if op=="fold":return FoldExpr(expr_from_dict(d["items"]),expr_from_dict(d["initial"]),str(d["accumulator"]),str(d["variable"]),expr_from_dict(d["body"]))
    raise UOLError(f"unknown expression opcode: {op}")

@dataclass(frozen=True)
class OperationalRule:
    rule_id:str; name:str; inputs:tuple[VariableSpec,...]; body:Expression
    domains:tuple[str,...]=("general",); status:RuleStatus=RuleStatus.ASSUMED; provenance:str="user_supplied"
    def __post_init__(self):
        if not _ID.fullmatch(self.rule_id) or not self.name.strip():raise UOLError("invalid rule identity")
        object.__setattr__(self,"inputs",tuple(self.inputs)); object.__setattr__(self,"domains",tuple(x.lower() for x in self.domains)); object.__setattr__(self,"status",RuleStatus(self.status))
        if len({x.name for x in self.inputs})!=len(self.inputs):raise UOLError("duplicate rule input")
        if not self.domains:raise UOLError("rule needs a domain")
    def to_dict(self):return {"rule_id":self.rule_id,"name":self.name,"inputs":[{"name":x.name,"value_type":x.value_type} for x in self.inputs],"body":expr_dict(self.body),"domains":list(self.domains),"status":self.status.value,"provenance":self.provenance}

class RuleCatalog:
    def __init__(self,rules:Iterable[OperationalRule]=()):
        self._rules={}
        for r in rules:
            if r.rule_id in self._rules and self._rules[r.rule_id]!=r:raise UOLError("conflicting rule id")
            self._rules[r.rule_id]=r
        self.validate()
    def values(self):return tuple(self._rules[k] for k in sorted(self._rules))
    def get(self,key):return self._rules.get(key)
    def validate(self):
        for r in self.values():
            refs={x.operator for x in _walk(r.body) if isinstance(x,Call) and x.operator not in _OPS}
            missing=refs-set(self._rules)
            if missing:raise UOLError(f"missing rule references: {sorted(missing)}")
            if r.rule_id in refs:raise UOLError("recursive rule")
            _infer(r.body,{x.name:x.value_type for x in r.inputs},self,set())
    @classmethod
    def from_knowledge_records(cls, records):
        from .knowledge_uol import KnowledgeRuleImporter
        return KnowledgeRuleImporter().import_records(records).catalog()

def _walk(e):
    yield e
    if isinstance(e,Call):
        for x in e.args:yield from _walk(x)
    elif isinstance(e,If):
        for x in (e.condition,e.yes,e.no):yield from _walk(x)
    elif isinstance(e,ArrayExpr):
        for x in e.items:yield from _walk(x)
    elif isinstance(e,RecordExpr):
        for x in e.fields.values():yield from _walk(x)
    elif isinstance(e,(Field,MapExpr)):
        yield from _walk(e.target if isinstance(e,Field) else e.items)
        yield from _walk(e.body) if isinstance(e,MapExpr) else ()
    elif isinstance(e,Index):yield from _walk(e.target);yield from _walk(e.index)
    elif isinstance(e,Let):yield from _walk(e.value);yield from _walk(e.body)
    elif isinstance(e,FoldExpr):yield from _walk(e.items);yield from _walk(e.initial);yield from _walk(e.body)

def _infer(e,env,rules,active):
    if isinstance(e,Lit):return _kind(e.value)
    if isinstance(e,Var):
        if e.name not in env:raise UOLError(f"unknown variable: {e.name}")
        return env[e.name]
    if isinstance(e,Call):
        ts=tuple(_infer(x,env,rules,active) for x in e.args)
        if e.operator in _OPS:
            arity=_OPS[e.operator]
            if len(ts)!=arity:raise UOLError(f"{e.operator} expects {arity} args")
            if e.operator in {"add","subtract","multiply","divide","modulo","power","minimum","maximum","negate","absolute","sum","product"}:
                if (e.operator in {"sum","product"} and ts[0]!="array") or (e.operator not in {"sum","product"} and any(x not in {"integer","number","any"} for x in ts)):raise UOLError("numeric operator type mismatch")
                return "number" if e.operator in {"divide","power","sum","product"} or "number" in ts else "integer"
            if e.operator in {"and","or","not"} and any(x not in {"boolean","any"} for x in ts):raise UOLError("boolean operator type mismatch")
            if e.operator in {"less_than","less_equal","greater_than","greater_equal"} and not(all(x in {"integer","number","any"} for x in ts) or all(x=="text" for x in ts)):raise UOLError("comparison type mismatch")
            if e.operator=="concat" and any(x not in {"text","any"} for x in ts):raise UOLError("concat type mismatch")
            if e.operator=="length" and ts[0] not in {"text","array","record","any"}:raise UOLError("length type mismatch")
            return "boolean" if e.operator in {"equal","less_than","less_equal","greater_than","greater_equal","and","or","not"} else "text" if e.operator=="concat" else "integer" if e.operator=="length" else "number" if e.operator in {"divide","power","sum","product"} or "number" in ts else "integer"
        r=rules.get(e.operator)
        if r is None or e.operator in active:raise UOLError(f"unknown or recursive rule {e.operator}")
        if len(ts)!=len(r.inputs):raise UOLError("rule arity mismatch")
        local={}
        for p,t in zip(r.inputs,ts):
            if p.value_type not in {"any",t} and not(p.value_type=="number" and t=="integer"):raise UOLError("rule input type mismatch")
            local[p.name]=t
        return _infer(r.body,local,rules,active|{e.operator})
    if isinstance(e,If):
        if _infer(e.condition,env,rules,active) not in {"boolean","any"}:raise UOLError("if condition must be boolean")
        a,b=_infer(e.yes,env,rules,active),_infer(e.no,env,rules,active)
        if a!=b and not {a,b}<={"integer","number"}:raise UOLError("if branch type mismatch")
        return "number" if "number" in {a,b} else a
    if isinstance(e,ArrayExpr):
        for x in e.items:_infer(x,env,rules,active)
        return "array"
    if isinstance(e,RecordExpr):
        for x in e.fields.values():_infer(x,env,rules,active)
        return "record"
    if isinstance(e,Field):
        if _infer(e.target,env,rules,active) not in {"record","any"}:raise UOLError("field access requires record")
        return "any"
    if isinstance(e,Index):
        if _infer(e.target,env,rules,active) not in {"array","any"} or _infer(e.index,env,rules,active) not in {"integer","any"}:raise UOLError("index type mismatch")
        return "any"
    if isinstance(e,Let):
        local=dict(env);local[e.name]=_infer(e.value,env,rules,active);return _infer(e.body,local,rules,active)
    if isinstance(e,MapExpr):
        if _infer(e.items,env,rules,active) not in {"array","any"}:raise UOLError("map needs array")
        local=dict(env);local[e.variable]="any";_infer(e.body,local,rules,active);return "array"
    if isinstance(e,FoldExpr):
        if _infer(e.items,env,rules,active) not in {"array","any"}:raise UOLError("fold needs array")
        t=_infer(e.initial,env,rules,active);local=dict(env);local[e.accumulator]=t;local[e.variable]="any";u=_infer(e.body,local,rules,active)
        if t!=u and "any" not in {t,u}:raise UOLError("fold must preserve accumulator type")
        return t
    raise UOLError("invalid expression")

def knowledge_rule(record, *, force_disputed=False):
    """Adapt one explicitly structured KnowledgeRecord without upgrading trust."""
    payload = record.content["uol_rule"]
    verification = record.verification
    level = getattr(verification.level, "value", verification.level)
    status = getattr(record.epistemic_status, "value", record.epistemic_status)
    provenance = record.provenance.to_dict()
    content_hash = provenance.get("content_sha256", "")
    has_source_lineage = (
        provenance.get("source") not in {"", "unknown"}
        and bool(provenance.get("source_ref"))
        and isinstance(content_hash, str)
        and len(content_hash) == 64
        and all(ch in "0123456789abcdef" for ch in content_hash)
    )
    has_verified_evidence = (
        level in {"formally_verified", "independently_reproduced"}
        and bool(verification.checks_passed)
        and not verification.checks_failed
        and bool(verification.verifier.strip())
        and has_source_lineage
    )
    if force_disputed or verification.checks_failed or status == "disputed":
        evidence = RuleStatus.DISPUTED
    elif has_verified_evidence:
        evidence = RuleStatus.VERIFIED
    elif status == "assumed":
        evidence = RuleStatus.ASSUMED
    elif status == "observed":
        evidence = RuleStatus.OBSERVED
    elif status == "derived":
        evidence = RuleStatus.DERIVED
    else:
        evidence = RuleStatus.UNKNOWN
    lineage = {
        "knowledge_id": record.knowledge_id,
        "knowledge_fingerprint": record.fingerprint,
        "epistemic_status": status,
        "verification": {
            "level": level,
            "checks_passed": list(verification.checks_passed),
            "checks_failed": list(verification.checks_failed),
            "verifier": verification.verifier,
            "verified_at": verification.verified_at,
        },
        "provenance": provenance,
    }
    return OperationalRule(
        payload.get("rule_id", record.knowledge_id),
        payload.get("name", record.subject),
        tuple(VariableSpec(item["name"], item["value_type"]) for item in payload["inputs"]),
        expr_from_dict(payload["body"]),
        tuple(payload.get("domains", (record.domain,))),
        evidence,
        "knowledge-record:" + json.dumps(lineage, sort_keys=True, separators=(",", ":")),
    )

@dataclass(frozen=True)
class Program:
    program_id:str; name:str; inputs:tuple[VariableSpec,...]; body:Expression; domain:str="general"; preconditions:tuple[Expression,...]=(); postconditions:tuple[Expression,...]=(); rule_refs:tuple[str,...]=()
    @property
    def fingerprint(self):return hashlib.sha256(json.dumps(program_dict(self),sort_keys=True,separators=(",",":")).encode()).hexdigest()

def program_dict(p):return {"program_id":p.program_id,"name":p.name,"inputs":[{"name":x.name,"value_type":x.value_type} for x in p.inputs],"body":expr_dict(p.body),"domain":p.domain,"preconditions":[expr_dict(x) for x in p.preconditions],"postconditions":[expr_dict(x) for x in p.postconditions],"rule_refs":list(p.rule_refs)}
def program_from_dict(d):return Program(d["program_id"],d["name"],tuple(VariableSpec(x["name"],x["value_type"]) for x in d["inputs"]),expr_from_dict(d["body"]),d.get("domain","general"),tuple(expr_from_dict(x) for x in d.get("preconditions",())),tuple(expr_from_dict(x) for x in d.get("postconditions",())),tuple(d.get("rule_refs",())))
def build_program(name,inputs,body,*,domain="general",preconditions=(),postconditions=()):
    refs=tuple(sorted({x.operator for expr in (body,)+tuple(preconditions)+tuple(postconditions) for x in _walk(expr) if isinstance(x,Call) and x.operator not in _OPS}))
    raw={"name":name,"inputs":[{"name":x.name,"value_type":x.value_type} for x in inputs],"body":expr_dict(body),"domain":domain,"preconditions":[expr_dict(x) for x in preconditions],"postconditions":[expr_dict(x) for x in postconditions],"rule_refs":list(refs)}
    return Program("uol:"+hashlib.sha256(json.dumps(raw,sort_keys=True,separators=(",",":")).encode()).hexdigest()[:24],name,tuple(inputs),body,domain,tuple(preconditions),tuple(postconditions),refs)

class UOL:
    VERSION="uol-v1"
    def __init__(self,rules=(),*,max_steps=10000,max_depth=64,max_items=10000,max_magnitude=1e100):
        for name,value in (("max_steps",max_steps),("max_depth",max_depth),("max_items",max_items)):
            if isinstance(value,bool) or not isinstance(value,int) or value<1:raise UOLError(f"{name} must be a positive integer")
        if isinstance(max_magnitude,bool) or not isinstance(max_magnitude,(int,float)) or not math.isfinite(max_magnitude) or max_magnitude<=0:raise UOLError("max_magnitude must be finite and positive")
        self.rules=RuleCatalog(rules);self.max_steps=max_steps;self.max_depth=max_depth;self.max_items=max_items;self.max_magnitude=max_magnitude
    def validate(self,p):
        env={x.name:x.value_type for x in p.inputs}
        t=_infer(p.body,env,self.rules,set())
        actual={x.operator for expr in (p.body,)+p.preconditions+p.postconditions for x in _walk(expr) if isinstance(x,Call) and x.operator not in _OPS}
        if actual!=set(p.rule_refs):raise UOLError("program rule references do not match its expression")
        for x in p.preconditions:
            if _infer(x,env,self.rules,set()) not in {"boolean","any"}:raise UOLError("precondition must be boolean")
        for x in p.postconditions:
            if _infer(x,{**env,"$result":t},self.rules,set()) not in {"boolean","any"}:raise UOLError("postcondition must be boolean")
        pending=list(actual);seen=set()
        while pending:
            rid=pending.pop()
            if rid in seen:continue
            seen.add(rid);rule=self.rules.get(rid)
            if not rule or (p.domain not in rule.domains and "general" not in rule.domains and "*" not in rule.domains):raise UOLError("missing or cross-domain rule")
            pending.extend(x.operator for x in _walk(rule.body) if isinstance(x,Call) and x.operator not in _OPS)
        return t
    def execute(self,p,inputs):
        output_type=self.validate(p);specs={x.name:x for x in p.inputs}
        if not isinstance(inputs,Mapping) or set(inputs)!=set(specs):raise UOLExecutionError("inputs must match the declared schema")
        env={}
        for name,value in inputs.items():
            value=_freeze(value)
            if not _matches(value,specs[name].value_type):raise UOLError(f"input {name} type mismatch")
            self._bound(value);env[name]=value
        budget=[0]
        for c in p.preconditions:
            if self._eval(c,env,budget,0,()) is not True:raise UOLExecutionError("precondition failed")
        result=self._eval(p.body,env,budget,0,());self._bound(result)
        if not _matches(result,output_type):raise UOLExecutionError("output type mismatch")
        env["$result"]=result
        for c in p.postconditions:
            if self._eval(c,env,budget,0,()) is not True:raise UOLExecutionError("postcondition failed")
        return result
    def _bound(self,v,depth=0):
        if depth>self.max_depth:raise UOLExecutionError("value depth limit")
        k=_kind(v)
        if k in {"integer","number"} and abs(v)>self.max_magnitude:raise UOLExecutionError("numeric magnitude limit")
        if k=="text" and len(v)>self.max_items*100:raise UOLExecutionError("text length limit")
        if k in {"array","record"}:
            if len(v)>self.max_items:raise UOLExecutionError("collection size limit")
            for x in (v.values() if k=="record" else v):self._bound(x,depth+1)
    def _eval(self,e,env,budget,depth,active):
        budget[0]+=1
        if budget[0]>self.max_steps or depth>self.max_depth:raise UOLExecutionError("execution budget exceeded")
        ev=lambda x:self._eval(x,env,budget,depth+1,active)
        if isinstance(e,Lit):return _freeze(e.value)
        if isinstance(e,Var):
            if e.name not in env:raise UOLExecutionError("unknown variable")
            return env[e.name]
        if isinstance(e,Call):
            a=tuple(ev(x) for x in e.args)
            if e.operator not in _OPS:
                r=self.rules.get(e.operator)
                if not r or r.status in {RuleStatus.UNKNOWN,RuleStatus.DISPUTED} or r.rule_id in active:raise UOLExecutionError("rule blocked")
                local={}
                for p,v in zip(r.inputs,a):
                    if not _matches(v,p.value_type):raise UOLError("rule argument type mismatch")
                    local[p.name]=v
                return self._eval(r.body,local,budget,depth+1,active+(r.rule_id,))
            op=e.operator
            if op in {"add","subtract","multiply","divide","modulo","power","minimum","maximum","negate","absolute","sum","product"}:
                vals=a[0] if op in {"sum","product"} else a
                if any(isinstance(x,bool) or not isinstance(x,(int,float)) for x in vals):raise UOLExecutionError("numeric operator received non-number")
                if op=="sum":out=sum(vals)
                elif op=="product":
                    out=1
                    for v in vals:out*=v
                elif op=="add":out=a[0]+a[1]
                elif op=="subtract":out=a[0]-a[1]
                elif op=="multiply":out=a[0]*a[1]
                elif op=="divide":
                    if a[1]==0:raise UOLExecutionError("division by zero")
                    out=a[0]/a[1]
                elif op=="modulo":
                    if a[1]==0:raise UOLExecutionError("modulo by zero")
                    out=a[0]%a[1]
                elif op=="power":
                    if abs(a[1])>16 or abs(a[0])>1e6:raise UOLExecutionError("power bounds exceeded")
                    out=a[0]**a[1]
                elif op=="minimum":out=min(a)
                elif op=="maximum":out=max(a)
                elif op=="negate":out=-a[0]
                else:out=abs(a[0])
                self._bound(out);return out
            if op=="equal":return a[0]==a[1]
            if op=="less_than":return a[0]<a[1]
            if op=="less_equal":return a[0]<=a[1]
            if op=="greater_than":return a[0]>a[1]
            if op=="greater_equal":return a[0]>=a[1]
            if op in {"and","or","not"} and any(not isinstance(x,bool) for x in a):raise UOLExecutionError("boolean operator received a non-boolean")
            if op=="and":return a[0] and a[1]
            if op=="or":return a[0] or a[1]
            if op=="not":return not a[0]
            if op=="concat":
                if not all(isinstance(x,str) for x in a):raise UOLExecutionError("concat needs text")
                out=a[0]+a[1];self._bound(out);return out
            if op=="length":return len(a[0])
        if isinstance(e,If):
            condition=ev(e.condition)
            if not isinstance(condition,bool):raise UOLExecutionError("if condition must be boolean")
            return self._eval(e.yes if condition else e.no,env,budget,depth+1,active)
        if isinstance(e,ArrayExpr):return tuple(ev(x) for x in e.items)
        if isinstance(e,RecordExpr):return {k:ev(v) for k,v in e.fields.items()}
        if isinstance(e,Field):
            v=ev(e.target)
            if not isinstance(v,Mapping) or e.name not in v:raise UOLExecutionError("record field missing")
            return v[e.name]
        if isinstance(e,Index):
            v,i=ev(e.target),ev(e.index)
            if not isinstance(v,(tuple,list)) or isinstance(i,bool) or not isinstance(i,int):raise UOLExecutionError("invalid array index")
            return v[i]
        if isinstance(e,Let):
            v=ev(e.value);local=dict(env);local[e.name]=v;return self._eval(e.body,local,budget,depth+1,active)
        if isinstance(e,MapExpr):
            items=ev(e.items)
            if not isinstance(items,(tuple,list)) or len(items)>self.max_items:raise UOLExecutionError("invalid map collection")
            out=[]
            for x in items:local=dict(env);local[e.variable]=x;out.append(self._eval(e.body,local,budget,depth+1,active))
            return tuple(out)
        if isinstance(e,FoldExpr):
            items,acc=ev(e.items),ev(e.initial)
            if not isinstance(items,(tuple,list)) or len(items)>self.max_items:raise UOLExecutionError("invalid fold collection")
            for x in items:local=dict(env);local[e.accumulator]=acc;local[e.variable]=x;acc=self._eval(e.body,local,budget,depth+1,active);self._bound(acc)
            return acc
        raise UOLExecutionError("unsupported expression node")
    def native_handler(self,p,output_name="result"):
        self.validate(p)
        return lambda values:{output_name:self.execute(p,values)}
