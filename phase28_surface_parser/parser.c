#include <ctype.h>
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "surface_ir.h"

typedef struct { size_t pos; size_t depth; unsigned seen_else[1025]; unsigned has_term[1025]; } parser_t;
static int is_name_start(int c){ return isalpha((unsigned char)c)||c=='_'||c=='-'; }
static int is_name_char(int c){ return isalnum((unsigned char)c)||c=='_'||c=='-'; }
static int skipws(const char*s,size_t n,size_t*p){ while(*p<n && isspace((unsigned char)s[*p])) (*p)++; return *p<n; }
static int token(const char*s,size_t n,size_t*p,const char**a,size_t*len){
  if(!skipws(s,n,p)) return 0;
  size_t st=*p;
  while(*p<n&&!isspace((unsigned char)s[*p])) (*p)++;
  *a=s+st; *len=*p-st; return 1;
}
static int eq(const char*a,size_t n,const char*b){ return strlen(b)==n && memcmp(a,b,n)==0; }
static int looks_numeric(const char*a,size_t n){
  if(!n)return 0;
  size_t i=(a[0]=='-'||a[0]=='+');
  if(i==n)return 0;
  for(;i<n;i++) if(!isdigit((unsigned char)a[i])) return 0;
  return 1;
}
static int valid_number(const char*a,size_t n){
  if(!n)return 0;
  size_t i=0;
  if(a[0]=='-'||a[0]=='+'){ if(n==1)return 0; i=1; }
  if(i==n)return 0;
  for(;i<n;i++) if(!isdigit((unsigned char)a[i])) return 0;
  errno=0; char buf[32];
  if(n>=sizeof buf)return 0;
  memcpy(buf,a,n); buf[n]=0;
  char*e; long v=strtol(buf,&e,10);
  return errno==0 && *e==0 && v>=0 && v<=65535;
}
static int copy_token(mirror7_surface_ir_t *out, uint8_t kind, const char *a, size_t len,
                      uint32_t pos, uint16_t number) {
  if (out->token_count >= MIRROR7_SURFACE_IR_MAX_TOKENS) return 0;
  if (kind != MIRROR7_IR_NUMBER && len > MIRROR7_SURFACE_IR_MAX_NAME) return 0;
  mirror7_ir_token_t *t = &out->tokens[out->token_count++];
  t->kind = kind;
  t->number = number;
  t->length = (uint16_t)len;
  t->source_pos = pos;
  if (len) memcpy(t->text, a, len);
  t->text[len] = 0;
  return 1;
}

int mirror7_surface_ir_validate(const mirror7_surface_ir_t *ir) {
  if (!ir || ir->version != MIRROR7_SURFACE_IR_VERSION || ir->token_count > MIRROR7_SURFACE_IR_MAX_TOKENS)
    return 0;
  for (uint16_t i = 0; i < ir->token_count; ++i) {
    const mirror7_ir_token_t *t = &ir->tokens[i];
    if (t->kind < MIRROR7_IR_COLON || t->kind > MIRROR7_IR_THEN) return 0;
    if (t->kind == MIRROR7_IR_NUMBER) {
      if (t->length != 0 || t->text[0] != 0) return 0;
    } else {
      if (t->length > MIRROR7_SURFACE_IR_MAX_NAME || t->text[t->length] != 0) return 0;
    }
  }
  return 1;
}

int mirror7_parse_ir(const char*s,size_t n,mirror7_surface_ir_t*out){
  if (!out || (!s && n!=0)) return 0;
  memset(out, 0, sizeof *out);
  out->version = MIRROR7_SURFACE_IR_VERSION;
  parser_t p={0}; int in_def=0; const char*a; size_t l;
  while(token(s,n,&p.pos,&a,&l)){
    size_t token_pos = (size_t)(a - s);
    uint8_t kind = 0; uint16_t number = 0;
    if(eq(a,l,":")){
      if(in_def) return 0;
      if(!copy_token(out,MIRROR7_IR_COLON,a,l,(uint32_t)token_pos,0)) return 0;
      if(!token(s,n,&p.pos,&a,&l)||l==0)return 0;
      token_pos=(size_t)(a-s);
      if(!is_name_start((unsigned char)a[0]))return 0;
      for(size_t i=1;i<l;i++)if(!is_name_char((unsigned char)a[i]))return 0;
      if(!copy_token(out,MIRROR7_IR_NAME,a,l,(uint32_t)token_pos,0)) return 0;
      in_def=1;p.depth=0;
      memset(p.seen_else,0,sizeof p.seen_else);
      memset(p.has_term,0,sizeof p.has_term);
      continue;
    }
    if(!in_def) return 0;
    if(eq(a,l,";")){ if(p.depth!=0)return 0; if(!copy_token(out,MIRROR7_IR_SEMI,a,l,(uint32_t)token_pos,0)) return 0; in_def=0; continue; }
    if(eq(a,l,"IF")){ if(p.depth>=1024)return 0; if(!copy_token(out,MIRROR7_IR_IF,a,l,(uint32_t)token_pos,0)) return 0; p.depth++; p.seen_else[p.depth]=0; p.has_term[p.depth]=0; continue; }
    if(eq(a,l,"ELSE")){ if(p.depth==0||p.seen_else[p.depth]||!p.has_term[p.depth])return 0; if(!copy_token(out,MIRROR7_IR_ELSE,a,l,(uint32_t)token_pos,0)) return 0; p.seen_else[p.depth]=1; p.has_term[p.depth]=0; continue; }
    if(eq(a,l,"THEN")){ if(p.depth==0||!p.has_term[p.depth])return 0; if(!copy_token(out,MIRROR7_IR_THEN,a,l,(uint32_t)token_pos,0)) return 0; p.seen_else[p.depth]=0; p.depth--; if(p.depth>0)p.has_term[p.depth]=1; continue; }
    if(valid_number(a,l)) { errno=0; char buf[32]; memcpy(buf,a,l); buf[l]=0; number=(uint16_t)strtoul(buf,NULL,10); kind=MIRROR7_IR_NUMBER; if(!copy_token(out,kind,a,0,(uint32_t)token_pos,number)) return 0; p.has_term[p.depth]=1; continue; }
    if(looks_numeric(a,l)) return 0;
    if(l==0||l>63)return 0;
    for(size_t i=0;i<l;i++) if(!is_name_char((unsigned char)a[i])) return 0;
    if(!copy_token(out,MIRROR7_IR_NAME,a,l,(uint32_t)token_pos,0)) return 0;
    p.has_term[p.depth]=1;
  }
  if(in_def || p.depth!=0) return 0;
  return mirror7_surface_ir_validate(out);
}

int mirror7_parse(const char*s,size_t n){
  mirror7_surface_ir_t ir;
  return mirror7_parse_ir(s,n,&ir);
}

#ifndef MIRROR7_NO_MAIN
#ifndef TEST
int main(int argc,char**argv){
  if(argc!=2){fprintf(stderr,"usage: %s '<mirr-source>'\n",argv[0]);return 2;}
  return mirror7_parse(argv[1],strlen(argv[1]))?0:1;
}
#else
static int t(const char*s,int want){int got=mirror7_parse(s,strlen(s)); if(got!=want){fprintf(stderr,"FAIL: %s => %d want %d\n",s,got,want);return 1;}return 0;}
int main(void){
 const char*ok[]={
  ": a 1 ;",
  ": a IF 1 THEN ;",
  ": a IF 1 ELSE 2 THEN ;",
  ": a IF IF 1 THEN ELSE 2 THEN ;",
  ": foo 123 bar IF baz ELSE qux THEN ; : z 0 ;",
  ": - 0 ;",
  ": a +1 ;",
  ": a 65535 ;"
 };
 const char*bad[]={
  "ELSE",": a ELSE ;",": a THEN ;",": a IF 1 ;",
  ": a IF 1 ELSE 2 ELSE 3 THEN ;",": a ; ;",": ;",
  ": 1 2 ;",": a IF ELSE THEN ;",": a IF IF 1 THEN ;",
  ": a 65536 ;",": a -1 ;",": a + ;",": a @@@ ;"
 };
 for(size_t i=0;i<sizeof(ok)/sizeof(ok[0]);i++) if(t(ok[i],1)) return 1;
 for(size_t i=0;i<sizeof(bad)/sizeof(bad[0]);i++) if(t(bad[i],0)) return 1;
 unsigned x=0x12345678; char s[256];
 for(int i=0;i<100000;i++){
  x=x*1664525u+1013904223u; int n=(int)(x%40); int pos=0;
  pos+=snprintf(s+pos,sizeof s-pos,": w ");
  for(int j=0;j<n;j++){
   x=x*1664525u+1013904223u;
   static const char*q[]={"IF","ELSE","THEN","1","foo",";","@","-","65535","65536"};
   pos+=snprintf(s+pos,sizeof s-pos," %s",q[x%10]);
  }
  if(i%3==0) strncat(s," ;",sizeof s-strlen(s)-1);
  (void)mirror7_parse(s,strlen(s));
 }
 mirror7_surface_ir_t ir;
 if(!mirror7_parse_ir(": alpha 65 emit IF 1 ELSE 2 THEN ;", strlen(": alpha 65 emit IF 1 ELSE 2 THEN ;"), &ir)) return 1;
 if(ir.version != MIRROR7_SURFACE_IR_VERSION || ir.token_count != 10) return 1;
 if(ir.tokens[0].kind != MIRROR7_IR_COLON || ir.tokens[1].kind != MIRROR7_IR_NAME || ir.tokens[2].kind != MIRROR7_IR_NUMBER || ir.tokens[2].number != 65) return 1;
 if(ir.tokens[3].kind != MIRROR7_IR_NAME || ir.tokens[4].kind != MIRROR7_IR_IF || ir.tokens[6].kind != MIRROR7_IR_ELSE || ir.tokens[9].kind != MIRROR7_IR_SEMI) return 1;
 if(!mirror7_surface_ir_validate(&ir)) return 1;
 puts("PHASE28_PARSER_TEST_PASS"); return 0;
}
#endif
#endif
