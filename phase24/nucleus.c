#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAGIC "M7NX1"
#define MAX_WORDS 1024
#define MAX_NAME 63
#define MAX_BODY 65536
#define MAX_TOK 256
#define MAX_STACK 65536
#define MAX_RSTACK 16384
#define MAX_STEPS 10000000ULL

enum {OP_HALT=0,OP_LIT=1,OP_ADD=2,OP_SUB=3,OP_DUP=4,OP_DROP=5,OP_SWAP=6,OP_LOAD=7,OP_STORE=8,OP_CALL=9,OP_RET=10,OP_EMIT=11,OP_NEXT=12,OP_EQ=13,OP_JZ=14,OP_JMP=15,OP_NEXTC=16,OP_SRC_POS=17,OP_SRC_LEN=18,OP_WORD_COUNT=19,OP_WORD_NAME_LEN=20,OP_WORD_NAME_CHAR=21,OP_WORD_CODE_LEN=22,OP_WORD_CODE_BYTE=23,OP_WORD_NEW=24,OP_WORD_APPEND=25,OP_WORD_EXEC=26};

typedef struct { char name[MAX_NAME+1]; uint32_t off; uint32_t len; } word_t;
typedef struct { uint32_t pos; char name[MAX_NAME+1]; } pending_t;
typedef struct { uint8_t code[MAX_BODY]; size_t n; word_t words[MAX_WORDS]; size_t nw; pending_t pending[4096]; size_t npending; } dict_t;

typedef struct { int kind; uint8_t *src; size_t len,pos; uint8_t out[1<<20]; size_t outn; } tokio_t;

typedef struct { uint32_t *ret; size_t rp; uint8_t *st; size_t sp; uint8_t mem[65536]; uint32_t pc; uint64_t steps; tokio_t *io; dict_t *d; } vm_t;

static int next_tok(tokio_t *io,char *buf,size_t cap){
    while(io->pos<io->len && isspace(io->src[io->pos])) io->pos++;
    if(io->pos>=io->len) return 0;
    size_t n=0; while(io->pos<io->len && !isspace(io->src[io->pos])) { if(n+1>=cap)return -2; buf[n++]=(char)io->src[io->pos++]; }
    buf[n]=0; return 1;
}
static int find_word(dict_t *d,const char *name){ for(size_t i=0;i<d->nw;i++) if(strcmp(d->words[i].name,name)==0)return (int)i; return -1; }
static int emitb(dict_t *d,uint8_t b){ if(d->n>=MAX_BODY)return 0; d->code[d->n++]=b; return 1; }
static int emit_u16(dict_t*d,uint16_t x){ return emitb(d,(uint8_t)x)&&emitb(d,(uint8_t)(x>>8)); }
static int push(vm_t*v,uint8_t x){if(v->sp>=MAX_STACK)return 0;v->st[v->sp++]=x;return 1;}
static int popv(vm_t*v,uint8_t*x){if(!v->sp)return 0;*x=v->st[--v->sp];return 1;}
static int popr(vm_t*v,uint32_t*x){if(!v->rp)return 0;*x=v->ret[--v->rp];return 1;}
static int word_to_id(dict_t*d,const char*n){int i=find_word(d,n); return i<0?-1:i;}
static int create_word_from_mem(dict_t*d,const uint8_t*mem,uint16_t addr,uint16_t len){
    if(!len || len>MAX_NAME || d->nw>=MAX_WORDS) return -1;
    char name[MAX_NAME+1]; memcpy(name,mem+addr,len); name[len]=0;
    if(find_word(d,name)>=0) return -1;
    if(d->n>=MAX_BODY) return -1;
    size_t id=d->nw++; d->words[id].off=(uint32_t)d->n; d->words[id].len=0;
    memcpy(d->words[id].name,name,len+1);
    return (int)id;
}
static int append_word_byte(dict_t*d,uint16_t id,uint8_t b){
    if(id>=d->nw || d->n>=MAX_BODY) return 0;
    d->code[d->n++]=b; d->words[id].len++; return 1;
}

static int compile_token(dict_t*d,const char*t){
    if(!strcmp(t,"literal")){ if(!emitb(d,OP_LIT)||!emitb(d,0))return 0; return 1; }
    if(!strcmp(t,"+")) return emitb(d,OP_ADD);
    if(!strcmp(t,"-")) return emitb(d,OP_SUB);
    if(!strcmp(t,"dup")) return emitb(d,OP_DUP);
    if(!strcmp(t,"drop")) return emitb(d,OP_DROP);
    if(!strcmp(t,"swap")) return emitb(d,OP_SWAP);
    if(!strcmp(t,"@")) return emitb(d,OP_LOAD);
    if(!strcmp(t,"!")) return emitb(d,OP_STORE);
    if(!strcmp(t,"emit")) return emitb(d,OP_EMIT);
    if(!strcmp(t,"next")) return emitb(d,OP_NEXT);
    if(!strcmp(t,"nextc")) return emitb(d,OP_NEXTC);
    if(!strcmp(t,"src-pos")) return emitb(d,OP_SRC_POS);
    if(!strcmp(t,"src-len")) return emitb(d,OP_SRC_LEN);
    if(!strcmp(t,"word-count")) return emitb(d,OP_WORD_COUNT);
    if(!strcmp(t,"word-name-len")) return emitb(d,OP_WORD_NAME_LEN);
    if(!strcmp(t,"word-name-char")) return emitb(d,OP_WORD_NAME_CHAR);
    if(!strcmp(t,"word-code-len")) return emitb(d,OP_WORD_CODE_LEN);
    if(!strcmp(t,"word-code-byte")) return emitb(d,OP_WORD_CODE_BYTE);
    if(!strcmp(t,"word-new")) return emitb(d,OP_WORD_NEW);
    if(!strcmp(t,"word-append")) return emitb(d,OP_WORD_APPEND);
    if(!strcmp(t,"word-exec")) return emitb(d,OP_WORD_EXEC);
    if(!strcmp(t,"=") ) return emitb(d,OP_EQ);
    if(!strcmp(t,"0branch")){ if(!emitb(d,OP_JZ)||!emit_u16(d,0))return 0;return 1; }
    if(!strcmp(t,"branch")){ if(!emitb(d,OP_JMP)||!emit_u16(d,0))return 0;return 1; }
    if(!strcmp(t,"exit"))return emitb(d,OP_RET);
    char *e; long v=strtol(t,&e,0); if(*t && *e==0 && v>=0&&v<=255){ if(!emitb(d,OP_LIT)||!emitb(d,(uint8_t)v))return 0; return 1; }
    int id=word_to_id(d,t);
    if(id>=0){ if(!emitb(d,OP_CALL)||!emit_u16(d,(uint16_t)id))return 0; return 1; }
    if(d->npending>=sizeof(d->pending)/sizeof(d->pending[0]) || strlen(t)>MAX_NAME) return 0;
    if(!emitb(d,OP_CALL)) return 0;
    uint32_t pos=(uint32_t)d->n;
    if(!emit_u16(d,0)) return 0;
    strcpy(d->pending[d->npending].name,t); d->pending[d->npending].pos=pos; d->npending++;
    return 1;
}

/* Bootstrap language: : name ... ;
   All words other than the primitive nucleus are defined by the source itself. */
static int build_dict(uint8_t *src,size_t slen,dict_t*d){
    memset(d,0,sizeof(*d));
    char tok[MAX_TOK]; tokio_t io={.src=src,.len=slen};
    while(1){ int rc=next_tok(&io,tok,sizeof(tok)); if(rc==0)break; if(rc<0)return 0;
        if(!strcmp(tok,":")){
            char name[MAX_TOK];
            if(next_tok(&io,name,sizeof(name))!=1)return 0;
            if(find_word(d,name)>=0 || d->nw>=MAX_WORDS || strlen(name)>MAX_NAME)return 0;
            uint32_t start=(uint32_t)d->n;
            size_t word_id=d->nw;
            strcpy(d->words[word_id].name,name); d->words[word_id].off=start; d->words[word_id].len=0; d->nw++;
            while(1){
                rc=next_tok(&io,tok,sizeof(tok)); if(rc!=1)return 0;
                if(!strcmp(tok,";"))break;
                if(!compile_token(d,tok))return 0;
            }
            if(!emitb(d,OP_RET)) return 0;
            d->words[word_id].len=(uint32_t)(d->n-start);
        } else if(!strcmp(tok,";")){ return 0; }
        else { return 0; }
    }
    for(size_t i=0;i<d->npending;i++){
        int id=find_word(d,d->pending[i].name);
        if(id<0 || d->pending[i].pos+2>d->n) return 0;
        d->code[d->pending[i].pos]=(uint8_t)id;
        d->code[d->pending[i].pos+1]=(uint8_t)(id>>8);
    }
    return 1;
}

static int run_word(dict_t*d,int id,uint8_t *src,size_t slen){
    if(id<0||(size_t)id>=d->nw) return 0;
    vm_t v; uint8_t st[MAX_STACK]; uint32_t ret[MAX_RSTACK];
    memset(&v,0,sizeof(v)); v.st=st; v.ret=ret; v.d=d; v.pc=d->words[id].off; v.rp=0;
    tokio_t io={.src=src,.len=slen,.pos=0};v.io=&io;
    for(;;){if(v.steps++>=MAX_STEPS)return 0;if(v.pc>=d->n)return 0;uint8_t op=d->code[v.pc++],a,b;uint16_t q;uint32_t t;
      switch(op){case OP_RET: if(v.rp==0)return 1; if(!popr(&v,&t))return 0;v.pc=t;break;
      case OP_LIT: if(v.pc>=d->n||!push(&v,d->code[v.pc++]))return 0;break;
      case OP_ADD: if(!popv(&v,&b)||!popv(&v,&a)||!push(&v,(uint8_t)(a+b)))return 0;break;
      case OP_SUB: if(!popv(&v,&b)||!popv(&v,&a)||!push(&v,(uint8_t)(a-b)))return 0;break;
      case OP_DUP: if(!v.sp||!push(&v,v.st[v.sp-1]))return 0;break;
      case OP_DROP: if(!popv(&v,&a))return 0;break;
      case OP_SWAP: if(!popv(&v,&a)||!popv(&v,&b)||!push(&v,a)||!push(&v,b))return 0;break;
      case OP_LOAD: if(!popv(&v,&a)||!push(&v,v.mem[a]))return 0;break;
      case OP_STORE: if(!popv(&v,&a)||!popv(&v,&b))return 0;v.mem[a]=b;break;
      case OP_EMIT: if(!popv(&v,&a)||putchar(a)==EOF)return 0;break;
      case OP_NEXT: { char z[MAX_TOK]; int rc=next_tok(&io,z,sizeof(z)); if(rc<0)return 0; push(&v,(uint8_t)(rc==0)); break; }
      case OP_NEXTC: { uint8_t c=0; if(io.pos<io.len)c=io.src[io.pos++]; if(!push(&v,c))return 0; break; }
      case OP_SRC_POS: if(!push(&v,(uint8_t)(io.pos&0xff))||!push(&v,(uint8_t)((io.pos>>8)&0xff))||!push(&v,(uint8_t)((io.pos>>16)&0xff))||!push(&v,(uint8_t)((io.pos>>24)&0xff)))return 0; break;
      case OP_SRC_LEN: if(!push(&v,(uint8_t)(io.len&0xff))||!push(&v,(uint8_t)((io.len>>8)&0xff))||!push(&v,(uint8_t)((io.len>>16)&0xff))||!push(&v,(uint8_t)((io.len>>24)&0xff)))return 0; break;
      case OP_WORD_COUNT: if(!push(&v,(uint8_t)(d->nw&0xff))||!push(&v,(uint8_t)((d->nw>>8)&0xff)))return 0; break;
      case OP_WORD_NAME_LEN: if(!popv(&v,&a)||a>=d->nw||!push(&v,(uint8_t)strlen(d->words[a].name)))return 0; break;
      case OP_WORD_NAME_CHAR: { uint8_t off,hi,lo; if(!popv(&v,&off)||!popv(&v,&hi)||!popv(&v,&lo))return 0; uint16_t wid=(uint16_t)lo|((uint16_t)hi<<8); if(wid>=d->nw)return 0; size_t ln=strlen(d->words[wid].name); if(!push(&v,(uint8_t)(off<ln?d->words[wid].name[off]:0)))return 0; break; }
      case OP_WORD_CODE_LEN: if(!popv(&v,&a)||a>=d->nw)return 0; if(!push(&v,(uint8_t)(d->words[a].len&0xff))||!push(&v,(uint8_t)((d->words[a].len>>8)&0xff)))return 0; break;
      case OP_WORD_CODE_BYTE: { uint8_t off,hi,lo; if(!popv(&v,&off)||!popv(&v,&hi)||!popv(&v,&lo))return 0; uint16_t wid=(uint16_t)lo|((uint16_t)hi<<8); if(wid>=d->nw)return 0; uint32_t base=d->words[wid].off; size_t ln=d->words[wid].len; if(!push(&v,(uint8_t)(off<ln?d->code[base+off]:0)))return 0; break; }
      case OP_WORD_NEW: { uint8_t lhi,llo,ahi,alo; if(!popv(&v,&lhi)||!popv(&v,&llo)||!popv(&v,&ahi)||!popv(&v,&alo))return 0; uint16_t addr=(uint16_t)alo|((uint16_t)ahi<<8); uint16_t len=(uint16_t)llo|((uint16_t)lhi<<8); if((size_t)addr+len>sizeof(v.mem))return 0; int id=create_word_from_mem(d,v.mem,addr,len); if(id<0)return 0; if(!push(&v,(uint8_t)id)||!push(&v,(uint8_t)(id>>8)))return 0; break; }
      case OP_WORD_APPEND: { uint8_t b,hi,lo; if(!popv(&v,&b)||!popv(&v,&hi)||!popv(&v,&lo))return 0; uint16_t id=(uint16_t)lo|((uint16_t)hi<<8); if(!append_word_byte(d,id,b))return 0; if(!push(&v,lo)||!push(&v,hi))return 0; break; }
      case OP_WORD_EXEC: { uint8_t hi,lo; if(!popv(&v,&hi)||!popv(&v,&lo))return 0; uint16_t id=(uint16_t)lo|((uint16_t)hi<<8); if(id>=d->nw||v.rp>=MAX_RSTACK)return 0; v.ret[v.rp++]=v.pc; v.pc=d->words[id].off; break; }
      case OP_EQ: if(!popv(&v,&b)||!popv(&v,&a)||!push(&v,(uint8_t)(a==b)))return 0;break;
      case OP_JZ: if(v.pc+1>=d->n||!popv(&v,&a))return 0;q=(uint16_t)d->code[v.pc]|((uint16_t)d->code[v.pc+1]<<8);v.pc+=2;if(a==0)v.pc=q;break;
      case OP_JMP: if(v.pc+1>=d->n)return 0;q=(uint16_t)d->code[v.pc]|((uint16_t)d->code[v.pc+1]<<8);v.pc=q;break;
      case OP_CALL: if(v.pc+1>=d->n||v.rp>=MAX_RSTACK)return 0;q=(uint16_t)d->code[v.pc]|((uint16_t)d->code[v.pc+1]<<8);v.pc+=2;v.ret[v.rp++]=v.pc;if(q>=d->nw)return 0;v.pc=d->words[q].off;break;
      default:return 0;}
    }
}

int main(int argc,char**argv){
  if(argc<2||argc>3){fprintf(stderr,"usage: nucleus SOURCE [WORD]\n");return 64;}
  FILE*f=fopen(argv[1],"rb");if(!f)return 65;fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);if(n<0||n>(1<<20)){fclose(f);return 66;}uint8_t*src=malloc((size_t)n+1);if(!src){fclose(f);return 67;}if(fread(src,1,(size_t)n,f)!=(size_t)n){fclose(f);free(src);return 68;}fclose(f);
  dict_t d; if(!build_dict(src,(size_t)n,&d)){fprintf(stderr,"bootstrap parse/compile failure\n");free(src);return 69;}
  if(argc==3){int id=find_word(&d,argv[2]);if(id<0){free(src);return 70;} if(!run_word(&d,id,src,(size_t)n)){free(src);return 71;}}
  free(src);return 0;
}
