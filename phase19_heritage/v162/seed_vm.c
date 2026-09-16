#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAGIC "M7BF1"
#define MAGIC_LEN 5
#define MAX_CODE (1u<<20)
#define MAX_INPUT (16u<<20)
#define MAX_STACK 65536
#define MAX_CALL 8192
#define MEM_SIZE 256
#define MAX_STEPS 5000000ull

typedef struct { uint8_t *code; uint32_t code_len; uint8_t *in; uint32_t in_len; } frame_t;
typedef struct { uint8_t st[MAX_STACK]; size_t sp; uint32_t call[MAX_CALL]; size_t cp; uint8_t mem[MEM_SIZE]; uint32_t pc; uint32_t ip; uint64_t steps; } vm_t;
static int read_u32(FILE *f,uint32_t *v){uint8_t b[4];if(fread(b,1,4,f)!=4)return 0;*v=(uint32_t)b[0]|((uint32_t)b[1]<<8)|((uint32_t)b[2]<<16)|((uint32_t)b[3]<<24);return 1;}
static int load_frame(const char *path,frame_t *fr){FILE *f=fopen(path,"rb");if(!f)return 2;char magic[MAGIC_LEN];if(fread(magic,1,MAGIC_LEN,f)!=MAGIC_LEN||memcmp(magic,MAGIC,MAGIC_LEN)!=0){fclose(f);return 3;}uint32_t cl,il;if(!read_u32(f,&cl)||!read_u32(f,&il)||cl>MAX_CODE||il>MAX_INPUT){fclose(f);return 4;}fr->code=(uint8_t*)malloc(cl?cl:1);fr->in=(uint8_t*)malloc(il?il:1);if(!fr->code||!fr->in){fclose(f);return 5;}fr->code_len=cl;fr->in_len=il;if((cl&&fread(fr->code,1,cl,f)!=cl)||(il&&fread(fr->in,1,il,f)!=il)){fclose(f);return 6;}if(fgetc(f)!=EOF){fclose(f);return 7;}fclose(f);return 0;}
static int push(vm_t *v,uint8_t x){if(v->sp>=MAX_STACK)return 10;v->st[v->sp++]=x;return 0;} static int popv(vm_t *v,uint8_t *x){if(v->sp==0)return 11;*x=v->st[--v->sp];return 0;} static int need(vm_t *v,size_t n){return v->sp<n?11:0;}
static int run(frame_t *f){vm_t v;memset(&v,0,sizeof(v));for(;;){if(v.steps++>=MAX_STEPS)return 20;if(v.pc>=f->code_len)return 21;uint8_t op=f->code[v.pc++],a,b;uint32_t t;switch(op){case 0:return 0;case 1:if(v.pc>=f->code_len)return 22;if(push(&v,f->code[v.pc++]))return 10;break;case 2:if(popv(&v,&a))return 11;break;case 3:if(need(&v,2))return 11;popv(&v,&b);popv(&v,&a);push(&v,(uint8_t)(a+b));break;case 4:if(need(&v,2))return 11;popv(&v,&b);popv(&v,&a);push(&v,(uint8_t)(a-b));break;case 5:if(popv(&v,&a))return 11;push(&v,v.mem[a]);break;case 6:if(need(&v,2))return 11;popv(&v,&a);popv(&v,&b);v.mem[a]=b;break;case 7:if(v.pc>=f->code_len)return 22;v.pc=f->code[v.pc];break;case 8:if(need(&v,1)||v.pc>=f->code_len)return 11;popv(&v,&a);t=f->code[v.pc++];if(a==0)v.pc=t;break;case 9:if(v.pc>=f->code_len)return 22;if(v.cp>=MAX_CALL)return 23;v.call[v.cp++]=v.pc+1;v.pc=f->code[v.pc];break;case 10:if(v.cp==0)return 24;v.pc=v.call[--v.cp];break;case 11:if(v.ip>=f->in_len)return 25;if(push(&v,f->in[v.ip++]))return 10;break;case 12:if(popv(&v,&a))return 11;if(putchar(a)==EOF)return 26;break;case 13:if(need(&v,1))return 11;if(push(&v,v.st[v.sp-1]))return 10;break;case 14:if(need(&v,2))return 11;popv(&v,&b);popv(&v,&a);push(&v,(uint8_t)(a==b));break;case 15:if(need(&v,2))return 11;popv(&v,&b);popv(&v,&a);push(&v,(uint8_t)(a<b));break;case 16:if(need(&v,2))return 11;popv(&v,&b);popv(&v,&a);push(&v,(uint8_t)(a&b));break;case 17:if(need(&v,2))return 11;popv(&v,&b);popv(&v,&a);push(&v,(uint8_t)(a|b));break;case 18:if(need(&v,2))return 11;popv(&v,&b);popv(&v,&a);push(&v,(uint8_t)(a^b));break;case 19:if(popv(&v,&a))return 11;break;default:return 27;}}}
int main(int argc,char **argv){if(argc!=2){fprintf(stderr,"usage: seed_vm FRAME\n");return 64;}frame_t f={0};int e=load_frame(argv[1],&f);if(e){fprintf(stderr,"frame error %d\n",e);return e;}int r=run(&f);free(f.code);free(f.in);return r;}
