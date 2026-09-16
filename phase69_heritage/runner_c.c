#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* Minimal native replay runner for the V69 bounded artifact contract. */
typedef struct {int c,s,nc,w,m;} T;
static long get_num(const char *s,const char *key){const char *p=strstr(s,key); if(!p) return -1; p=strchr(p,':'); return p?strtol(p+1,NULL,10):-1;}
int main(void){
 char *buf=NULL; size_t cap=0,len=0; int ch; while((ch=fgetc(stdin))!=EOF){if(len+1>=cap){cap=cap?cap*2:4096;buf=realloc(buf,cap);}buf[len++]=(char)ch;} if(!buf)return 2;buf[len]=0;
 const char *ap=strstr(buf,"\"artifact\""); if(!ap)return 3; const char *mp=strstr(ap,"\"machine\""); if(!mp)return 4;
 long start=get_num(mp,"\"start_control\""), halt=get_num(mp,"\"halt_control\"");
 const char *tp=strstr(mp,"\"transitions\""); if(!tp)return 5;
 const char *tap=strstr(buf,"\"tape\":"); if(!tap)return 6; tap=strchr(tap,'['); if(!tap)return 7; const char *te=strchr(tap,']'); if(!te)return 8;
 size_t cnt=1; for(const char *q=tap;q<te;q++)if(*q==',')cnt++; long *tape=calloc(cnt,sizeof(long)); size_t ti=0; const char *q=tap+1; while(q<te&&ti<cnt){char *e;tape[ti++]=strtol(q,&e,10);if(e==q)break;q=e;while(q<te&&(*q==' '||*q==','))q++;}
 T *tr=NULL; size_t nt=0; const char *p=strchr(tp,'['); if(!p)return 9; const char *end=strchr(p,']'); if(!end)return 10;
 while((p=strstr(p,"\"control\""))&&p<end){long c=get_num(p,"\"control\""),s=get_num(p,"\"symbol\""),nc=get_num(p,"\"next_control\""),w=get_num(p,"\"write_symbol\""),m=get_num(p,"\"move\""); if(c<0||s<0||nc<0||w<0||m<-1)return 11;tr=realloc(tr,(nt+1)*sizeof(T));tr[nt++]=(T){(int)c,(int)s,(int)nc,(int)w,(int)m};p+=8;} if(!nt)return 12;
 int state=(int)start,head=0; for(long step=0;step<4096;step++){if(state==(int)halt)break;if(head<0||(size_t)head>=cnt)return 13;T *hit=NULL;for(size_t i=0;i<nt;i++)if(tr[i].c==state&&tr[i].s==(int)tape[head]){hit=&tr[i];break;}if(!hit)return 14;tape[head]=hit->w;head+=hit->m;state=hit->nc;} if(state!=(int)halt)return 15;
 putchar('[');for(size_t i=0;i<cnt;i++){if(i)putchar(',');printf("%ld",tape[i]);}puts("]");free(tape);free(tr);free(buf);return 0;
}
