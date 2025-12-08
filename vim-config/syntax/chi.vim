" Vim syntax file
" Language:     chi

if !exists("main_syntax")
  if version < 600
    syntax clear
  elseif exists("b:current_syntax")
    finish
  endif
  let main_syntax = 'chi'
endif


syn keyword chiliCommentDoc         contained param return
syn region  chiliComment            start=+//+ end=/$/ contains=chiliCommentDoc,@Spell extend keepend
syn region  chiliComment            start=+/\*+  end=+\*/+ contains=chiliCommentDoc,@Spell fold extend keepend

syn match   chiliSpecial            contained "\v\\%(x\x\x|u%(\x{4}|\{\x{4,5}})|c\u|.)"
syn region  chiliString             start=+\z(["']\)+  skip=+\\\%(\z1\|$\)+  end=+\z1+ end=+$+  contains=chiliSpecial extend

syn match   chiliSymbol             "\v`([a-zA-Z0-9_.:/_\-]*)"

syn match   chiliNull               "\v<0n>"
syn match   chiliBoolean            "\v<[01]+b>"
syn match   chiliInfinity           "\v<-?0w(f32|e)>"

syn match   Number                  "\v<0x[0-9a-fA-F]+>"
syn match   Float                   "\v<[0-9]+\.[0-9]*(e[+-]=[0-9]+)=>"
syn match   Float                   "\v\.[0-9]+(e[+-]=[0-9]+)="
syn match   Float                   "\v<[0-9]+e[+-]=[0-9]+>"
syn match   Number                  "\v<[0-9]+([efhij]|u8|u16|u32|u64|i8|i16|i32|i64|i128)=>"

syn match   chiliIdentifier         "\v\.[a-zA-Z][a-zA-Z0-9_]*(\.[a-zA-Z0-9_]+)*"

" temporal types
syn match   chiliDate               "\v<([12]\d{3}\.(0[1-9]|1[0-2])\.(0[1-9]|[12]\d|3[01]))>"
syn match   chiliTime               "\v<([01]\d|2[0-3])(:([0-5]\d)(:([0-5]\d(\.\d{0,3})?))?)>"
syn match   chiliTimespan           "\v<\d+D(([01]\d|2[0-3])(:([0-5]\d)(:([0-5]\d(\.\d{0,9})?))?)?)?>"
syn match   chiliTimestamp          "\v<([12]\d{3}\.(0[1-9]|1[0-2])\.(0[1-9]|[12]\d|3[01]))D>"
syn match   chiliTimestamp          "\v<([12]\d{3}\.(0[1-9]|1[0-2])\.(0[1-9]|[12]\d|3[01]))D([01]\d|2[0-3])(:([0-5]\d)(:([0-5]\d(\.\d{0,9})?))?)?>"
syn match   chiliDatetime           "\v<([12]\d{3}\.(0[1-9]|1[0-2])\.(0[1-9]|[12]\d|3[01]))T>"
syn match   chiliDatetime           "\v<([12]\d{3}\.(0[1-9]|1[0-2])\.(0[1-9]|[12]\d|3[01]))T([01]\d|2[0-3])(:([0-5]\d)(:([0-5]\d(\.\d{0,3})?))?)?>"

syn keyword chiliConditional        do while if function return raise try catch
" use list() to output all built-in functions
syn keyword chiliKeyword            abs acos acosh aj all anti args as asc asin asinh assert atan atanh bfill bottom cbrt ccount ceil cj clip cmax cmin col cols concat console corr cos cosh cot count cov0 cov1 cprod cross csum del desc describe diff differ div each emean enlist equal estd eval evalc evali evar exists exit exp explode extend fail fby fill filter first fj flag flatten flip floor get hash hdel hstack ij import in interp intersect inv join key kurtosis last like list lit lj ln load local log log10 log1p lowercase matches max mean median min mmax mmean mmedian mmin mod mode mquantile mskew mstd0 mstd1 msum mvar0 mvar1 neg next not now null over pad par parallel pc pivot pow prev prod quantile range rank rbin rcsv rdatabase replace replay replay_q reshape reverse rexcel rjson rotate round rparquet rtxt scan schema semi set shift show shuffle sign sin sinh skew split sqrt ss ssr std0 std1 sub_q sum tables tan tanh tick timeit today top transpose trim trime trims type tz uc union unique unpivot uppercase upsert utc var0 var1 vstack wbin wcsv wdatabase wexcel when within wj wjson wmean wpar wparquet wsum wtxt xasc xbar xdesc xrename xreorder
syn keyword chiliDML                select update delete from by fby where aj ej ij lj uj wj upsert

" Define the default highlighting.
" Only when an item doesn't have highlighting yet
hi def link chiliComment                Comment
hi def link chiliCommentDoc             Keyword
hi def link chiliSpecial                Special
hi def link chiliString                 String
hi def link chiliKeyword                Keyword
hi def link chiliConditional            Keyword
hi def link chiliDML                    Keyword
hi def link chiliBoolean                Boolean
hi def link chiliSymbol                 Constant
hi def link chiliNull                   Constant
hi def link chiliInfinity               Constant
hi def link chiliDate                   Constant
hi def link chiliMonth                  Constant
hi def link chiliTime                   Constant
hi def link chiliTimespan               Constant
hi def link chiliTimestamp              Constant
hi def link chiliDatetime               Constant
hi def link chiliCommand                Constant
hi def link chiliIdentifier             Function

let b:current_syntax = "chi"
if main_syntax == 'chi'
  unlet main_syntax
endif
