h: .handle.open `chili://:18000;

pub: {[]
  n: 10;
  .log.info ("publish"; n ; "record");
  h (`.tick.upd; `trade; ([]sym: n?`a`b`c; price: n?1.0; size: n?10; side: n?`B`S; time: n#now[`]));
  h (`.tick.upd; `quote; ([]sym: n?`a`b`c; bid: n?1.0; ask: n?1.0; time: n#now[`]));
 };

.job.add["pub"; now[`]; now[`]+1D00:00:00; 0D00:00:01; "publish"];
