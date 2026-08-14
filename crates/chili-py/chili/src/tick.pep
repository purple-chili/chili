.tick.schema: {};

.tick.createLog: {[logDir; filename]
  .tick.msgLog: logDir + filename;
  // Recovery: if the plain dated path is missing but <date>.gz exists
  // (hk / log-rotation layout), use the gzip archive.
  if[(not exists .tick.msgLog) & exists[.tick.msgLog + ".gz"];
    .tick.msgLog: .tick.msgLog + ".gz"
  ];
  .tick.logFile: "file://" + .tick.msgLog;
  // tick is using handle 0 for internal tick count (message count from validateSeq).
  // To set an absolute counter after init (e.g. a per-row high-water), use:
  //   tock[0; n]
  // or: tick[0; neg tick[0; 0]]; tick[0; n]
  tick[0; .broker.validateSeq[.tick.msgLog; 0b]];
  // close existing handle
  if[not null get[`.tick.msgHandle];.handle.close get[`.tick.msgHandle]];
  .tick.msgHandle: .handle.open .tick.logFile;
};

.tick.rollLog: {[logDir; filename]
  .tick.msgLog: logDir + filename;
  .tick.logFile: "file://" + .tick.msgLog;
  .handle.rotate[.tick.msgHandle; .tick.logFile];
  // reset tick[0]
  tick[0; neg[tick[0; 0]]];
  // use validate message count for tick[0]
  tick[0; tick[.tick.msgHandle; 0]];
};

.tick.upd: {[table; data]
  // lpt: log-write + publish + tick[tick_index; 1] under one lock
  lpt[table; data; 0]
};

.tick.subscribe: {[topics]
  topics: $[count topics; topics; key .tick.schema];
  // this is reserved for current stack
  // this.h is the handle for the IPC connection of current stack
  .broker.subscribe[this.h; ] each topics;
  (.tick.msgLog; tick[0; 0]; .tick.schema)
};

// Register a per-handle row filter for one topic.
.tick.subscribeFiltered: {[topic; column; values]
  .broker.subscribeFiltered[this.h; topic; column; values];
  (.tick.msgLog; tick[0; 0]; .tick.schema)
};

.tick.unsubscribe: {[topics]
  topics: $[count topics; topics; key .tick.schema];
  // this is reserved for current stack
  // this.h is the handle for the IPC connection of current stack
  .broker.unsubscribe[this.h; ] each topics;
};

.tick.eod: {[date] .broker.eod[(`eod; date)]; };
