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
  // Set, not add: a second createLog on the same engine must not stack the
  // new log's count on top of the old one.
  tock[0; .broker.validateSeq[.tick.msgLog; 0b]];
  // close existing handle
  if[not null get[`.tick.msgHandle];.handle.close get[`.tick.msgHandle]];
  .tick.msgHandle: .handle.open .tick.logFile;
};

.tick.rollLog: {[logDir; filename]
  .tick.msgLog: logDir + filename;
  .tick.logFile: "file://" + .tick.msgLog;
  // Rotate and set tick[0] to the new log's message count in one lpt_lock
  // section: a .tick.upd running between a rotation and a separate counter
  // reset left the replay bound one frame short for the rest of the log.
  // Rolling to the log that is already open changes nothing.
  .handle.rotateTick[.tick.msgHandle; .tick.logFile; 0];
};

.tick.upd: {[table; data]
  // lpt[table; data; 0; handle]: log + publish + tick[0; 1] under one lock.
  // lpt[table; data; `seq; handle]: stamp seq = tick[0; 0] + i, then tick[0; count data].
  lpt[table; data; 0; .tick.msgHandle]
};

.tick.subscribe: {[topics]
  topics: $[count topics; topics; key .tick.schema];
  // this.h is the handle for the IPC connection of current stack.
  // .broker.subscribe registers this.h pending and returns the replay bound
  // (tick[0; 0]) from the same lpt_lock section: every frame after the bound
  // is buffered for this.h and written right after this response.
  bound: .broker.subscribe[this.h; topics];
  (.tick.msgLog; bound; .tick.schema)
};

// Register a per-handle row filter for one topic.
.tick.subscribeFiltered: {[topic; column; values]
  bound: .broker.subscribeFiltered[this.h; topic; column; values];
  (.tick.msgLog; bound; .tick.schema)
};

.tick.unsubscribe: {[topics]
  topics: $[count topics; topics; key .tick.schema];
  // this is reserved for current stack
  // this.h is the handle for the IPC connection of current stack
  .broker.unsubscribe[this.h; ] each topics;
};

.tick.eod: {[date] .broker.eod[(`eod; date)]; };
