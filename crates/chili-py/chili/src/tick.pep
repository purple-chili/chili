.tick.schema: {};

.tick.createLog: {[logDir; date]
  .tick.msgLog: logDir + date;
  .tick.logFile: "file://" + .tick.msgLog;
  // tick is using handle 0 for internal tick count
  tick[0; .broker.validateSeq[.tick.msgLog; 0b]];
  // close existing handle
  if[not null get[`.tick.msgHandle];.handle.close get[`.tick.msgHandle]];
  .tick.msgHandle: .handle.open .tick.logFile;
};

.tick.upd: {[table; data]
  .tick.msgHandle (`upd; table; data);
  .broker.publish[`upd; table; data];
  // tick[0; 1] is a built-in function for updating internal tick count
  // use `tick[0; 0]` to get current tick count, `tick[0; neg tick[0; 0]]` to reset tick count.
  tick[0; 1];
};

.tick.subscribe: {[topics]
  topics: $[count topics; topics; key .tick.schema];
  // this is reserved for current stack
  // this.h is the handle for the IPC connection of current stack
  .broker.subscribe[this.h; ] each topics;
  (.tick.msgLog; tick[0; 0]; .tick.schema)
};

.tick.unsubscribe: {[topics]
  topics: $[count topics; topics; key .tick.schema];
  // this is reserved for current stack
  // this.h is the handle for the IPC connection of current stack
  .broker.unsubscribe[this.h; ] each topics;
};

.tick.eod: {[date] .broker.eod[(`eod; date)]; };
