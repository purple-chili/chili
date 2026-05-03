upd: {[table; data] table upsert data; tick[this.h; 1]; };

.sub.init: {[tickSocket; topics]
  .sub.topics: topics;
  h: .handle.open tickSocket;
  .handle.onDisconnected[h; `.sub.recover];
  info: h (`.tick.subscribe; topics);
  (set) each info[2];
  // .log.info ("broker info"; info);
  replay[info[0]; 0; info[1]; (); 1b; h];
  .handle.subscribing[h];
};

// this function will be called when the connection is lost, retry every minute until no error
.sub.recover: {[handle]
  .handle.connect[handle];
  info: handle (`.tick.subscribe; .sub.topics);
  replay[info[0]; tick[0]; info[1]; (); 1b; handle];
  .handle.subscribing[handle];
};
