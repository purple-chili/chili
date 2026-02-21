upd: {[table; data] table upsert data; tick[1]; };

.sub.init: {[]
  h: .handle.open `chili://:18000:sub1:token789;
  .handle.onDisconnected[h; `.sub.recover];
  info: h (`.tick.subscribe; ());
  (set) each info[2];
  // .log.info ("broker info"; info);
  replay[info[0]; 0; info[1]; (); 1b];
  .handle.subscribing[h];
};

// this function will be called when the connection is lost, retry every minute until no error
.sub.recover: {[handle]
  .handle.connect[handle];
  info: handle (`.tick.subscribe; ());
  replay[info[0]; tick[0]; info[1]; (); 1b];
  .handle.subscribing[handle];
};

.sub.init[];
