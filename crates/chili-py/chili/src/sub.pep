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

// Subscribe to one topic with a per-handle row filter.
// Replay is unfiltered; live broadcasts are filtered. State persists for .sub.recover.
.sub.initFiltered: {[tickSocket; topic; column; values]
  .sub.topics: (enlist topic);
  .sub.filterTopic: topic;
  .sub.filterColumn: column;
  .sub.filterValues: values;
  h: .handle.open tickSocket;
  .handle.onDisconnected[h; `.sub.recover];
  info: h (`.tick.subscribeFiltered; topic; column; values);
  (set) each info[2];
  replay[info[0]; 0; info[1]; (); 1b; h];
  .handle.subscribing[h];
};

// this function will be called when the connection is lost, retry every minute until no error
.sub.recover: {[handle]
  .handle.connect[handle];
  info: $[null get[`.sub.filterTopic];
    handle (`.tick.subscribe; .sub.topics);
    handle (`.tick.subscribeFiltered; .sub.filterTopic; .sub.filterColumn; .sub.filterValues)];
  replay[info[0]; tick[0]; info[1]; (); 1b; handle];
  .handle.subscribing[handle];
};
