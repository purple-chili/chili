upd: {[table; data] table upsert data; tick[this.h; 1]; };

.sub.init: {[tickSocket; topics]
  .sub.topics: topics;
  h: .handle.open tickSocket;
  .handle.onDisconnected[h; `.sub.recover];
  info: h (`.tick.subscribe; topics);
  (set) each info[2];
  // Read live frames into a local hold buffer while replaying up to the
  // bound, so the backlog of a slow replay sits here rather than in the
  // tickerplant's outbound queue. .handle.release applies it in order.
  .handle.holding[h];
  replay[info[0]; 0; info[1]; (); 1b; h];
  .handle.release[h];
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
  .handle.holding[h];
  replay[info[0]; 0; info[1]; (); 1b; h];
  .handle.release[h];
};

// this function will be called when the connection is lost, retry every minute until no error
.sub.recover: {[handle]
  .handle.connect[handle];
  info: $[null get[`.sub.filterTopic];
    handle (`.tick.subscribe; .sub.topics);
    handle (`.tick.subscribeFiltered; .sub.filterTopic; .sub.filterColumn; .sub.filterValues)];
  .handle.holding[handle];
  replay[info[0]; tick[0; 0]; info[1]; (); 1b; handle];
  .handle.release[handle];
};
