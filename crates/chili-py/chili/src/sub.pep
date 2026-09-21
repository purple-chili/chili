upd: {[table; data] table upsert data; tick[this.h; 1]; };

// ---------------------------------------------------------------------------
// Per-handle subscription state, kept in a variable named after the handle:
//   (topics; filterColumn; filterValues; logPath; full)
// topics is always a list; filterColumn is null for an unfiltered
// subscription; full is 1b when the handle receives every table unfiltered.
// One process may hold several subscriber handles (engine.subscribe opens one
// per filtered topic plus one for the unfiltered topics), so recover must not
// read process-wide globals.
// ---------------------------------------------------------------------------
.sub.stateName: {[h] ".sub.state." + (`str$h)};

.sub.request: {[h; st]
  $[null st[1];
    h (`.tick.subscribe; st[0]);
    h (`.tick.subscribeFiltered; first st[0]; st[1]; st[2])]
};

// Reset one table to its empty schema.
.sub.resetTable: {[schema; topic] set[topic; schema[topic]]};

// Replay the rest of a log this handle was already following, then restart
// the handle's message count for the next log.
.sub.finishLog: {[h; path]
  $[exists path; replay[path; tick[h; 0]; 9223372036854775807; (); 1b; h]; 0];
  tock[h; 0];
};

.sub.init: {[tickSocket; topics]
  .sub.topics: topics;
  h: .handle.open tickSocket;
  .handle.onDisconnected[h; `.sub.recover];
  info: h (`.tick.subscribe; topics);
  (set) each info[2];
  full: (0 = count topics) | (count topics) = count key info[2];
  set[.sub.stateName[h]; (topics; `; (); info[0]; full)];
  // Read live frames into a local hold buffer while replaying up to the
  // bound, so the backlog of a slow replay sits here rather than in the
  // tickerplant's outbound queue. .handle.release applies it in order.
  .handle.holding[h];
  replay[info[0]; 0; info[1]; (); 1b; h];
  .handle.release[h];
};

// Subscribe to one topic with a per-handle row filter.
// Replay is unfiltered; live broadcasts are filtered.
.sub.initFiltered: {[tickSocket; topic; column; values]
  .sub.topics: (enlist topic);
  .sub.filterTopic: topic;
  .sub.filterColumn: column;
  .sub.filterValues: values;
  h: .handle.open tickSocket;
  .handle.onDisconnected[h; `.sub.recover];
  info: h (`.tick.subscribeFiltered; topic; column; values);
  (set) each info[2];
  set[.sub.stateName[h]; (enlist topic; column; values; info[0]; 0b)];
  .handle.holding[h];
  replay[info[0]; 0; info[1]; (); 1b; h];
  .handle.release[h];
};

// Called when the connection is lost; retried with backoff until it succeeds.
//
// A handle that receives every table unfiltered sees every logged message, so
// tick[handle] (advanced by upd, for replayed and live frames alike) is its
// position in the log: recover continues from there, and when the tickerplant
// moved to a new log meanwhile it first finishes the old one.
//
// Any other handle sees only part of the log, so its position is unknown:
// its own tables are reset and replayed from the start of the current log.
.sub.recover: {[handle]
  .handle.connect[handle];
  name: .sub.stateName[handle];
  st: get name;
  info: .sub.request[handle; st];
  .handle.holding[handle];
  $[st[4];
    $[info[0] = st[3]; 0; .sub.finishLog[handle; st[3]]];
    .sub.resetTable[info[2]; ] each st[0]];
  start: $[st[4]; tick[handle; 0]; 0];
  replay[info[0]; start; info[1]; $[st[4]; (); st[0]]; 1b; handle];
  set[name; (st[0]; st[1]; st[2]; info[0]; st[4])];
  .handle.release[handle];
};
