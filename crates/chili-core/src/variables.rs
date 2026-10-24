use std::{
    collections::{HashMap, HashSet},
    ops::Deref,
    sync::{Arc, Weak},
};

use parking_lot::RwLock;
use polars::prelude::*;

use crate::{SpicyError, SpicyObj, SpicyResult, ensure_df_rectangular};

/// All non-append mutations invalidate an outstanding compaction. Keeping the
/// invalidation in this wrapper prevents new variable writers from forgetting it.
pub(crate) struct Variables {
    values: HashMap<String, SpicyObj>,
    compactions: HashMap<String, Weak<()>>,
    pending_compaction: HashSet<String>,
}

pub(crate) struct Compaction {
    lease: Arc<()>,
    pub(crate) rows: usize,
}

impl From<HashMap<String, SpicyObj>> for Variables {
    fn from(values: HashMap<String, SpicyObj>) -> Self {
        Self {
            values,
            compactions: HashMap::new(),
            pending_compaction: HashSet::new(),
        }
    }
}

// Deliberately no DerefMut: mutations must go through the invalidating methods.
impl Deref for Variables {
    type Target = HashMap<String, SpicyObj>;
    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl Variables {
    pub(crate) fn insert(&mut self, id: String, value: SpicyObj) -> Option<SpicyObj> {
        self.compactions.remove(&id);
        self.pending_compaction.remove(&id);
        self.values.insert(id, value)
    }

    pub(crate) fn remove(&mut self, id: &str) -> Option<SpicyObj> {
        self.compactions.remove(id);
        self.pending_compaction.remove(id);
        self.values.remove(id)
    }

    pub(crate) fn get_mut(&mut self, id: &str) -> Option<&mut SpicyObj> {
        self.compactions.remove(id);
        self.pending_compaction.remove(id);
        self.values.get_mut(id)
    }

    /// Only symbol vstack may use this: existing rows and schema must survive
    /// unchanged, including when an append fails and rolls back.
    pub(crate) fn get_mut_for_append(&mut self, id: &str) -> Option<&mut SpicyObj> {
        self.values.get_mut(id)
    }

    pub(crate) fn mark_appended(&mut self, id: &str) {
        if !self.pending_compaction.contains(id) {
            self.pending_compaction.insert(id.into());
        }
    }

    pub(crate) fn pending_compactions(&self) -> Vec<String> {
        self.pending_compaction.iter().cloned().collect()
    }

    pub(crate) fn begin_pending_compaction(
        &mut self,
        id: &str,
    ) -> SpicyResult<Option<(DataFrame, Compaction)>> {
        if !self.pending_compaction.contains(id)
            || self.compactions.get(id).and_then(Weak::upgrade).is_some()
        {
            return Ok(None);
        }
        let snapshot = self.begin_compaction(id)?;
        self.pending_compaction.remove(id);
        Ok(Some(snapshot))
    }

    pub(crate) fn begin_compaction(&mut self, id: &str) -> SpicyResult<(DataFrame, Compaction)> {
        let df = self
            .values
            .get(id)
            .ok_or_else(|| SpicyError::NameErr(id.into()))?
            .df()?;
        ensure_df_rectangular(df)?;
        if self.compactions.get(id).and_then(Weak::upgrade).is_some() {
            return Err(SpicyError::Err(format!(
                "compaction already running for '{id}'"
            )));
        }
        let snapshot = df.clone();
        let token = Compaction {
            lease: Arc::new(()),
            rows: df.height(),
        };
        self.compactions
            .insert(id.into(), Arc::downgrade(&token.lease));
        Ok((snapshot, token))
    }

    /// Publish only if the original prefix is still valid. New vstack rows are
    /// retained in order. Return the old frame so its destruction happens after
    /// releasing the variable lock. On error, compacted is also dropped outside.
    pub(crate) fn finish_compaction(
        &mut self,
        id: &str,
        token: &Compaction,
        compacted: &mut DataFrame,
    ) -> SpicyResult<DataFrame> {
        if !self
            .compactions
            .get(id)
            .and_then(Weak::upgrade)
            .is_some_and(|lease| Arc::ptr_eq(&lease, &token.lease))
        {
            return Err(SpicyError::Err(format!(
                "compaction cancelled: '{id}' changed; retry on the current table"
            )));
        }
        let current = self
            .values
            .get_mut(id)
            .expect("live compaction has a table")
            .mut_df()?;
        ensure_df_rectangular(current)?;
        if current.height() < token.rows || compacted.height() != token.rows {
            return Err(SpicyError::Err("compaction prefix height changed".into()));
        }
        if current.height() > token.rows {
            let suffix = current.slice(token.rows as i64, current.height() - token.rows);
            compacted
                .vstack_mut(&suffix)
                .map_err(|e| SpicyError::Err(e.to_string()))?;
        }
        let retired = std::mem::replace(current, std::mem::take(compacted));
        self.compactions.remove(id);
        Ok(retired)
    }
}

pub(crate) fn compact_and_publish(
    vars: &RwLock<Variables>,
    id: &str,
    snapshot: DataFrame,
    token: Compaction,
    target_rows: usize,
) -> SpicyResult<usize> {
    let mut compacted = compact_dataframe(&snapshot, target_rows)?;
    let retired = vars.write().finish_compaction(id, &token, &mut compacted)?;
    // Retired descriptors, the original snapshot, and failed results must all
    // be destroyed after releasing the variable lock.
    drop(retired);
    Ok(token.rows)
}

/// Coalesce small adjacent chunks independently per column. Already-large
/// chunks keep their buffers, so a large seeded prefix is not copied each time
/// a small append tail is compacted. Chunk boundaries need not align by column.
pub fn compact_dataframe(df: &DataFrame, target_rows: usize) -> SpicyResult<DataFrame> {
    if target_rows == 0 {
        return Err(SpicyError::Err(
            "compact target_rows must be positive".into(),
        ));
    }
    ensure_df_rectangular(df)?;
    let columns = df
        .columns()
        .iter()
        .map(|column| {
            let series = column.as_materialized_series();
            if series.n_chunks() <= 1 {
                return Ok(column.clone());
            }
            let mut output = series.clear();
            let mut pending: Option<Series> = None;
            let flush = |pending: &mut Option<Series>, output: &mut Series| -> PolarsResult<()> {
                if let Some(group) = pending.take() {
                    output.append_owned(group.rechunk())?;
                }
                Ok(())
            };
            for (i, chunk) in series.chunks().iter().enumerate() {
                if chunk.is_empty() {
                    continue;
                }
                let chunk_series = series.select_chunk(i);
                if chunk.len() >= target_rows {
                    flush(&mut pending, &mut output)?;
                    output.append_owned(chunk_series)?;
                } else {
                    match &mut pending {
                        Some(group) => {
                            group.append_owned(chunk_series)?;
                        }
                        None => pending = Some(chunk_series),
                    }
                    if pending
                        .as_ref()
                        .is_some_and(|group| group.len() >= target_rows)
                    {
                        flush(&mut pending, &mut output)?;
                    }
                }
            }
            flush(&mut pending, &mut output)?;
            Ok(output.into_column())
        })
        .collect::<PolarsResult<Vec<_>>>()
        .map_err(|e| SpicyError::Err(e.to_string()))?;
    DataFrame::new(df.height(), columns).map_err(|e| SpicyError::Err(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vars() -> Variables {
        let mut df = df!("seq" => [1i64, 2]).unwrap();
        df.vstack_mut(&df!("seq" => [3i64, 4]).unwrap()).unwrap();
        HashMap::from([("buf".into(), SpicyObj::DataFrame(df))]).into()
    }

    #[test]
    fn publish_preserves_appends_after_snapshot_and_old_snapshot() {
        let mut vars = vars();
        let (snapshot, token) = vars.begin_compaction("buf").unwrap();
        let mut compacted = compact_dataframe(&snapshot, 64).unwrap();
        vars.get_mut_for_append("buf")
            .unwrap()
            .mut_df()
            .unwrap()
            .vstack_mut(&df!("seq" => [5i64, 6]).unwrap())
            .unwrap();
        let retired = vars
            .finish_compaction("buf", &token, &mut compacted)
            .unwrap();
        assert_eq!(retired.height(), 6);
        assert_eq!(snapshot, df!("seq" => [1i64, 2, 3, 4]).unwrap());
        assert_eq!(
            vars.get("buf").unwrap().df().unwrap(),
            &df!("seq" => [1i64, 2, 3, 4, 5, 6]).unwrap()
        );
    }

    #[test]
    fn replacement_removal_and_mutable_access_invalidate_publication() {
        for mutation in 0..3 {
            let mut vars = vars();
            let (snapshot, token) = vars.begin_compaction("buf").unwrap();
            let mut compacted = compact_dataframe(&snapshot, 64).unwrap();
            let replacement = SpicyObj::DataFrame(df!("seq" => [90i64, 91, 92, 93]).unwrap());
            match mutation {
                0 => {
                    vars.insert("buf".into(), replacement.clone());
                }
                1 => {
                    vars.remove("buf");
                    vars.insert("buf".into(), replacement.clone());
                }
                _ => {
                    *vars.get_mut("buf").unwrap() = replacement.clone();
                }
            }
            assert!(
                vars.finish_compaction("buf", &token, &mut compacted)
                    .is_err()
            );
            assert_eq!(vars.get("buf"), Some(&replacement));
        }
    }

    #[test]
    fn failed_or_cancelled_jobs_do_not_block_or_publish_over_a_new_job() {
        let mut vars = vars();
        let (_, abandoned) = vars.begin_compaction("buf").unwrap();
        assert!(vars.begin_compaction("buf").is_err());
        drop(abandoned);
        let (old, stale) = vars.begin_compaction("buf").unwrap();
        vars.insert(
            "buf".into(),
            SpicyObj::DataFrame(df!("seq" => [99i64]).unwrap()),
        );
        let (new, live) = vars.begin_compaction("buf").unwrap();
        assert!(
            vars.finish_compaction("buf", &stale, &mut old.clone())
                .is_err()
        );
        vars.finish_compaction("buf", &live, &mut new.clone())
            .unwrap();
        assert_eq!(vars.get("buf").unwrap().df().unwrap(), &new);
    }

    #[test]
    fn background_queue_tracks_appends_and_discards_invalidated_tables() {
        let mut vars = vars();
        vars.mark_appended("buf");
        let (_, manual) = vars.begin_compaction("buf").unwrap();
        assert!(vars.begin_pending_compaction("buf").unwrap().is_none());
        drop(manual);
        let (_, pending) = vars.begin_pending_compaction("buf").unwrap().unwrap();
        assert!(vars.pending_compactions().is_empty());
        vars.mark_appended("buf");
        assert_eq!(vars.pending_compactions(), ["buf"]);
        drop(pending);
        vars.get_mut("buf");
        assert!(vars.begin_pending_compaction("buf").unwrap().is_none());
        vars.mark_appended("buf");
        vars.remove("buf");
        assert!(vars.begin_pending_compaction("buf").unwrap().is_none());
    }
}
