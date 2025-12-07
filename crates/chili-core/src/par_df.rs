use std::{fmt::Display, path::PathBuf};

use polars::prelude::{LazyFrame, PlPath, ScanArgsParquet};

use crate::{SpicyError, SpicyObj, SpicyResult};

#[derive(PartialEq, Debug, Clone)]
pub enum DFType {
    Single,
    ByDate,
    ByYear,
}

impl Display for DFType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                DFType::Single => "single",
                DFType::ByDate => "date",
                DFType::ByYear => "year",
            }
        )
    }
}

#[derive(PartialEq, Debug, Clone)]
pub struct PartitionedDataFrame {
    pub name: String,
    pub df_type: DFType,
    pub path: String,
    pub pars: Vec<i32>,
}

impl PartitionedDataFrame {
    pub fn new(name: String, df_type: DFType, path: String, pars: Vec<i32>) -> Self {
        Self {
            name,
            df_type,
            path,
            pars,
        }
    }

    pub fn get_par_glob(&self, par_num: i32) -> String {
        if self.df_type == DFType::ByDate {
            format!("{}_*", SpicyObj::Date(par_num))
        } else {
            format!("{}_*", par_num)
        }
    }

    pub fn get_empty_schema(&self) -> &str {
        "schema"
    }

    pub fn scan_partition(&self, par_num: i32) -> SpicyResult<LazyFrame> {
        if self.df_type == DFType::Single {
            return LazyFrame::scan_parquet(
                PlPath::Local(PathBuf::from(&self.path).as_path().into()),
                ScanArgsParquet::default(),
            )
            .map_err(|e| SpicyError::Err(format!("failed to scan single {}: {}", self.name, e)));
        }
        let mut par_path = PathBuf::from(&self.path);
        let args = ScanArgsParquet::default();
        let lazy_df = if self.pars.binary_search(&par_num).is_ok() {
            par_path.push(self.get_par_glob(par_num));
            LazyFrame::scan_parquet(PlPath::Local(par_path.as_path().into()), args).map_err(
                |e| SpicyError::Err(format!("failed to scan partitioned {}: {}", self.name, e)),
            )?
        } else {
            par_path.push(self.get_empty_schema());
            LazyFrame::scan_parquet(PlPath::Local(par_path.as_path().into()), args).map_err(
                |e| SpicyError::Err(format!("failed to scan partitioned {}: {}", self.name, e)),
            )?
        };
        Ok(lazy_df)
    }

    pub fn scan_partitions(&self, par_nums: &[i32]) -> SpicyResult<LazyFrame> {
        let mut pars: Vec<PathBuf> = Vec::new();
        let mut par_path = PathBuf::from(&self.path);
        let args = ScanArgsParquet::default();
        let mut par_nums = par_nums.to_vec();
        par_nums.sort();
        par_nums.dedup();
        for par_num in par_nums {
            if self.pars.binary_search(&par_num).is_ok() {
                let mut par_path = par_path.clone();
                par_path.push(self.get_par_glob(par_num));
                for entry in glob::glob(&par_path.to_string_lossy())
                    .map_err(|e| SpicyError::EvalErr(e.to_string()))?
                {
                    match entry {
                        Ok(path) => pars.push(path),
                        Err(e) => eprintln!("{:?}", e),
                    }
                }
            }
        }
        let lazy_df = if !pars.is_empty() {
            LazyFrame::scan_parquet_files(
                pars.into_iter()
                    .map(|p| PlPath::Local(p.as_path().into()))
                    .collect(),
                args,
            )
            .map_err(|e| SpicyError::EvalErr(e.to_string()))?
        } else {
            par_path.push(self.get_empty_schema());
            LazyFrame::scan_parquet(PlPath::Local(par_path.as_path().into()), args).map_err(
                |e| SpicyError::Err(format!("failed to scan partitioned {}: {}", self.name, e)),
            )?
        };
        Ok(lazy_df)
    }

    pub fn scan_partition_by_range(&self, start_par: i32, end_par: i32) -> SpicyResult<LazyFrame> {
        let mut par_path = PathBuf::from(&self.path);
        let args = ScanArgsParquet::default();
        let start_index = match self.pars.binary_search(&start_par) {
            Ok(i) => i,
            Err(i) => i,
        };
        let mut end_index = match self.pars.binary_search(&end_par) {
            Ok(i) => i + 1,
            Err(i) => i,
        };
        end_index = end_index.min(self.pars.len());
        let lazy_df = if start_index < end_index {
            let mut pars: Vec<PathBuf> = Vec::new();
            for par_num in &self.pars[start_index..end_index] {
                let mut par_path = par_path.clone();
                par_path.push(self.get_par_glob(*par_num));
                for entry in glob::glob(&par_path.to_string_lossy())
                    .map_err(|e| SpicyError::Err(e.to_string()))?
                {
                    match entry {
                        Ok(path) => pars.push(path),
                        Err(e) => eprintln!("{:?}", e),
                    }
                }
            }
            LazyFrame::scan_parquet_files(
                pars.into_iter()
                    .map(|p| PlPath::Local(p.as_path().into()))
                    .collect(),
                args,
            )
            .map_err(|e| {
                SpicyError::Err(format!("failed to scan partitioned {}: {}", self.name, e))
            })?
        } else {
            par_path.push(self.get_empty_schema());
            LazyFrame::scan_parquet(PlPath::Local(par_path.as_path().into()), args).map_err(
                |e| SpicyError::Err(format!("failed to scan partitioned {}: {}", self.name, e)),
            )?
        };
        Ok(lazy_df)
    }
}

impl Display for PartitionedDataFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "([] `par_df {}, {:?})", self.name, self.df_type)
    }
}
