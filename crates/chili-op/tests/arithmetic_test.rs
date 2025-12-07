use chili_core::SpicyObj;
use chili_op::operator;
use indexmap::IndexMap;
use polars::{
    datatypes::{DataType, TimeUnit},
    prelude::NamedFrom,
    series::Series,
};

#[test]
fn add() {
    let nu = SpicyObj::Null;
    let b = SpicyObj::Boolean(true);
    let c = SpicyObj::U8(3);
    let h = SpicyObj::I16(-5);
    let i = SpicyObj::I32(7);
    let j = SpicyObj::I64(-11);

    let e = SpicyObj::F32(2.71);
    let f = SpicyObj::F64(3.13);

    // 2024.04.30
    let d = SpicyObj::Date(19843);
    // 0D02:31:08.110000000
    let t = SpicyObj::Time(9068110000000);
    // 2024.04.30T02:32:01.123
    let z = SpicyObj::Datetime(767759521123);
    // 2024.04.30D02:32:01.123456789
    let p = SpicyObj::Timestamp(767759521123456789);
    // 1D12:34:56.000000000
    let n = SpicyObj::Duration(131696000000000);

    let sb = SpicyObj::Series(Series::new("".into(), vec![Some(true), Some(false), None]));
    let sc = SpicyObj::Series(Series::new("".into(), vec![Some(13u8), Some(17u8), None]));
    let sh = SpicyObj::Series(Series::new(
        "".into(),
        vec![Some(-19i16), Some(23i16), None],
    ));
    let si = SpicyObj::Series(Series::new(
        "".into(),
        vec![Some(-29i32), Some(31i32), None],
    ));
    let sj = SpicyObj::Series(Series::new(
        "".into(),
        vec![Some(-37i64), Some(41i64), None],
    ));
    let se = SpicyObj::Series(Series::new(
        "".into(),
        vec![Some(-2.71f32), Some(3.13f32), Some(f32::INFINITY), None],
    ));
    let sf = SpicyObj::Series(Series::new(
        "".into(),
        vec![Some(-2.71f64), Some(3.13f64), Some(f64::INFINITY), None],
    ));
    let l = SpicyObj::MixedList(vec![b.clone(), c.clone(), e.clone()]);
    let mut m = IndexMap::new();
    m.insert("b".to_string(), b.clone());
    m.insert("c".to_string(), c.clone());
    m.insert("e".to_string(), e.clone());
    let m = SpicyObj::Dict(m);
    for (args, expect) in vec![
        (vec![&nu, &c], SpicyObj::Null),
        (vec![&b, &b], SpicyObj::I64(2)),
        (vec![&b, &c], SpicyObj::U8(4)),
        (vec![&b, &h], SpicyObj::I16(-4)),
        (vec![&b, &i], SpicyObj::I32(8)),
        (vec![&b, &j], SpicyObj::I64(-10)),
        (vec![&b, &e], SpicyObj::F32(3.71)),
        (vec![&b, &f], SpicyObj::F64(4.13)),
        (vec![&b, &d], SpicyObj::Date(19844)),
        (vec![&b, &t], SpicyObj::Time(9068110000001)),
        (vec![&b, &z], SpicyObj::Datetime(767759521124)),
        (vec![&b, &p], SpicyObj::Timestamp(767759521123456790)),
        (vec![&b, &n], SpicyObj::Duration(131696000000001)),
        // list
        (
            vec![&c, &sb],
            SpicyObj::Series(Series::new("".into(), vec![Some(4u8), Some(3u8), None])),
        ),
        (
            vec![&c, &sc],
            SpicyObj::Series(Series::new("".into(), vec![Some(16u8), Some(20u8), None])),
        ),
        (
            vec![&c, &sh],
            SpicyObj::Series(Series::new(
                "".into(),
                vec![Some(-16i16), Some(26i16), None],
            )),
        ),
        (
            vec![&c, &si],
            SpicyObj::Series(Series::new(
                "".into(),
                vec![Some(-26i32), Some(34i32), None],
            )),
        ),
        (
            vec![&c, &sj],
            SpicyObj::Series(Series::new("".into(), vec![Some(-34), Some(44), None])),
        ),
        // float
        (vec![&e, &c], SpicyObj::F32(5.71)),
        (vec![&e, &h], SpicyObj::F32(-2.2899999618530273)),
        (vec![&e, &i], SpicyObj::F32(9.710000038146973)),
        (vec![&e, &j], SpicyObj::F32(-8.289999961853027)),
        (vec![&e, &e], SpicyObj::F32(5.420000076293945)),
        (vec![&e, &f], SpicyObj::F64(5.8400000381469725)),
        (vec![&e, &d], SpicyObj::F32(19845.71)),
        (vec![&e, &t], SpicyObj::F32(9068110000002.71)),
        (vec![&e, &z], SpicyObj::F32(767759550000.0)),
        (vec![&e, &p], SpicyObj::F32(7.677595e17)),
        (vec![&e, &n], SpicyObj::F32(131696000000000.0)),
        (vec![&d, &t], SpicyObj::Date(19843)),
        (
            vec![&d, &si],
            SpicyObj::Series(
                Series::new("".into(), vec![Some(19814), Some(19874), None])
                    .cast(&DataType::Date)
                    .unwrap(),
            ),
        ),
        (vec![&t, &t], SpicyObj::Duration(18136220000000)),
        (vec![&z, &t], SpicyObj::Datetime(767768589233)),
        (
            vec![&l, &l],
            SpicyObj::MixedList(vec![SpicyObj::I64(2), SpicyObj::U8(6), SpicyObj::F32(5.42)]),
        ),
        (
            vec![&l, &m],
            SpicyObj::Dict(
                [
                    ("b".to_string(), SpicyObj::I64(2)),
                    ("c".to_string(), SpicyObj::U8(6)),
                    ("e".to_string(), SpicyObj::F32(5.42)),
                ]
                .into_iter()
                .collect(),
            ),
        ),
        (
            vec![&m, &sh],
            SpicyObj::Dict(
                [
                    ("b".to_string(), SpicyObj::I16(-18)),
                    ("c".to_string(), SpicyObj::I16(26)),
                    ("e".to_string(), SpicyObj::Null),
                ]
                .into_iter()
                .collect(),
            ),
        ),
        (
            vec![&m, &m],
            SpicyObj::Dict(
                [
                    ("b".to_string(), SpicyObj::I64(2)),
                    ("c".to_string(), SpicyObj::U8(6)),
                    ("e".to_string(), SpicyObj::F32(5.42)),
                ]
                .into_iter()
                .collect(),
            ),
        ),
        (
            vec![&c, &se],
            SpicyObj::Series(Series::new(
                "".into(),
                vec![
                    Some(0.289_999_96_f32),
                    Some(6.13f32),
                    Some(f32::INFINITY),
                    None,
                ],
            )),
        ),
        (
            vec![&c, &sf],
            SpicyObj::Series(Series::new(
                "".into(),
                vec![
                    Some(0.29000000000000004f64),
                    Some(6.13f64),
                    Some(f64::INFINITY),
                    None,
                ],
            )),
        ),
        (
            vec![&sh, &sj],
            SpicyObj::Series(Series::new(
                "".into(),
                vec![Some(-56i64), Some(64i64), None],
            )),
        ),
    ]
    .iter()
    {
        match operator::add(args) {
            Ok(j) => assert_eq!(j, *expect, "test case - {:?}", args),
            Err(e) => panic!("{} - {:?}", e, args),
        }
    }

    for args in [[&d, &d], [&d, &z], [&d, &p]].iter() {
        assert!(operator::add(args).is_err(), "error case - {:?}", args)
    }
}

#[test]
fn minus() {
    let nu = SpicyObj::Null;
    let b = SpicyObj::Boolean(true);
    let c = SpicyObj::U8(3);
    let h = SpicyObj::I16(-5);
    let i = SpicyObj::I32(7);
    let j = SpicyObj::I64(-11);

    let e = SpicyObj::F32(2.71);
    let f = SpicyObj::F64(3.13);

    // 2024.04.30
    let d = SpicyObj::Date(19843);
    // 0D02:31:08.110000000
    let t = SpicyObj::Time(9068110000000);
    // 2024.04.30T02:32:01.123
    let z = SpicyObj::Datetime(767759521123);
    // 2024.04.30D02:32:01.123456789
    let p = SpicyObj::Timestamp(767759521123456789);
    // 1D12:34:56.000000000
    let n = SpicyObj::Duration(131696000000000);

    let sb = SpicyObj::Series(Series::new("".into(), vec![Some(true), Some(false), None]));
    let sc = SpicyObj::Series(Series::new("".into(), vec![Some(13u8), Some(17u8), None]));
    let sh = SpicyObj::Series(Series::new(
        "".into(),
        vec![Some(-19i16), Some(23i16), None],
    ));
    let si = SpicyObj::Series(Series::new(
        "".into(),
        vec![Some(-29i32), Some(31i32), None],
    ));
    let sj = SpicyObj::Series(Series::new(
        "".into(),
        vec![Some(-37i64), Some(41i64), None],
    ));
    let se = SpicyObj::Series(Series::new(
        "".into(),
        vec![Some(-2.71f32), Some(3.13f32), Some(f32::INFINITY), None],
    ));
    let sf = SpicyObj::Series(Series::new(
        "".into(),
        vec![Some(-2.71f64), Some(3.13f64), Some(f64::INFINITY), None],
    ));
    let sd = SpicyObj::Series(
        Series::new("".into(), vec![Some(19843), Some(19844), None])
            .cast(&DataType::Date)
            .unwrap(),
    );
    let sz = SpicyObj::Series(
        Series::new("".into(), vec![Some(767759521123i64), None])
            .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
            .unwrap(),
    );
    let sp = SpicyObj::Series(
        Series::new("".into(), vec![Some(767759521123456789i64), None])
            .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
            .unwrap(),
    );
    let l = SpicyObj::MixedList(vec![b.clone(), c.clone(), e.clone()]);
    let mut m = IndexMap::new();
    m.insert("b".to_string(), b.clone());
    m.insert("c".to_string(), c.clone());
    m.insert("e".to_string(), e.clone());
    let m = SpicyObj::Dict(m);
    for (args, expect) in vec![
        (vec![&nu, &c], SpicyObj::Null),
        (vec![&b, &b], SpicyObj::I64(0)),
        (vec![&b, &c], SpicyObj::U8(254)),
        (vec![&b, &h], SpicyObj::I16(6)),
        (vec![&b, &i], SpicyObj::I32(-6)),
        (vec![&b, &j], SpicyObj::I64(12)),
        (vec![&b, &e], SpicyObj::F32(-1.71)),
        (vec![&b, &f], SpicyObj::F64(-2.13)),
        // list
        (
            vec![&c, &sb],
            SpicyObj::Series(Series::new("".into(), vec![Some(2), Some(3), None])),
        ),
        (
            vec![&c, &sc],
            SpicyObj::Series(Series::new("".into(), vec![Some(246u8), Some(242u8), None])),
        ),
        (
            vec![&c, &sh],
            SpicyObj::Series(Series::new("".into(), vec![Some(22), Some(-20), None])),
        ),
        (
            vec![&c, &si],
            SpicyObj::Series(Series::new("".into(), vec![Some(32), Some(-28), None])),
        ),
        (
            vec![&c, &sj],
            SpicyObj::Series(Series::new("".into(), vec![Some(40), Some(-38), None])),
        ),
        (
            vec![&c, &se],
            SpicyObj::Series(Series::new(
                "".into(),
                vec![
                    Some(5.71_f32),
                    Some(-0.130_000_11_f32),
                    Some(f32::NEG_INFINITY),
                    None,
                ],
            )),
        ),
        (
            vec![&c, &sf],
            SpicyObj::Series(Series::new(
                "".into(),
                vec![
                    Some(5.71),
                    Some(-0.1299999999999999),
                    Some(f64::NEG_INFINITY),
                    None,
                ],
            )),
        ),
        (vec![&d, &d], SpicyObj::I64(0)),
        (vec![&d, &z], SpicyObj::Duration(946675678877000000)),
        (vec![&d, &p], SpicyObj::Duration(946675678876543211)),
        // float
        (vec![&e, &c], SpicyObj::F32(-0.28999996)),
        (vec![&e, &h], SpicyObj::F32(7.71)),
        (vec![&e, &i], SpicyObj::F32(-4.29)),
        (vec![&e, &j], SpicyObj::F32(13.71)),
        (vec![&e, &e], SpicyObj::F32(0.0)),
        (vec![&e, &f], SpicyObj::F64(-0.41999996185302724)),
        (vec![&e, &d], SpicyObj::F32(-19840.29)),
        (vec![&e, &t], SpicyObj::F32(-9068110000000.0)),
        (vec![&e, &z], SpicyObj::F32(-767759550000.0)),
        (vec![&e, &p], SpicyObj::F32(-7.677595e17)),
        (vec![&e, &n], SpicyObj::F32(-131696000000000.0)),
        // date
        (vec![&d, &t], SpicyObj::Date(19843)),
        (
            vec![&d, &si],
            SpicyObj::Series(
                Series::new("".into(), vec![Some(19872), Some(19812), None])
                    .cast(&DataType::Date)
                    .unwrap(),
            ),
        ),
        (vec![&t, &n], SpicyObj::Duration(-122627890000000)),
        (vec![&z, &t], SpicyObj::Datetime(767750453013)),
        (
            vec![&l, &l],
            SpicyObj::MixedList(vec![SpicyObj::I64(0), SpicyObj::U8(0), SpicyObj::F32(0.0)]),
        ),
        (
            vec![&l, &m],
            SpicyObj::Dict(
                [
                    ("b".to_string(), SpicyObj::I64(0)),
                    ("c".to_string(), SpicyObj::U8(0)),
                    ("e".to_string(), SpicyObj::F32(0.0)),
                ]
                .into_iter()
                .collect(),
            ),
        ),
        (
            vec![&m, &sh],
            SpicyObj::Dict(
                [
                    ("b".to_string(), SpicyObj::I16(20)),
                    ("c".to_string(), SpicyObj::I16(-20)),
                    ("e".to_string(), SpicyObj::Null),
                ]
                .into_iter()
                .collect(),
            ),
        ),
        (
            vec![&m, &m],
            SpicyObj::Dict(
                [
                    ("b".to_string(), SpicyObj::I64(0)),
                    ("c".to_string(), SpicyObj::U8(0)),
                    ("e".to_string(), SpicyObj::F32(0.0)),
                ]
                .into_iter()
                .collect(),
            ),
        ),
        (
            vec![&sh, &sj],
            SpicyObj::Series(Series::new(
                "".into(),
                vec![Some(18i64), Some(-18i64), None],
            )),
        ),
    ]
    .iter()
    {
        match operator::minus(args) {
            Ok(j) => assert_eq!(j, *expect, "test case - {:?}", args),
            Err(e) => panic!("{} - {:?}", e, args),
        }
    }

    for args in vec![vec![&c, &sd], vec![&c, &sz], vec![&c, &sp], vec![&sh, &sd]].iter() {
        assert!(operator::minus(args).is_err(), "error case - {:?}", args)
    }
}

#[test]
fn mul() {
    let nu = SpicyObj::Null;
    let b = SpicyObj::Boolean(true);
    let c = SpicyObj::U8(3);
    let h = SpicyObj::I16(-5);
    let i = SpicyObj::I32(7);
    let j = SpicyObj::I64(-11);

    let e = SpicyObj::F32(2.71);
    let f = SpicyObj::F64(3.13);

    // 2024.04.30
    let d = SpicyObj::Date(19843);
    // 0D02:31:08.110000000
    let t = SpicyObj::Time(9068110000000);
    // 2024.04.30T02:32:01.123
    let z = SpicyObj::Datetime(767759521123);
    // 2024.04.30D02:32:01.123456789
    let p = SpicyObj::Timestamp(767759521123456789);
    // 1D12:34:56.000000000
    let n = SpicyObj::Duration(131696000000000);

    let sb = SpicyObj::Series(Series::new("".into(), vec![Some(true), Some(false), None]));
    let sc = SpicyObj::Series(Series::new("".into(), vec![Some(13u8), Some(17u8), None]));
    let sh = SpicyObj::Series(Series::new(
        "".into(),
        vec![Some(-19i16), Some(23i16), None],
    ));
    let si = SpicyObj::Series(Series::new(
        "".into(),
        vec![Some(-29i32), Some(31i32), None],
    ));
    let sj = SpicyObj::Series(Series::new(
        "".into(),
        vec![Some(-37i64), Some(41i64), None],
    ));
    let se = SpicyObj::Series(Series::new(
        "".into(),
        vec![Some(-2.71f32), Some(3.13f32), Some(f32::INFINITY), None],
    ));
    let sf = SpicyObj::Series(Series::new(
        "".into(),
        vec![Some(-2.71f64), Some(3.13f64), Some(f64::INFINITY), None],
    ));
    let sd = SpicyObj::Series(
        Series::new("".into(), vec![Some(19843), Some(19844), None])
            .cast(&DataType::Date)
            .unwrap(),
    );
    let sz = SpicyObj::Series(
        Series::new("".into(), vec![Some(767759521123i64), None])
            .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
            .unwrap(),
    );
    let sp = SpicyObj::Series(
        Series::new("".into(), vec![Some(767759521123456789i64), None])
            .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
            .unwrap(),
    );
    let l = SpicyObj::MixedList(vec![b.clone(), c.clone(), e.clone()]);
    let mut m = IndexMap::new();
    m.insert("b".to_string(), b.clone());
    m.insert("c".to_string(), c.clone());
    m.insert("e".to_string(), e.clone());
    let m = SpicyObj::Dict(m);
    for (args, expect) in vec![
        (vec![&nu, &c], SpicyObj::Null),
        (vec![&b, &b], SpicyObj::I64(1)),
        (vec![&b, &c], SpicyObj::U8(3)),
        (vec![&b, &h], SpicyObj::I16(-5)),
        (vec![&b, &i], SpicyObj::I32(7)),
        (vec![&b, &j], SpicyObj::I64(-11)),
        (vec![&b, &e], SpicyObj::F32(2.71)),
        (vec![&b, &f], SpicyObj::F64(3.13)),
        (vec![&b, &n], SpicyObj::Duration(131696000000000)),
        // list
        (
            vec![&c, &sb],
            SpicyObj::Series(Series::new("".into(), vec![Some(3), Some(0), None])),
        ),
        (
            vec![&c, &sc],
            SpicyObj::Series(Series::new("".into(), vec![Some(39), Some(51), None])),
        ),
        (
            vec![&c, &sh],
            SpicyObj::Series(Series::new("".into(), vec![Some(-57), Some(69), None])),
        ),
        (
            vec![&c, &si],
            SpicyObj::Series(Series::new("".into(), vec![Some(-87), Some(93), None])),
        ),
        (
            vec![&c, &sj],
            SpicyObj::Series(Series::new("".into(), vec![Some(-111), Some(123), None])),
        ),
        // float
        (vec![&e, &c], SpicyObj::F32(8.13)),
        (vec![&e, &h], SpicyObj::F32(-13.550000190734863)),
        (vec![&e, &i], SpicyObj::F32(18.97000026702881)),
        (vec![&e, &j], SpicyObj::F32(-29.8100004196167)),
        (vec![&e, &e], SpicyObj::F32(7.344100206756593)),
        (vec![&e, &f], SpicyObj::F64(8.482300119400024)),
        (vec![&e, &n], SpicyObj::Duration(356896165023803)),
        (
            vec![&l, &l],
            SpicyObj::MixedList(vec![
                SpicyObj::I64(1),
                SpicyObj::U8(9),
                SpicyObj::F32(7.3441),
            ]),
        ),
        (
            vec![&l, &m],
            SpicyObj::Dict(
                [
                    ("b".to_string(), SpicyObj::I64(1)),
                    ("c".to_string(), SpicyObj::U8(9)),
                    ("e".to_string(), SpicyObj::F32(7.3441)),
                ]
                .into_iter()
                .collect(),
            ),
        ),
        (
            vec![&m, &sh],
            SpicyObj::Dict(
                [
                    ("b".to_string(), SpicyObj::I16(-19)),
                    ("c".to_string(), SpicyObj::I16(69)),
                    ("e".to_string(), SpicyObj::Null),
                ]
                .into_iter()
                .collect(),
            ),
        ),
        (
            vec![&m, &m],
            SpicyObj::Dict(
                [
                    ("b".to_string(), SpicyObj::I64(1)),
                    ("c".to_string(), SpicyObj::U8(9)),
                    ("e".to_string(), SpicyObj::F32(7.3441)),
                ]
                .into_iter()
                .collect(),
            ),
        ),
        (
            vec![&c, &se],
            SpicyObj::Series(Series::new(
                "".into(),
                vec![Some(-8.13_f32), Some(9.39_f32), Some(f32::INFINITY), None],
            )),
        ),
        (
            vec![&c, &sf],
            SpicyObj::Series(Series::new(
                "".into(),
                vec![
                    Some(-8.129999999999999),
                    Some(9.39),
                    Some(f64::INFINITY),
                    None,
                ],
            )),
        ),
        (
            vec![&sh, &sj],
            SpicyObj::Series(Series::new(
                "".into(),
                vec![Some(703i64), Some(943i64), None],
            )),
        ),
    ]
    .iter()
    {
        match operator::mul(args) {
            Ok(j) => assert_eq!(j, *expect, "test case - {:?}", args),
            Err(e) => panic!("{} - {:?}", e, args),
        }
    }

    for args in vec![
        vec![&b, &d],
        vec![&b, &z],
        vec![&b, &p],
        vec![&c, &sd],
        vec![&c, &sz],
        vec![&c, &sp],
        vec![&e, &d],
        vec![&e, &t],
        vec![&e, &z],
        vec![&e, &p],
        vec![&d, &d],
        vec![&d, &z],
        vec![&d, &p],
        vec![&d, &t],
        vec![&d, &si],
        vec![&t, &t],
        vec![&z, &t],
    ]
    .iter()
    {
        assert!(operator::mul(args).is_err(), "error case - {:?}", args)
    }
}

#[test]
fn div() {
    let nu = SpicyObj::Null;
    let b = SpicyObj::Boolean(true);
    let c = SpicyObj::U8(3);
    let h = SpicyObj::I16(-5);
    let i = SpicyObj::I32(7);
    let j = SpicyObj::I64(-11);

    let e = SpicyObj::F32(2.71);
    let f = SpicyObj::F64(3.13);

    // 2024.04.30
    let d = SpicyObj::Date(19843);
    // 0D02:31:08.110000000
    let t = SpicyObj::Time(9068110000000);
    // 2024.04.30T02:32:01.123
    let z = SpicyObj::Datetime(767759521123);
    // 2024.04.30D02:32:01.123456789
    let p = SpicyObj::Timestamp(767759521123456789);
    // 1D12:34:56.000000000
    let n = SpicyObj::Duration(131696000000000);

    let sb = SpicyObj::Series(Series::new("".into(), vec![Some(true), Some(false), None]));
    let sc = SpicyObj::Series(Series::new("".into(), vec![Some(13u8), Some(17u8), None]));
    let sh = SpicyObj::Series(Series::new(
        "".into(),
        vec![Some(-19i16), Some(23i16), None],
    ));
    let si = SpicyObj::Series(Series::new(
        "".into(),
        vec![Some(-29i32), Some(31i32), None],
    ));
    let sj = SpicyObj::Series(Series::new(
        "".into(),
        vec![Some(-37i64), Some(41i64), None],
    ));
    let se = SpicyObj::Series(Series::new(
        "".into(),
        vec![Some(-2.71f32), Some(3.13f32), Some(f32::INFINITY), None],
    ));
    let sf = SpicyObj::Series(Series::new(
        "".into(),
        vec![Some(-2.71f64), Some(3.13f64), Some(f64::INFINITY), None],
    ));
    let sd = SpicyObj::Series(
        Series::new("".into(), vec![Some(19843), Some(19844), None])
            .cast(&DataType::Date)
            .unwrap(),
    );
    let l = SpicyObj::MixedList(vec![b.clone(), c.clone(), e.clone()]);
    let mut m = IndexMap::new();
    m.insert("b".to_string(), b.clone());
    m.insert("c".to_string(), c.clone());
    m.insert("e".to_string(), e.clone());
    let m = SpicyObj::Dict(m);
    for (args, expect) in vec![
        (vec![&nu, &c], SpicyObj::Null),
        (vec![&b, &b], SpicyObj::F64(1.0)),
        (vec![&b, &c], SpicyObj::F64(0.333_333_333_333_333_3)),
        (vec![&b, &h], SpicyObj::F64(-0.2)),
        (vec![&b, &i], SpicyObj::F64(0.14285714285714285)),
        (vec![&b, &j], SpicyObj::F64(-0.090_909_090_909_090_91)),
        (vec![&b, &e], SpicyObj::F64(0.3690036848426666)),
        (vec![&b, &f], SpicyObj::F64(0.3194888178913738)),
        // list
        (
            vec![&c, &sb],
            SpicyObj::Series(Series::new(
                "".into(),
                vec![Some(3.0), Some(f64::INFINITY), None],
            )),
        ),
        (
            vec![&c, &sc],
            SpicyObj::Series(Series::new(
                "".into(),
                vec![Some(0.23076923076923078), Some(0.17647058823529413), None],
            )),
        ),
        (
            vec![&c, &sh],
            SpicyObj::Series(Series::new(
                "".into(),
                vec![Some(-0.15789473684210525), Some(0.13043478260869565), None],
            )),
        ),
        (
            vec![&c, &si],
            SpicyObj::Series(Series::new(
                "".into(),
                vec![
                    Some(-0.10344827586206896),
                    Some(0.096_774_193_548_387_1),
                    None,
                ],
            )),
        ),
        (
            vec![&c, &sj],
            SpicyObj::Series(Series::new(
                "".into(),
                vec![
                    Some(-0.081_081_081_081_081_09),
                    Some(0.073_170_731_707_317_07),
                    None,
                ],
            )),
        ),
        (
            vec![&c, &se],
            SpicyObj::Series(Series::new(
                "".into(),
                vec![
                    Some(-1.107_011_1_f32),
                    Some(0.958_466_4_f32),
                    Some(0.0f32),
                    None,
                ],
            )),
        ),
        (
            vec![&c, &sf],
            SpicyObj::Series(Series::new(
                "".into(),
                vec![
                    Some(-1.1070110701107012),
                    Some(0.9584664536741214),
                    Some(0.0),
                    None,
                ],
            )),
        ),
        // float
        (vec![&e, &c], SpicyObj::F64(0.903_333_346_048_990_8)),
        (vec![&e, &h], SpicyObj::F64(-0.542_000_007_629_394_5)),
        (vec![&e, &i], SpicyObj::F64(0.38714286259242464)),
        (vec![&e, &j], SpicyObj::F64(-0.24636363983154297)),
        (vec![&e, &e], SpicyObj::F64(1.0)),
        (vec![&e, &f], SpicyObj::F64(0.8658147086731542)),
        (
            vec![&l, &l],
            SpicyObj::MixedList(vec![
                SpicyObj::F64(1.0),
                SpicyObj::F64(1.0),
                SpicyObj::F64(1.0),
            ]),
        ),
        (
            vec![&l, &m],
            SpicyObj::Dict(
                [
                    ("b".to_string(), SpicyObj::F64(1.0)),
                    ("c".to_string(), SpicyObj::F64(1.0)),
                    ("e".to_string(), SpicyObj::F64(1.0)),
                ]
                .into_iter()
                .collect(),
            ),
        ),
        (
            vec![&m, &sh],
            SpicyObj::Dict(
                [
                    ("b".to_string(), SpicyObj::F64(-0.052_631_578_947_368_42)),
                    ("c".to_string(), SpicyObj::F64(0.13043478260869565)),
                    ("e".to_string(), SpicyObj::Null),
                ]
                .into_iter()
                .collect(),
            ),
        ),
        (
            vec![&m, &m],
            SpicyObj::Dict(
                [
                    ("b".to_string(), SpicyObj::F64(1.0)),
                    ("c".to_string(), SpicyObj::F64(1.0)),
                    ("e".to_string(), SpicyObj::F64(1.0)),
                ]
                .into_iter()
                .collect(),
            ),
        ),
        (
            vec![&sh, &sj],
            SpicyObj::Series(Series::new(
                "".into(),
                vec![
                    Some(0.513_513_513_513_513_5),
                    Some(0.560_975_609_756_097_6),
                    None,
                ],
            )),
        ),
    ]
    .iter()
    {
        match operator::true_div(args) {
            Ok(j) => assert_eq!(j, *expect, "test case - {:?}", args),
            Err(e) => panic!("{} - {:?}", e, args),
        }
    }

    for args in vec![
        vec![&b, &d],
        vec![&b, &z],
        vec![&b, &p],
        vec![&b, &t],
        vec![&b, &n],
        vec![&d, &d],
        vec![&d, &z],
        vec![&d, &p],
        vec![&d, &t],
        vec![&d, &si],
        vec![&e, &d],
        vec![&e, &t],
        vec![&e, &z],
        vec![&e, &p],
        vec![&e, &n],
        vec![&z, &t],
        vec![&sd, &sh],
    ]
    .iter()
    {
        assert!(operator::true_div(args).is_err(), "error case - {:?}", args)
    }
}

#[test]
fn gt() {
    let nu = SpicyObj::Null;
    let b = SpicyObj::Boolean(true);
    let c = SpicyObj::U8(3);
    let h = SpicyObj::I16(-5);
    let i = SpicyObj::I32(7);
    let j = SpicyObj::I64(-11);

    let e = SpicyObj::F32(2.71);
    let f = SpicyObj::F64(3.13);

    // 2024.04.30
    let d = SpicyObj::Date(19843);
    // 0D02:31:08.110000000
    let t = SpicyObj::Time(9068110000000);
    // 1994.05.01T02:32:01.123
    let z = SpicyObj::Datetime(767759521123);
    // 1994.05.01D02:32:01.123456789
    let p = SpicyObj::Timestamp(767759521123456789);
    // 1D12:34:56.000000000
    let n = SpicyObj::Duration(131696000000000);

    let sb = SpicyObj::Series(Series::new("".into(), vec![Some(true), Some(false), None]));
    let sc = SpicyObj::Series(Series::new("".into(), vec![Some(13u8), Some(17u8), None]));
    let sh = SpicyObj::Series(Series::new(
        "".into(),
        vec![Some(-19i16), Some(23i16), None],
    ));
    let si = SpicyObj::Series(Series::new(
        "".into(),
        vec![Some(-29i32), Some(31i32), None],
    ));
    let sj = SpicyObj::Series(Series::new(
        "".into(),
        vec![Some(-37i64), Some(41i64), None],
    ));
    let se = SpicyObj::Series(Series::new(
        "".into(),
        vec![Some(-2.71f32), Some(3.13f32), Some(f32::INFINITY), None],
    ));
    let sf = SpicyObj::Series(Series::new(
        "".into(),
        vec![Some(-2.71f64), Some(3.13f64), Some(f64::INFINITY), None],
    ));
    let l = SpicyObj::MixedList(vec![b.clone(), c.clone(), e.clone()]);
    let mut m = IndexMap::new();
    m.insert("b".to_string(), b.clone());
    m.insert("c".to_string(), c.clone());
    m.insert("e".to_string(), e.clone());
    let m = SpicyObj::Dict(m);
    for (args, expect) in vec![
        (vec![&nu, &c], SpicyObj::Null),
        (vec![&b, &b], SpicyObj::Boolean(false)),
        (vec![&b, &c], SpicyObj::Boolean(false)),
        (vec![&b, &h], SpicyObj::Boolean(true)),
        (vec![&b, &i], SpicyObj::Boolean(false)),
        (vec![&b, &j], SpicyObj::Boolean(true)),
        (vec![&b, &e], SpicyObj::Boolean(false)),
        (vec![&b, &f], SpicyObj::Boolean(false)),
        (vec![&b, &d], SpicyObj::Boolean(false)),
        (vec![&b, &z], SpicyObj::Boolean(false)),
        (vec![&b, &p], SpicyObj::Boolean(false)),
        (vec![&b, &t], SpicyObj::Boolean(false)),
        (vec![&b, &n], SpicyObj::Boolean(false)),
        // list
        (
            vec![&c, &sb],
            SpicyObj::Series(Series::new("".into(), vec![Some(true), Some(true), None])),
        ),
        (
            vec![&c, &sc],
            SpicyObj::Series(Series::new("".into(), vec![Some(false), Some(false), None])),
        ),
        (
            vec![&c, &sh],
            SpicyObj::Series(Series::new("".into(), vec![Some(true), Some(false), None])),
        ),
        (
            vec![&c, &si],
            SpicyObj::Series(Series::new("".into(), vec![Some(true), Some(false), None])),
        ),
        (
            vec![&c, &sj],
            SpicyObj::Series(Series::new("".into(), vec![Some(true), Some(false), None])),
        ),
        (
            vec![&c, &se],
            SpicyObj::Series(Series::new(
                "".into(),
                vec![Some(true), Some(false), Some(false), None],
            )),
        ),
        (
            vec![&c, &sf],
            SpicyObj::Series(Series::new(
                "".into(),
                vec![Some(true), Some(false), Some(false), None],
            )),
        ),
        (vec![&d, &d], SpicyObj::Boolean(false)),
        (vec![&d, &z], SpicyObj::Boolean(true)),
        (vec![&d, &p], SpicyObj::Boolean(true)),
        (
            vec![&d, &si],
            SpicyObj::Series(Series::new("".into(), vec![Some(true), Some(true), None])),
        ),
        // float
        (vec![&e, &c], SpicyObj::Boolean(false)),
        (vec![&e, &h], SpicyObj::Boolean(true)),
        (vec![&e, &i], SpicyObj::Boolean(false)),
        (vec![&e, &j], SpicyObj::Boolean(true)),
        (vec![&e, &e], SpicyObj::Boolean(false)),
        (vec![&e, &f], SpicyObj::Boolean(false)),
        (vec![&e, &d], SpicyObj::Boolean(false)),
        (vec![&e, &t], SpicyObj::Boolean(false)),
        (vec![&e, &z], SpicyObj::Boolean(false)),
        (vec![&e, &p], SpicyObj::Boolean(false)),
        (vec![&e, &n], SpicyObj::Boolean(false)),
        (
            vec![&l, &l],
            SpicyObj::MixedList(vec![
                SpicyObj::Boolean(false),
                SpicyObj::Boolean(false),
                SpicyObj::Boolean(false),
            ]),
        ),
        (
            vec![&l, &m],
            SpicyObj::Dict(
                [
                    ("b".to_string(), SpicyObj::Boolean(false)),
                    ("c".to_string(), SpicyObj::Boolean(false)),
                    ("e".to_string(), SpicyObj::Boolean(false)),
                ]
                .into_iter()
                .collect(),
            ),
        ),
        (
            vec![&m, &sh],
            SpicyObj::Dict(
                [
                    ("b".to_string(), SpicyObj::Boolean(true)),
                    ("c".to_string(), SpicyObj::Boolean(false)),
                    ("e".to_string(), SpicyObj::Null),
                ]
                .into_iter()
                .collect(),
            ),
        ),
        (
            vec![&m, &m],
            SpicyObj::Dict(
                [
                    ("b".to_string(), SpicyObj::Boolean(false)),
                    ("c".to_string(), SpicyObj::Boolean(false)),
                    ("e".to_string(), SpicyObj::Boolean(false)),
                ]
                .into_iter()
                .collect(),
            ),
        ),
        (vec![&z, &t], SpicyObj::Boolean(true)),
        (
            vec![&sh, &sj],
            SpicyObj::Series(Series::new("".into(), vec![Some(true), Some(false), None])),
        ),
    ]
    .iter()
    {
        match operator::gt(args) {
            Ok(obj) => assert_eq!(obj, *expect, "test case - {:?}", args),
            Err(e) => panic!("{} - {:?}", e, args),
        }
    }

    for args in [vec![&d, &t], vec![&t, &n]].iter() {
        assert!(operator::gt(args).is_err(), "error case - {:?}", args)
    }
}
