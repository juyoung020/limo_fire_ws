//! .pt → safetensors (float32).
//!
//! torch.save 파일은 zip 안에 `*/data.pkl`(pickle)과 `*/data/<키>`(텐서 저장소 바이트)가 있다. pickle 을 파이썬처럼
//! "실행"하면 임의 코드가 돌 수 있으므로, 여기서는 연산 부호를 자료 구조로만 해석한다: GLOBAL 은 (모듈, 이름) 표시로만
//! 남기고, REDUCE·NEWOBJ·BUILD 는 "어떤 클래스·인자·상태" 묶음으로만 만든다. 텐서는
//! torch._utils._rebuild_tensor_v2 / _rebuild_parameter 묶음을 알아보고 저장소 바이트에서 꺼낸다.
//! nn.Module 객체는 상태의 `_parameters`·`_buffers`·`_modules` 를 따라가 PyTorch state_dict 와 같은 이름을 만든다.
//!
//! 사용: ptconv <in.pt> <out.safetensors> [최상위 키, 예: model] [--list]
use anyhow::{anyhow, bail, Context, Result};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io::Read;
use std::rc::Rc;

#[derive(Clone, Debug)]
#[allow(dead_code)]
enum V {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
    Tuple(Vec<V>),
    List(Rc<RefCell<Vec<V>>>),
    Dict(Rc<RefCell<Vec<(V, V)>>>),
    Global(String, String),
    PersId(Box<V>),
    // REDUCE / NEWOBJ: 호출 대상과 인자, BUILD 로 받은 상태
    Obj(Rc<RefCell<Obj>>),
    Mark,
}
#[derive(Debug)]
struct Obj {
    callable: V,
    args: V,
    state: Option<V>,
}

fn obj(callable: V, args: V) -> V {
    V::Obj(Rc::new(RefCell::new(Obj { callable, args, state: None })))
}

struct Pk<'a> {
    d: &'a [u8],
    p: usize,
}
impl<'a> Pk<'a> {
    fn u8(&mut self) -> Result<u8> {
        let v = *self.d.get(self.p).context("pickle 끝")?;
        self.p += 1;
        Ok(v)
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.p + n > self.d.len() {
            bail!("pickle 끝");
        }
        let s = &self.d[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn le(&mut self, n: usize) -> Result<u64> {
        let s = self.take(n)?;
        let mut v = 0u64;
        for (i, b) in s.iter().enumerate() {
            v |= (*b as u64) << (8 * i);
        }
        Ok(v)
    }
    fn line(&mut self) -> Result<String> {
        let st = self.p;
        while self.u8()? != b'\n' {}
        Ok(String::from_utf8_lossy(&self.d[st..self.p - 1]).into_owned())
    }
}

fn pop_mark(st: &mut Vec<V>) -> Result<Vec<V>> {
    let i = st.iter().rposition(|v| matches!(v, V::Mark)).context("MARK 없음")?;
    let items = st.split_off(i + 1);
    st.pop();
    Ok(items)
}

fn unpickle(d: &[u8]) -> Result<V> {
    let mut r = Pk { d, p: 0 };
    let mut st: Vec<V> = Vec::new();
    let mut memo: BTreeMap<u64, V> = BTreeMap::new();
    loop {
        let op = r.u8()?;
        match op {
            0x80 => {
                r.u8()?;
            } // PROTO
            0x95 => {
                r.le(8)?;
            } // FRAME
            b'.' => return st.pop().context("빈 스택"),
            b'(' => st.push(V::Mark),
            b')' => st.push(V::Tuple(vec![])),
            b']' => st.push(V::List(Rc::new(RefCell::new(vec![])))),
            b'}' => st.push(V::Dict(Rc::new(RefCell::new(vec![])))),
            0x8f => st.push(V::List(Rc::new(RefCell::new(vec![])))), // EMPTY_SET (집합도 목록으로)
            0x90 => {
                // ADDITEMS
                let items = pop_mark(&mut st)?;
                if let Some(V::List(l)) = st.last() {
                    l.borrow_mut().extend(items);
                }
            }
            0x91 => {
                let items = pop_mark(&mut st)?;
                st.push(V::Tuple(items));
            } // FROZENSET
            b'N' => st.push(V::None),
            0x88 => st.push(V::Bool(true)),
            0x89 => st.push(V::Bool(false)),
            b'J' => st.push(V::Int(r.le(4)? as u32 as i32 as i64)),
            b'K' => st.push(V::Int(r.u8()? as i64)),
            b'M' => st.push(V::Int(r.le(2)? as i64)),
            0x8a => {
                // LONG1
                let n = r.u8()? as usize;
                let s = r.take(n)?;
                let mut v: i64 = 0;
                for (i, b) in s.iter().enumerate().take(8) {
                    v |= (*b as i64) << (8 * i);
                }
                if n > 0 && n < 8 && s[n - 1] & 0x80 != 0 {
                    v -= 1i64 << (8 * n);
                }
                st.push(V::Int(v));
            }
            b'G' => {
                let s = r.take(8)?;
                st.push(V::Float(f64::from_be_bytes(s.try_into().unwrap())));
            }
            b'X' => {
                let n = r.le(4)? as usize;
                st.push(V::Str(String::from_utf8_lossy(r.take(n)?).into_owned()));
            }
            0x8c => {
                let n = r.u8()? as usize;
                st.push(V::Str(String::from_utf8_lossy(r.take(n)?).into_owned()));
            }
            0x8d => {
                let n = r.le(8)? as usize;
                st.push(V::Str(String::from_utf8_lossy(r.take(n)?).into_owned()));
            }
            b'T' => {
                let n = r.le(4)? as usize;
                st.push(V::Str(String::from_utf8_lossy(r.take(n)?).into_owned()));
            }
            b'U' => {
                let n = r.u8()? as usize;
                st.push(V::Str(String::from_utf8_lossy(r.take(n)?).into_owned()));
            }
            b'B' => {
                let n = r.le(4)? as usize;
                st.push(V::Bytes(r.take(n)?.to_vec()));
            }
            b'C' => {
                let n = r.u8()? as usize;
                st.push(V::Bytes(r.take(n)?.to_vec()));
            }
            0x8e => {
                let n = r.le(8)? as usize;
                st.push(V::Bytes(r.take(n)?.to_vec()));
            }
            b'c' => {
                let m = r.line()?;
                let n = r.line()?;
                st.push(V::Global(m, n));
            }
            0x93 => {
                // STACK_GLOBAL
                let n = st.pop().context("스택")?;
                let m = st.pop().context("스택")?;
                match (m, n) {
                    (V::Str(m), V::Str(n)) => st.push(V::Global(m, n)),
                    _ => bail!("STACK_GLOBAL 인자"),
                }
            }
            b'q' => {
                let k = r.u8()? as u64;
                memo.insert(k, st.last().context("스택")?.clone());
            }
            b'r' => {
                let k = r.le(4)?;
                memo.insert(k, st.last().context("스택")?.clone());
            }
            0x94 => {
                let k = memo.len() as u64;
                memo.insert(k, st.last().context("스택")?.clone());
            } // MEMOIZE
            b'h' => {
                let k = r.u8()? as u64;
                st.push(memo.get(&k).context("memo")?.clone());
            }
            b'j' => {
                let k = r.le(4)?;
                st.push(memo.get(&k).context("memo")?.clone());
            }
            b't' => {
                let items = pop_mark(&mut st)?;
                st.push(V::Tuple(items));
            }
            0x85 => {
                let a = st.pop().context("스택")?;
                st.push(V::Tuple(vec![a]));
            }
            0x86 => {
                let b = st.pop().context("스택")?;
                let a = st.pop().context("스택")?;
                st.push(V::Tuple(vec![a, b]));
            }
            0x87 => {
                let c = st.pop().context("스택")?;
                let b = st.pop().context("스택")?;
                let a = st.pop().context("스택")?;
                st.push(V::Tuple(vec![a, b, c]));
            }
            b'l' => {
                let items = pop_mark(&mut st)?;
                st.push(V::List(Rc::new(RefCell::new(items))));
            }
            b'd' => {
                let items = pop_mark(&mut st)?;
                let kv = items.chunks(2).map(|c| (c[0].clone(), c[1].clone())).collect();
                st.push(V::Dict(Rc::new(RefCell::new(kv))));
            }
            b'a' => {
                let v = st.pop().context("스택")?;
                if let Some(V::List(l)) = st.last() {
                    l.borrow_mut().push(v);
                }
            }
            b'e' => {
                let items = pop_mark(&mut st)?;
                if let Some(V::List(l)) = st.last() {
                    l.borrow_mut().extend(items);
                }
            }
            b's' => {
                let v = st.pop().context("스택")?;
                let k = st.pop().context("스택")?;
                if let Some(V::Dict(dd)) = st.last() {
                    dd.borrow_mut().push((k, v));
                }
            }
            b'u' => {
                let items = pop_mark(&mut st)?;
                if let Some(V::Dict(dd)) = st.last() {
                    for c in items.chunks(2) {
                        dd.borrow_mut().push((c[0].clone(), c[1].clone()));
                    }
                }
            }
            b'Q' => {
                let p = st.pop().context("스택")?;
                st.push(V::PersId(Box::new(p)));
            } // BINPERSID
            b'R' => {
                let args = st.pop().context("스택")?;
                let f = st.pop().context("스택")?;
                st.push(obj(f, args));
            }
            0x81 => {
                let args = st.pop().context("스택")?;
                let c = st.pop().context("스택")?;
                st.push(obj(c, args));
            } // NEWOBJ
            0x92 => {
                let _kw = st.pop().context("스택")?;
                let args = st.pop().context("스택")?;
                let c = st.pop().context("스택")?;
                st.push(obj(c, args));
            } // NEWOBJ_EX
            b'b' => {
                let state = st.pop().context("스택")?;
                match st.last() {
                    Some(V::Obj(o)) => o.borrow_mut().state = Some(state),
                    Some(V::Dict(dd)) => {
                        if let V::Dict(s2) = state {
                            dd.borrow_mut().extend(s2.borrow().iter().cloned());
                        }
                    }
                    _ => {}
                }
            }
            b'0' => {
                st.pop();
            } // POP
            b'1' => {
                pop_mark(&mut st)?;
            } // POP_MARK
            b'2' => {
                let v = st.last().context("스택")?.clone();
                st.push(v);
            } // DUP
            _ => bail!("모르는 pickle 연산 0x{:02x} (위치 {})", op, r.p - 1),
        }
    }
}

// ---- 텐서 ----
struct TInfo {
    key: String,
    dtype: String,
    offset: usize,
    size: Vec<usize>,
    stride: Vec<usize>,
}

fn as_int(v: &V) -> Option<i64> {
    if let V::Int(i) = v { Some(*i) } else { None }
}
fn ints(v: &V) -> Vec<usize> {
    match v {
        V::Tuple(t) => t.iter().filter_map(as_int).map(|x| x as usize).collect(),
        _ => vec![],
    }
}

fn tensor_of(v: &V) -> Option<TInfo> {
    let V::Obj(o) = v else { return None };
    let o = o.borrow();
    let V::Global(m, n) = &o.callable else { return None };
    let V::Tuple(a) = &o.args else { return None };
    if m == "torch._utils" && n == "_rebuild_parameter" {
        return tensor_of(a.first()?);
    }
    if m == "torch._utils" && (n == "_rebuild_tensor_v2" || n == "_rebuild_tensor") {
        let V::PersId(p) = a.first()? else { return None };
        let V::Tuple(pt) = p.as_ref() else { return None };
        // ('storage', <StorageType>, key, location, numel)
        let dtype = match pt.get(1)? {
            V::Global(_, s) => s.clone(),
            _ => return None,
        };
        let key = match pt.get(2)? {
            V::Str(s) => s.clone(),
            _ => return None,
        };
        return Some(TInfo { key, dtype, offset: as_int(a.get(1)?)? as usize, size: ints(a.get(2)?), stride: ints(a.get(3)?) });
    }
    None
}

fn dict_get<'a>(d: &'a [(V, V)], k: &str) -> Option<&'a V> {
    d.iter().find(|(kk, _)| matches!(kk, V::Str(s) if s == k)).map(|(_, v)| v)
}
fn dict_items(v: &V) -> Vec<(String, V)> {
    match v {
        V::Dict(d) => d.borrow().iter().filter_map(|(k, v)| if let V::Str(s) = k { Some((s.clone(), v.clone())) } else { None }).collect(),
        // collections.OrderedDict 는 REDUCE 뒤 SETITEMS 로 채워진다 — 상태에 있지 않고 묶음 자체가 dict 처럼 쓰인다
        V::Obj(o) => {
            let o = o.borrow();
            if let Some(s) = &o.state {
                return dict_items(s);
            }
            vec![]
        }
        _ => vec![],
    }
}

// 객체 트리를 돌며 텐서를 이름과 함께 모은다
fn walk(v: &V, prefix: &str, out: &mut Vec<(String, TInfo)>, depth: usize) {
    if depth > 64 {
        return;
    }
    if let Some(t) = tensor_of(v) {
        out.push((prefix.trim_end_matches('.').to_string(), t));
        return;
    }
    match v {
        V::Obj(o) => {
            let st = o.borrow().state.clone();
            if let Some(V::Dict(d)) = &st {
                let d = d.borrow();
                let is_module = dict_get(&d, "_modules").is_some();
                if is_module {
                    for key in ["_parameters", "_buffers"] {
                        if let Some(pv) = dict_get(&d, key) {
                            for (k, t) in dict_items(pv) {
                                walk(&t, &format!("{prefix}{k}"), out, depth + 1);
                            }
                        }
                    }
                    if let Some(mv) = dict_get(&d, "_modules") {
                        for (k, m) in dict_items(mv) {
                            walk(&m, &format!("{prefix}{k}."), out, depth + 1);
                        }
                    }
                    return;
                }
            }
            if let Some(s) = st {
                walk(&s, prefix, out, depth + 1);
            }
        }
        V::Dict(_) => {
            for (k, x) in dict_items(v) {
                walk(&x, &format!("{prefix}{k}."), out, depth + 1);
            }
        }
        _ => {}
    }
}

fn elem(dtype: &str) -> Result<usize> {
    Ok(match dtype {
        "HalfStorage" | "BFloat16Storage" => 2,
        "FloatStorage" | "IntStorage" => 4,
        "DoubleStorage" | "LongStorage" => 8,
        "ByteStorage" | "BoolStorage" | "CharStorage" => 1,
        _ => bail!("모르는 저장소 {dtype}"),
    })
}
fn to_f32(dtype: &str, b: &[u8]) -> f64 {
    match dtype {
        "HalfStorage" => half::f16::from_le_bytes([b[0], b[1]]).to_f64(),
        "BFloat16Storage" => half::bf16::from_le_bytes([b[0], b[1]]).to_f64(),
        "FloatStorage" => f32::from_le_bytes(b.try_into().unwrap()) as f64,
        "DoubleStorage" => f64::from_le_bytes(b.try_into().unwrap()),
        "LongStorage" => i64::from_le_bytes(b.try_into().unwrap()) as f64,
        "IntStorage" => i32::from_le_bytes(b.try_into().unwrap()) as f64,
        "ByteStorage" | "BoolStorage" => b[0] as f64,
        "CharStorage" => b[0] as i8 as f64,
        _ => 0.0,
    }
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        bail!("사용: ptconv <in.pt> <out.safetensors> [키] [--list]");
    }
    let key = a.get(3).filter(|s| !s.starts_with("--")).cloned();
    let list = a.iter().any(|s| s == "--list");
    let f = std::fs::File::open(&a[1])?;
    let mut z = zip::ZipArchive::new(f)?;
    let names: Vec<String> = (0..z.len()).map(|i| z.by_index(i).unwrap().name().to_string()).collect();
    let pkl = names.iter().find(|n| n.ends_with("data.pkl")).context("data.pkl 없음")?.clone();
    let root = pkl.trim_end_matches("data.pkl").to_string();
    let mut buf = Vec::new();
    z.by_name(&pkl)?.read_to_end(&mut buf)?;
    let top = unpickle(&buf)?;
    let start = match &key {
        Some(k) => dict_items(&top).into_iter().find(|(kk, _)| kk == k).map(|(_, v)| v).ok_or_else(|| anyhow!("키 {k} 없음"))?,
        None => top,
    };
    let mut ts = Vec::new();
    walk(&start, "", &mut ts, 0);
    let mut data: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    let mut cache: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for (name, t) in &ts {
        if !cache.contains_key(&t.key) {
            let mut b = Vec::new();
            z.by_name(&format!("{root}data/{}", t.key))?.read_to_end(&mut b)?;
            cache.insert(t.key.clone(), b);
        }
        let raw = &cache[&t.key];
        let es = elem(&t.dtype)?;
        let n: usize = t.size.iter().product();
        let mut vals = Vec::with_capacity(n * 4);
        // stride 를 따라 읽는다(뷰도 맞게)
        let nd = t.size.len();
        let mut idx = vec![0usize; nd];
        for _ in 0..n {
            let off = t.offset + idx.iter().zip(&t.stride).map(|(i, s)| i * s).sum::<usize>();
            let v = to_f32(&t.dtype, &raw[off * es..off * es + es]) as f32;
            vals.extend_from_slice(&v.to_le_bytes());
            for d in (0..nd).rev() {
                idx[d] += 1;
                if idx[d] < t.size[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        if list {
            println!("{name} {:?} {}", t.size, t.dtype);
        }
        data.push((name.clone(), t.size.clone(), vals));
    }
    let views: Vec<(String, safetensors::tensor::TensorView)> = data
        .iter()
        .map(|(n, s, v)| (n.clone(), safetensors::tensor::TensorView::new(safetensors::Dtype::F32, s.clone(), v).unwrap()))
        .collect();
    safetensors::serialize_to_file(views, &None, std::path::Path::new(&a[2]))?;
    eprintln!("텐서 {} 개 → {}", data.len(), a[2]);
    Ok(())
}
