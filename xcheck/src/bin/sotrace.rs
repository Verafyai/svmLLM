// Instruction tracer for compiled Solana programs (.so, SBPF v0), on Anza's solana-sbpf.
//
// Used to find the instruction subset the TinySVM settlement programs actually
// execute (svllm/settlement/subset.py). It loads the ELF, maps a serialized
// loader input buffer at MM_INPUT_START, runs the interpreter with instruction
// tracing on, and counts every executed opcode and syscall.
//
// stdin: one line per case: "<so_path> <input_hex> <cu_limit>"
// stdout: "ok <r0> <count> <op_hex:n,...> <syscall:n,...|-> <input_hex_after>"
//      or "err <kind> <count> <op_hex:n,...> <syscall:n,...|->"
use solana_sbpf::{
    aligned_memory::AlignedMemory,
    declare_builtin_function, ebpf,
    elf::Executable,
    error::StableResult,
    memory_region::{AccessType, MemoryMapping, MemoryRegion},
    program::BuiltinProgram,
    vm::{Config, ContextObject, EbpfVm},
};
use std::collections::BTreeMap;
use std::io::BufRead;
use std::sync::Arc;

#[derive(Default)]
struct Ctx {
    remaining: u64,
    pcs: Vec<u64>,
    syscalls: BTreeMap<&'static str, u64>,
}
impl ContextObject for Ctx {
    fn trace(&mut self, s: [u64; 12]) {
        self.pcs.push(s[11]);
    }
    fn consume(&mut self, a: u64) {
        self.remaining = self.remaining.saturating_sub(a);
    }
    fn get_remaining(&self) -> u64 {
        self.remaining
    }
}

type Err = Box<dyn std::error::Error>;

fn host(mm: &MemoryMapping, t: AccessType, addr: u64, len: u64) -> Result<u64, Err> {
    match mm.map(t, addr, len) {
        StableResult::Ok(h) => Ok(h),
        StableResult::Err(e) => Err(Box::new(e)),
    }
}

declare_builtin_function!(
    SolLog,
    fn rust(ctx: &mut Ctx, _a: u64, _b: u64, _c: u64, _d: u64, _e: u64, _m: &mut MemoryMapping,) -> Result<u64, Err> {
        *ctx.syscalls.entry("sol_log_").or_default() += 1;
        Ok(0)
    }
);
declare_builtin_function!(
    Abort,
    fn rust(ctx: &mut Ctx, _a: u64, _b: u64, _c: u64, _d: u64, _e: u64, _m: &mut MemoryMapping,) -> Result<u64, Err> {
        *ctx.syscalls.entry("abort").or_default() += 1;
        Err("abort".into())
    }
);
declare_builtin_function!(
    SolMemcpy,
    fn rust(ctx: &mut Ctx, dst: u64, src: u64, n: u64, _d: u64, _e: u64, m: &mut MemoryMapping,) -> Result<u64, Err> {
        *ctx.syscalls.entry("sol_memcpy_").or_default() += 1;
        if n > 0 {
            let s = host(m, AccessType::Load, src, n)?;
            let d = host(m, AccessType::Store, dst, n)?;
            unsafe { std::ptr::copy(s as *const u8, d as *mut u8, n as usize) };
        }
        Ok(0)
    }
);
declare_builtin_function!(
    SolMemset,
    fn rust(ctx: &mut Ctx, dst: u64, c: u64, n: u64, _d: u64, _e: u64, m: &mut MemoryMapping,) -> Result<u64, Err> {
        *ctx.syscalls.entry("sol_memset_").or_default() += 1;
        if n > 0 {
            let d = host(m, AccessType::Store, dst, n)?;
            unsafe { std::ptr::write_bytes(d as *mut u8, c as u8, n as usize) };
        }
        Ok(0)
    }
);

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

fn main() {
    let config = Config { enable_instruction_tracing: true, ..Config::default() };
    let mut loader = BuiltinProgram::<Ctx>::new_loader(config);
    loader.register_function("sol_log_", SolLog::vm).unwrap();
    loader.register_function("abort", Abort::vm).unwrap();
    loader.register_function("sol_memcpy_", SolMemcpy::vm).unwrap();
    loader.register_function("sol_memset_", SolMemset::vm).unwrap();
    let loader = Arc::new(loader);
    let mut cache: BTreeMap<String, Executable<Ctx>> = BTreeMap::new();
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let p: Vec<&str> = line.split_whitespace().collect();
        let exe = cache.entry(p[0].to_string()).or_insert_with(|| {
            let bytes = std::fs::read(p[0]).expect("read .so");
            Executable::<Ctx>::load(&bytes, loader.clone()).expect("load ELF")
        });
        let mut input = unhex(p[1]);
        let limit: u64 = p[2].parse().unwrap();
        let mut ctx = Ctx { remaining: limit, ..Ctx::default() };
        let ver = exe.get_sbpf_version();
        let cfg = exe.get_config();
        let mut stack = AlignedMemory::<{ ebpf::HOST_ALIGN }>::zero_filled(cfg.stack_size());
        let stack_len = stack.len();
        let mut heap = AlignedMemory::<{ ebpf::HOST_ALIGN }>::zero_filled(32 * 1024);
        let gap = if !ver.dynamic_stack_frames() && cfg.enable_stack_frame_gaps { cfg.stack_frame_size as u64 } else { 0 };
        let regions = vec![
            exe.get_ro_region(),
            MemoryRegion::new_writable_gapped(stack.as_slice_mut(), ebpf::MM_STACK_START, gap),
            MemoryRegion::new_writable(heap.as_slice_mut(), ebpf::MM_HEAP_START),
            MemoryRegion::new_writable(&mut input, ebpf::MM_INPUT_START),
        ];
        let mm = MemoryMapping::new(regions, cfg, ver).unwrap();
        let mut vm = EbpfVm::new(loader.clone(), ver, &mut ctx, mm, stack_len);
        vm.registers[1] = ebpf::MM_INPUT_START;
        let (count, res) = vm.execute_program(exe, true);
        drop(vm);
        let (_, text) = exe.get_text_bytes();
        // Key: the opcode byte; a call (0x85) also carries its src field (0 syscall, 1 internal) as 0x1_85.
        let mut ops: BTreeMap<u16, u64> = BTreeMap::new();
        for pc in &ctx.pcs {
            let i = (*pc as usize) * 8;
            let op = text[i] as u16;
            let key = if op == 0x85 { op | (((text[i + 1] >> 4) as u16) << 8) } else { op };
            *ops.entry(key).or_default() += 1;
        }
        let ops_s: Vec<String> = ops.iter().map(|(o, n)| format!("{:x}:{}", o, n)).collect();
        let sys_s: Vec<String> = ctx.syscalls.iter().map(|(k, n)| format!("{}:{}", k, n)).collect();
        let sys = if sys_s.is_empty() { "-".to_string() } else { sys_s.join(",") };
        match res {
            StableResult::Ok(v) => {
                let mh: String = input.iter().map(|b| format!("{:02x}", b)).collect();
                println!("ok {} {} {} {} {}", v, count, ops_s.join(","), sys, mh);
            }
            StableResult::Err(e) => {
                let k = format!("{:?}", e);
                let k = k.split(|c: char| c == '(' || c == ' ').next().unwrap().to_string();
                println!("err {} {} {} {}", k, count, ops_s.join(","), sys);
            }
        }
    }
}
