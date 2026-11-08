// Ground truth for the full-layout machine: Anza's solana-sbpf interpreter (SBPF V0) with the
// real memory map and default config: 64 frames of 4 KB (gaps on or off), 32 KB heap, the input
// region, and BPF-to-BPF calls through a function registry.
// stdin, one case per line:  <prog_hex> <input_hex|-> <cu_limit> <gaps 0|1> <key:pc,...|->
// stdout: "ok <r0> <count> <r0..r10 hex,comma> <fnv stack> <fnv heap> <fnv input>"
//      or "err <kind> <count>"
use solana_sbpf::{
    aligned_memory::AlignedMemory, ebpf, elf::Executable,
    memory_region::{MemoryMapping, MemoryRegion},
    program::{BuiltinProgram, FunctionRegistry, SBPFVersion},
    vm::{Config, ContextObject, EbpfVm},
};
use std::io::BufRead;
use std::sync::Arc;

struct Ctx { remaining: u64, last: [u64; 12] }
impl ContextObject for Ctx {
    fn trace(&mut self, s: [u64; 12]) { self.last = s; }
    fn consume(&mut self, a: u64) { self.remaining = self.remaining.saturating_sub(a); }
    fn get_remaining(&self) -> u64 { self.remaining }
}

fn unhex(s: &str) -> Vec<u8> {
    if s == "-" { return vec![]; }
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

fn fnv(b: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for x in b { h ^= *x as u64; h = h.wrapping_mul(0x100000001b3); }
    h
}

pub const HEAP: usize = 32 * 1024;

fn main() {
    let loaders: Vec<Arc<BuiltinProgram<Ctx>>> = [false, true].iter().map(|g| {
        Arc::new(BuiltinProgram::<Ctx>::new_loader(Config { enable_stack_frame_gaps: *g, enable_instruction_tracing: true, ..Config::default() }))
    }).collect();
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let p: Vec<&str> = line.split_whitespace().collect();
        let prog = unhex(p[0]);
        let mut input = unhex(p[1]);
        let limit: u64 = p[2].parse().unwrap();
        let loader = loaders[(p[3] == "1") as usize].clone();
        let mut fr = FunctionRegistry::<usize>::default();
        if p[4] != "-" {
            for kv in p[4].split(',') {
                let (k, v) = kv.split_once(':').unwrap();
                fr.register_function(k.parse().unwrap(), format!("f{}", k), v.parse().unwrap()).unwrap();
            }
        }
        let exe = match Executable::<Ctx>::from_text_bytes(&prog, loader.clone(), SBPFVersion::V0, fr) {
            Ok(e) => e, Err(e) => { println!("err load:{:?} 0", e); continue; }
        };
        let mut ctx = Ctx { remaining: limit, last: [0; 12] };
        let ver = exe.get_sbpf_version();
        let cfg = exe.get_config();
        let mut stack = AlignedMemory::<{ ebpf::HOST_ALIGN }>::zero_filled(cfg.stack_size());
        let stack_len = stack.len();
        let mut heap = AlignedMemory::<{ ebpf::HOST_ALIGN }>::zero_filled(HEAP);
        let gap = if cfg.enable_stack_frame_gaps { cfg.stack_frame_size as u64 } else { 0 };
        let regions = vec![
            exe.get_ro_region(),
            MemoryRegion::new_writable_gapped(stack.as_slice_mut(), ebpf::MM_STACK_START, gap),
            MemoryRegion::new_writable(heap.as_slice_mut(), ebpf::MM_HEAP_START),
            MemoryRegion::new_writable(&mut input, ebpf::MM_INPUT_START),
        ];
        let mm = MemoryMapping::new(regions, cfg, ver).unwrap();
        let mut vm = EbpfVm::new(loader.clone(), ver, &mut ctx, mm, stack_len);
        vm.registers[1] = ebpf::MM_INPUT_START;
        let (count, res) = vm.execute_program(&exe, true);
        drop(vm);
        let regs: Vec<String> = ctx.last[..11].iter().map(|r| format!("{:x}", r)).collect();
        match res {
            solana_sbpf::error::StableResult::Ok(v) => println!("ok {} {} {} {:x} {:x} {:x}", v, count, regs.join(","),
                fnv(stack.as_slice()), fnv(heap.as_slice()), fnv(&input)),
            solana_sbpf::error::StableResult::Err(e) => {
                let k = format!("{:?}", e); let k = k.split(|c: char| c == '(' || c == ' ').next().unwrap().to_string();
                println!("err {} {}", k, count);
            }
        }
    }
}
