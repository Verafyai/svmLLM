// Cross-check harness: runs programs on Anza's solana-sbpf interpreter (SBPF V0).
// stdin: one line per case: "<prog_hex> <mem_hex> <cu_limit>"
// stdout: "ok <r0> <count> <r0..r10 hex,comma> <mem_hex>" or "err <kind> <count>"
use solana_sbpf::{
    aligned_memory::AlignedMemory, ebpf, elf::Executable,
    memory_region::{MemoryMapping, MemoryRegion},
    program::{BuiltinProgram, FunctionRegistry, SBPFVersion},
    vm::{Config, ContextObject, EbpfVm},
};
use std::io::BufRead;
use std::sync::Arc;

struct Ctx { remaining: u64 }
impl ContextObject for Ctx {
    fn trace(&mut self, _s: [u64; 12]) {}
    fn consume(&mut self, a: u64) { self.remaining = self.remaining.saturating_sub(a); }
    fn get_remaining(&self) -> u64 { self.remaining }
}

fn unhex(s: &str) -> Vec<u8> {
    if s == "-" { return vec![]; }
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

fn main() {
    let config = Config { enable_stack_frame_gaps: false, ..Config::default() };
    let loader = Arc::new(BuiltinProgram::<Ctx>::new_loader(config));
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let parts: Vec<&str> = line.split_whitespace().collect();
        let prog = unhex(parts[0]);
        let mut mem = unhex(parts[1]);
        let limit: u64 = parts[2].parse().unwrap();
        let exe = match Executable::<Ctx>::from_text_bytes(&prog, loader.clone(), SBPFVersion::V0, FunctionRegistry::default()) {
            Ok(e) => e, Err(e) => { println!("err load:{:?} 0", e); continue; }
        };
        let mut ctx = Ctx { remaining: limit };
        let ver = exe.get_sbpf_version();
        let mut stack = AlignedMemory::<{ ebpf::HOST_ALIGN }>::zero_filled(exe.get_config().stack_size());
        let stack_len = stack.len();
        let mut heap = AlignedMemory::<{ ebpf::HOST_ALIGN }>::with_capacity(0);
        let regions = vec![
            exe.get_ro_region(),
            // toy model: only the top 128 bytes of the first frame are mapped (0x200000F80..0x200001000)
            MemoryRegion::new_writable(&mut stack.as_slice_mut()[0xF80..0x1000], ebpf::MM_STACK_START + 0xF80),
            MemoryRegion::new_writable(heap.as_slice_mut(), ebpf::MM_HEAP_START),
            MemoryRegion::new_writable(&mut mem, ebpf::MM_INPUT_START),
        ];
        let mm = MemoryMapping::new(regions, exe.get_config(), ver).unwrap();
        let mut vm = EbpfVm::new(loader.clone(), ver, &mut ctx, mm, stack_len);
        vm.registers[1] = ebpf::MM_INPUT_START;
        let (count, res) = vm.execute_program(&exe, true);
        let regs: Vec<String> = vm.registers[..11].iter().map(|r| format!("{:x}", r)).collect();
        drop(vm);
        match res {
            solana_sbpf::error::StableResult::Ok(v) => {
                let mh: String = mem.iter().map(|b| format!("{:02x}", b)).collect();
                println!("ok {} {} {} {}", v, count, regs.join(","), if mh.is_empty() { "-".into() } else { mh });
            }
            solana_sbpf::error::StableResult::Err(e) => {
                let k = format!("{:?}", e); let k = k.split(|c: char| c == '(' || c == ' ').next().unwrap().to_string();
                println!("err {} {}", k, count);
            }
        }
    }
}
