// Load an ELF exactly as Agave 4.3.2 does (its syscall registry and VM config, mainnet features)
// and print the loaded image the VM runs: relocated text, the read-only region, the entry pc,
// the function registry and the syscall registry. JSON on stdout.
use litesvm::LiteSVM;
use serde_json::json;
use solana_program_runtime::{execution_budget::SVMTransactionExecutionBudget, invoke_context::InvokeContext};
use solana_sbpf::elf::Executable;

fn main() {
    let path = std::env::args().nth(1).expect("usage: elfimage <program.so>");
    let bytes = std::fs::read(&path).unwrap();
    let fs = LiteSVM::mainnet_feature_set().runtime_features();
    let env = solana_syscalls::create_program_runtime_environment(&fs, &SVMTransactionExecutionBudget::new_with_defaults(false), false, false).unwrap();
    let loader: std::sync::Arc<solana_sbpf::program::BuiltinProgram<InvokeContext<'static, 'static>>> = (*env).clone();
    let exe = Executable::<InvokeContext>::load(&bytes, loader).unwrap();
    let (tv, text) = exe.get_text_bytes();
    let ro = exe.get_ro_region();
    let funcs: Vec<_> = exe.get_function_registry().iter().map(|(k, (_n, pc))| json!([k, pc])).collect();
    let sys: Vec<_> = exe.get_loader().get_function_registry().iter().map(|(k, (n, _f))| json!([k, String::from_utf8_lossy(n)])).collect();
    let cfg = exe.get_config();
    println!("{}", json!({
        "version": format!("{:?}", exe.get_sbpf_version()), "text_vaddr": tv, "text": hex::encode(text),
        "ro_vaddr": ro.vm_addr_range().start, "ro": hex::encode(exe.get_ro_section()), "entry": exe.get_entrypoint_instruction_offset(),
        "funcs": funcs, "syscalls": sys,
        "config": {"gaps": cfg.enable_stack_frame_gaps, "aligned": cfg.aligned_memory_mapping, "max_call_depth": cfg.max_call_depth, "frame": cfg.stack_frame_size}
    }));
}
