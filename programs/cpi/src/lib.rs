//! CPI exerciser. Parses the loader input (duplicates included), then performs the invocation
//! its instruction data describes, through sol_invoke_signed_c:
//!   mode u8 (bit 0: write 0xAB into account 0's first data byte before the call;
//!            bit 1: move 1 lamport from account 0 to account 1 before the call)
//!   program u8 (account index whose key is the callee), nmetas u8, nmetas x (index, signer, writable),
//!   ninfos u8, ninfos x index, data_len u16, data, nsigners u8, per signer: nseeds u8, per seed: len u8, bytes
//! Returns the syscall's result (0 on success).
#![no_std]

extern "C" {
    fn sol_invoke_signed_c(ix: *const u8, infos: *const u8, ninfos: u64, seeds: *const u8, nseeds: u64) -> u64;
    fn abort() -> !;
}

#[repr(C)]
struct SolInstruction { program_id: *const u8, accounts: *const SolAccountMeta, accounts_len: u64, data: *const u8, data_len: u64 }
#[repr(C)]
#[derive(Clone, Copy)]
struct SolAccountMeta { pubkey: *const u8, is_writable: bool, is_signer: bool }
#[repr(C)]
#[derive(Clone, Copy)]
struct SolAccountInfo { key: *const u8, lamports: *mut u64, data_len: u64, data: *mut u8, owner: *const u8, rent_epoch: u64, is_signer: bool, is_writable: bool, executable: bool }
#[repr(C)]
#[derive(Clone, Copy)]
struct SolBytes { addr: *const u8, len: u64 }
#[repr(C)]
#[derive(Clone, Copy)]
struct SolSignerSeeds { addr: *const SolBytes, len: u64 }

const MAX: usize = 16;

unsafe fn rd(p: *const u8) -> u64 {
    let mut v = 0u64;
    for i in 0..8 { v |= (*p.add(i) as u64) << (8 * i); }
    v
}

#[no_mangle]
pub unsafe extern "C" fn entrypoint(input: *mut u8) -> u64 {
    let n = rd(input) as usize;
    if n > MAX { return 1000; }
    let empty = SolAccountInfo { key: core::ptr::null(), lamports: core::ptr::null_mut(), data_len: 0, data: core::ptr::null_mut(), owner: core::ptr::null(), rent_epoch: 0, is_signer: false, is_writable: false, executable: false };
    let mut infos = [empty; MAX];
    let mut p = 8usize;
    for i in 0..n {
        let dup = *input.add(p);
        if dup != 0xff {
            infos[i] = infos[dup as usize];
            p += 8;
            continue;
        }
        let a = &mut infos[i];
        a.is_signer = *input.add(p + 1) != 0;
        a.is_writable = *input.add(p + 2) != 0;
        a.executable = *input.add(p + 3) != 0;
        a.key = input.add(p + 8);
        a.owner = input.add(p + 40);
        a.lamports = input.add(p + 72) as *mut u64;
        a.data_len = rd(input.add(p + 80));
        a.data = input.add(p + 88);
        let mut q = p + 88 + a.data_len as usize + 10240;
        q += (8 - q % 8) % 8;
        a.rent_epoch = rd(input.add(q));
        p = q + 8;
    }
    let dlen = rd(input.add(p)) as usize;
    let d = input.add(p + 8);
    let mut k = 0usize;
    let mode = *d; k += 1;
    let prog = *d.add(k) as usize; k += 1;
    let nm = *d.add(k) as usize; k += 1;
    let mut metas = [SolAccountMeta { pubkey: core::ptr::null(), is_writable: false, is_signer: false }; MAX];
    for j in 0..nm.min(MAX) {
        let idx = *d.add(k) as usize;
        metas[j] = SolAccountMeta { pubkey: infos[idx % MAX].key, is_signer: *d.add(k + 1) != 0, is_writable: *d.add(k + 2) != 0 };
        k += 3;
    }
    let ni = *d.add(k) as usize; k += 1;
    let mut pass = [empty; MAX];
    for j in 0..ni.min(MAX) {
        pass[j] = infos[*d.add(k) as usize % MAX];
        k += 1;
    }
    let ilen = (*d.add(k) as usize) | ((*d.add(k + 1) as usize) << 8); k += 2;
    let idata = d.add(k); k += ilen;
    let ns = *d.add(k) as usize; k += 1;
    let mut seeds = [[SolBytes { addr: core::ptr::null(), len: 0 }; 4]; 4];
    let mut signers = [SolSignerSeeds { addr: core::ptr::null(), len: 0 }; 4];
    for s in 0..ns.min(4) {
        let c = *d.add(k) as usize; k += 1;
        for t in 0..c.min(4) {
            let l = *d.add(k) as usize; k += 1;
            seeds[s][t] = SolBytes { addr: d.add(k), len: l as u64 };
            k += l;
        }
        signers[s] = SolSignerSeeds { addr: seeds[s].as_ptr(), len: c.min(4) as u64 };
    }
    if k > dlen { return 1001; }
    if mode & 1 != 0 && n > 0 && infos[0].data_len > 0 { *infos[0].data = 0xAB; }
    if mode & 2 != 0 && n > 1 { *infos[0].lamports -= 1; *infos[1].lamports += 1; }
    let ix = SolInstruction { program_id: infos[prog % MAX].key, accounts: metas.as_ptr(), accounts_len: nm as u64, data: idata, data_len: ilen as u64 };
    let r = sol_invoke_signed_c(&ix as *const _ as *const u8, pass.as_ptr() as *const u8, ni as u64, signers.as_ptr() as *const u8, ns as u64);
    if mode & 4 != 0 { abort(); }
    r
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { unsafe { abort() } }
