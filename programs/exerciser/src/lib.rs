//! Syscall exerciser. Instruction data: a sequence of 41-byte ops, each a tag byte and five
//! little-endian u64 arguments, passed to the syscall as given (raw VM addresses included).
//! Each syscall's return value is stored at output[512 + 8 * i]; account 0 is the output.
#![no_std]

extern "C" {
    fn abort() -> !;
    fn sol_panic_(file: *const u8, len: u64, line: u64, column: u64) -> !;
    fn sol_log_(p: *const u8, n: u64);
    fn sol_log_64_(a: u64, b: u64, c: u64, d: u64, e: u64);
    fn sol_log_compute_units_();
    fn sol_log_pubkey(p: *const u8);
    fn sol_log_data(p: *const u8, n: u64);
    fn sol_memcpy_(d: *mut u8, s: *const u8, n: u64);
    fn sol_memmove_(d: *mut u8, s: *const u8, n: u64);
    fn sol_memset_(d: *mut u8, c: u8, n: u64);
    fn sol_memcmp_(a: *const u8, b: *const u8, n: u64, r: *mut i32);
    fn sol_sha256(v: *const u8, n: u64, r: *mut u8) -> u64;
    fn sol_keccak256(v: *const u8, n: u64, r: *mut u8) -> u64;
    fn sol_set_return_data(p: *const u8, n: u64);
    fn sol_get_return_data(p: *mut u8, n: u64, id: *mut u8) -> u64;
    fn sol_get_stack_height() -> u64;
    fn sol_remaining_compute_units() -> u64;
}

unsafe fn rd(p: *const u8) -> u64 {
    let mut v = 0u64;
    for i in 0..8 {
        v |= (*p.add(i) as u64) << (8 * i);
    }
    v
}

unsafe fn wr(p: *mut u8, v: u64) {
    for i in 0..8 {
        *p.add(i) = (v >> (8 * i)) as u8;
    }
}

#[no_mangle]
pub unsafe extern "C" fn entrypoint(input: *mut u8) -> u64 {
    // one account, no duplicates: 8 (count) + 88 (header), then data, 10 KiB, alignment, rent epoch
    let dlen = rd(input.add(8 + 80)) as usize;
    let out = input.add(96);
    let mut p = 96 + dlen + 10240;
    p += (8 - p % 8) % 8;
    p += 8;
    let n = rd(input.add(p)) as usize;
    let data = input.add(p + 8);
    let mut i = 0;
    let mut k = 0;
    while i + 41 <= n {
        let t = *data.add(i);
        let a = |j: usize| rd(data.add(i + 1 + 8 * j));
        let (a0, a1, a2, a3, a4) = (a(0), a(1), a(2), a(3), a(4));
        let r: u64 = match t {
            0 => { sol_memmove_(a0 as *mut u8, a1 as *const u8, a2); 0 }
            1 => { sol_memcmp_(a0 as *const u8, a1 as *const u8, a2, a3 as *mut i32); 0 }
            2 => { sol_memcpy_(a0 as *mut u8, a1 as *const u8, a2); 0 }
            3 => { sol_memset_(a0 as *mut u8, a1 as u8, a2); 0 }
            4 => { sol_log_(a0 as *const u8, a1); 0 }
            5 => { sol_log_64_(a0, a1, a2, a3, a4); 0 }
            6 => { sol_log_compute_units_(); 0 }
            7 => sol_panic_(a0 as *const u8, a1, a2, a3),
            8 => sol_sha256(a0 as *const u8, a1, a2 as *mut u8),
            9 => sol_keccak256(a0 as *const u8, a1, a2 as *mut u8),
            10 => { sol_log_pubkey(a0 as *const u8); 0 }
            11 => { sol_log_data(a0 as *const u8, a1); 0 }
            12 => { sol_set_return_data(a0 as *const u8, a1); 0 }
            13 => sol_get_return_data(a0 as *mut u8, a1, a2 as *mut u8),
            14 => sol_get_stack_height(),
            15 => sol_remaining_compute_units(),
            16 => abort(),
            _ => 0,
        };
        if k < 64 && dlen >= 1024 {
            wr(out.add(512 + 8 * k), r);
        }
        k += 1;
        i += 41;
    }
    0
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    unsafe { abort() }
}
